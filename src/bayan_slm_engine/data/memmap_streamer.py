"""M1.4: 16-bit zero-copy binary packer & memory-mapped streamer.

SSOT: docs/BLUEPRINT.md §3 steps 4-5; docs/EXECUTION_ROADMAP.md M1.4.

Packs normalized SADA22 text rows + synthetic multi-turn JSON pairs into a
contiguous ``uint16`` binary file (zero host-RAM overhead via ``np.memmap``),
sidecar ``.index.npz`` (starts/lengths/kind), and ``.meta.json`` (vocab size,
tokenizer blake2b hash, totals). A hard zero-OOV gate (M1.1 BPE Coverage Rule)
rejects synthetic rows containing Arabic byte-fallback tokens before packing.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from bayan_slm_engine.data.distillation import read_dialogues_jsonl
from bayan_slm_engine.tokenizer.bpe_trainer import is_char_fallback
from bayan_slm_engine.tokenizer.normalizer import ArabicNormalizer

# Sidecar file suffixes derived from the output .bin path.
INDEX_SUFFIX = ".index.npz"
META_SUFFIX = ".meta.json"

# uint16 ceiling: V = 16,000 < 65,535 (BLUEPRINT §3 token math).
MAX_UINT16 = 65_535

# <unk> sentinel id (BPE special tokens are assigned ids 0..4 in order).
UNK_ID = 0


class PackKind(IntEnum):
    """Row provenance encoded in the sidecar index (u8)."""

    RAW = 0  # SADA22 ``text`` rows (or plain-text lines)
    SYNTHETIC = 1  # M1.3 distilled multi-turn JSON pairs


class PackConfig(BaseModel):
    """Pydantic v2 contract for a packing run (M1.4 Step A)."""

    input_dir: Path  # directory of UTF-8 .txt files; one SADA22 ``text`` row per line
    tokenizer_path: Path  # frozen M1.1 16k BPE tokenizer.json (required)
    output_bin: Path  # target flat uint16 binary
    synthetic_jsonl: Path | None = None  # optional M1.3 dialogues.jsonl (user/assistant/dialect)
    pad_id: int = Field(default=0, ge=0, le=MAX_UINT16)  # padding token for seq_len packing


class PackReport(BaseModel):
    """Machine-readable outcome of a packing run (atomic-commit contract)."""

    output_bin: str
    index_path: str
    meta_path: str
    total_tokens: int = Field(ge=0)
    rows_raw: int = Field(ge=0)
    rows_synthetic: int = Field(ge=0)
    oov_errors: int = Field(ge=0)


class BinaryPackingError(Exception):
    """Fatal contract violation during packing (hard OOV gate, M1.1 rule)."""


class BinaryPacker:
    """Streams text/synthetic rows -> normalized ids -> flat uint16 + sidecars.

    Contract: deterministic row order (resume-offset semantics), atomic writes
    (``.tmp`` + ``os.rename``), and a hard zero-OOV gate on Arabic byte-fallback
    tokens for synthetic pairs (clitic allowlist exempt, ``ا`` is a fallback).
    """

    def __init__(self, config: PackConfig) -> None:
        self.config = config
        self._tokenizer = Tokenizer.from_file(str(config.tokenizer_path))
        self._normalizer = ArabicNormalizer()
        # Token-id -> surface-string map (decode contract for OOV diagnostics).
        self._id_to_token = {v: k for k, v in self._tokenizer.get_vocab().items()}

    def pack(self) -> PackReport:
        """Execute the full packing run; writes .bin + sidecars atomically."""
        cfg = self.config
        cfg.output_bin.parent.mkdir(parents=True, exist_ok=True)

        starts: list[int] = []
        lengths: list[int] = []
        kinds: list[bytes] = []
        total = 0
        oov = 0
        tokenizer_hash = self._tokenizer_hash()

        # 1) Raw text rows: one normalized SADA22 ``text`` row per line.
        raw_rows = self._iter_raw_rows(cfg.input_dir)
        # 2) Synthetic multi-turn JSON pairs (optional).
        synthetic_rows = (
            self._iter_synthetic_rows(cfg.synthetic_jsonl) if cfg.synthetic_jsonl else ()
        )

        def write_row(ids: list[int], kind: PackKind) -> None:
            nonlocal total
            starts.append(total)
            lengths.append(len(ids))
            kinds.append(bytes([int(kind)]))
            total += len(ids)

        fd, tmp_path = tempfile.mkstemp(
            prefix=cfg.output_bin.name + ".", suffix=".tmp", dir=str(cfg.output_bin.parent)
        )
        try:
            with (
                os.fdopen(fd, "wb") as fh,
                open(cfg.output_bin.with_suffix(cfg.output_bin.suffix + INDEX_SUFFIX), "wb") as _,
            ):
                # Stream rows in deterministic order: all RAW then all SYNTHETIC.
                for row_idx, text in raw_rows:
                    ids = self._normalize_and_encode(text)
                    self._validate_oov(row_idx, ids)
                    write_row(ids, PackKind.RAW)
                    fh.write(np.asarray(ids, dtype="uint16").tobytes())
                for row_idx, text in synthetic_rows:
                    ids = self._normalize_and_encode(text)
                    self._validate_oov(row_idx, ids)
                    write_row(ids, PackKind.SYNTHETIC)
                    fh.write(np.asarray(ids, dtype="uint16").tobytes())
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, cfg.output_bin)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise

        # Sidecar index (npz) + meta (json) written atomically after the bin.
        index_path = cfg.output_bin.with_suffix(cfg.output_bin.suffix + INDEX_SUFFIX)
        meta_path = cfg.output_bin.with_suffix(cfg.output_bin.suffix + META_SUFFIX)
        self._write_index(index_path, starts, lengths, kinds)
        meta = {
            "vocab_size": self._tokenizer.get_vocab_size(),
            "tokenizer_hash": tokenizer_hash,
            "total_tokens": total,
            "rows_raw": len([k for k in kinds if k == bytes([int(PackKind.RAW)])]),
            "rows_synthetic": len([k for k in kinds if k == bytes([int(PackKind.SYNTHETIC)])]),
            "pad_id": cfg.pad_id,
        }
        tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
        tmp_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_meta, meta_path)

        return PackReport(
            output_bin=str(cfg.output_bin),
            index_path=str(index_path),
            meta_path=str(meta_path),
            total_tokens=total,
            rows_raw=meta["rows_raw"],
            rows_synthetic=meta["rows_synthetic"],
            oov_errors=oov,
        )

    def _tokenizer_hash(self) -> str:
        """Stable blake2b hash of the frozen tokenizer JSON (meta provenance)."""
        digest = hashlib.blake2b(self.config.tokenizer_path.read_bytes(), digest_size=8).hexdigest()
        return digest

    def _iter_raw_rows(self, input_dir: Path) -> Iterator[tuple[int, str]]:
        """Yield (row_idx, text) for every non-empty line across *.txt files.

        Row order is deterministic: sorted file names, in-file line order — this
        is the resume-offset contract M4.2 relies on.
        """
        row_idx = 0
        for path in sorted(input_dir.glob("*.txt")):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    yield row_idx, line
                    row_idx += 1

    def _iter_synthetic_rows(self, jsonl_path: Path) -> Iterator[tuple[int, str]]:
        """Yield (row_idx, text) for M1.3 dialogues: ``user`` + ``assistant`` turns."""
        for row_idx, dialogue in enumerate(read_dialogues_jsonl(jsonl_path)):
            yield row_idx, f"{dialogue.user} {dialogue.assistant}"

    def _normalize_and_encode(self, text: str) -> list[int]:
        """Normalize (M1.1) then encode with the frozen 16k tokenizer."""
        normalized = self._normalizer.normalize(text)
        encoding = self._tokenizer.encode(normalized)
        return list(encoding.ids)

    def _validate_oov(self, row_idx: int, ids: Sequence[int]) -> None:
        """Hard zero-OOV gate: raise BinaryPackingError on any OOV token id.

        M1.1 BPE Coverage Rule: synthetic pairs must encode with zero OOV
        errors against the frozen 16k vocabulary. Deterministic checks:
        1. ``token_id`` outside ``[0, vocab_size)`` -> real OOV (hard cap).
        2. ``token_id == UNK_ID`` (``<unk>`` sentinel) -> unknown-token path.
        3. ``is_char_fallback(rendered_token)`` -> raw single-byte token =
           genuine byte-level fallback (BLUEPRINT §2 semantics; allowlisted
           clitics exempt). With ByteLevel BPE, Arabic letters render as
           byte-PAIR tokens (e.g. ``Ø§`` -> ``ا``, len 2) which are normal
           merges, NOT fallbacks; only a lone byte (len 1, e.g. ``Ø``) means
           a character was uncovered by the corpus merges.
        """
        vocab_size = self._tokenizer.get_vocab_size()
        for token_id in ids:
            if token_id < 0 or token_id >= vocab_size:
                raise BinaryPackingError(
                    f"OOV token id {token_id} outside frozen vocab [0, {vocab_size}) "
                    f"in row {row_idx}"
                )
            if token_id == UNK_ID:
                raise BinaryPackingError(f"OOV <unk> sentinel id in row {row_idx}")
            token = self._id_to_token.get(token_id)
            if token is not None and is_char_fallback(token):
                raise BinaryPackingError(
                    f"OOV byte-fallback token {token!r} (id={token_id}) in row {row_idx}"
                )

    @staticmethod
    def _write_index(
        path: Path,
        starts: list[int],
        lengths: list[int],
        kinds: list[bytes],
    ) -> None:
        """Write the sidecar .index.npz atomically (.tmp + os.replace).

        ``np.savez_compressed`` appends ``.npz`` to a filename argument, so we
        pass an open file handle (no suffix mangling) and rename manually.
        """
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as handle:
            np.savez_compressed(
                handle,
                starts=np.asarray(starts, dtype="uint64"),
                lengths=np.asarray(lengths, dtype="uint64"),
                kind=np.asarray([k[0] for k in kinds], dtype="uint8"),
            )
        os.replace(tmp, path)


@dataclass(frozen=True)
class _RowRecord:
    """Indexed row: byte offset, token length, provenance kind."""

    offset: int
    length: int
    kind: PackKind


def _zero_copy_view(arr: np.ndarray) -> torch.Tensor:
    """Zero-copy ``torch.from_numpy`` with the benign non-writable warning off.

    Memmap views are read-only by design; the loader never writes through them,
    so PyTorch's warning about non-writable backing is noise here.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*non-writable.*", category=UserWarning)
        return torch.from_numpy(arr)


class MemmapDataLoader:
    """Zero-copy ``np.memmap(mode='r')`` dataloader yielding ``[B, S]`` tensors.

    Contract (M1.4): strictly ``num_workers=1`` (no multiprocessing), batches
    are pre-sliced zero-copy ``torch.from_numpy`` views of the memmap.

    **ADR (dtype):** full batches are yielded as ``torch.uint16`` views — the
    ONLY path that keeps host RAM flat (< 50 MB on the 100M-token DoD read;
    empirically, any per-batch uint16->long cast retains ~file-size memory in
    the allocator arena). The ``.long()`` conversion happens once per batch in
    the M4.1 trainer during the GPU transfer (``batch.to(device, dtype=long)``),
    so device tensors are long while host RSS stays flat.

    ``state()`` / ``restore()`` snapshot the byte offset for M4.2 checkpoint
    resume.
    """

    def __init__(self, mmap: np.memmap, index: list[_RowRecord] | None, pad_id: int = 0) -> None:
        self._mmap = mmap
        self._index = index
        self._pad_id = pad_id
        self._offset = 0

    @classmethod
    def from_flat(cls, bin_path: Path) -> MemmapDataLoader:
        """Open a flat uint16 binary (no index): contiguous-token streaming.

        Suppresses PyTorch's benign non-writable-view warning: memmap views
        are read-only by design; the loader never writes through them.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*non-writable.*",
                category=UserWarning,
            )
            mmap = np.memmap(bin_path, dtype="uint16", mode="r")
        return cls(mmap=mmap, index=None)

    @classmethod
    def from_indexed(
        cls,
        bin_path: Path,
        index_path: Path,
        meta_path: Path,
    ) -> MemmapDataLoader:
        """Open a packed binary with sidecar index/meta: row-aware streaming.

        Suppresses PyTorch's benign non-writable-view warning (see
        ``from_flat``).
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*non-writable.*",
                category=UserWarning,
            )
            mmap = np.memmap(bin_path, dtype="uint16", mode="r")
        index = np.load(index_path)
        records = [
            _RowRecord(offset=int(o), length=int(n), kind=PackKind(int(k)))
            for o, n, k in zip(index["starts"], index["lengths"], index["kind"], strict=True)
        ]
        # Load meta purely as a contract guard (kept for provenance/debugging).
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["vocab_size"] <= MAX_UINT16, "index/meta mismatch"
        return cls(mmap=mmap, index=records, pad_id=int(meta["pad_id"]))

    def iter_batches(
        self,
        batch_size: int,
        seq_len: int,
        *,
        kinds: Sequence[PackKind] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Yield ``[batch_size, seq_len]`` zero-copy uint16 batches (CPU).

        Flat mode: contiguous chunks of the binary as zero-copy views. Indexed
        mode: samples rows by kind filter, pads with ``pad_id`` and truncates
        to ``seq_len`` (per-row ``np.pad`` copies — row scale, negligible vs.
        the 100M flat read). The final partial flat batch is padded once.
        """
        if batch_size < 1 or seq_len < 1:
            raise ValueError("batch_size and seq_len must be >= 1")
        needed = batch_size * seq_len
        offset = self._offset

        if self._index is None:
            # Flat mode: contiguous chunks, zero-copy views into the memmap.
            total = self._mmap.size
            while offset < total:
                end = min(offset + needed, total)
                chunk = self._mmap[offset:end]
                offset = end
                n = chunk.size
                if n == needed:
                    yield _zero_copy_view(chunk).reshape(batch_size, seq_len)
                elif n > 0:
                    # Final partial batch: one-time padded copy (keeps shape).
                    padded = np.pad(chunk, (0, needed - n), constant_values=self._pad_id)
                    yield _zero_copy_view(padded).reshape(batch_size, seq_len)
                self._offset = offset
            return

        # Indexed mode: rows ordered by start offset (deterministic), filtered.
        rows = [r for r in self._index if kinds is None or r.kind in kinds]
        batch_rows: list[torch.Tensor] = []
        batch_count = 0
        for record in rows:
            if offset is not None and record.offset < offset:
                continue
            ids = self._mmap[record.offset : record.offset + record.length]
            n = min(ids.size, seq_len)
            padded = np.pad(ids[:n], (0, seq_len - n), constant_values=self._pad_id)
            row = _zero_copy_view(padded)
            batch_rows.append(row)
            batch_count += 1
            if batch_count == batch_size:
                yield torch.stack(batch_rows)
                batch_rows = []
                batch_count = 0
                offset = self._offset
        if batch_rows:
            # Pad the final partial batch so the shape stays invariant.
            needed = batch_size - len(batch_rows)
            pad_row = torch.zeros((seq_len,), dtype=torch.uint16)
            batch_rows.extend([pad_row] * needed)
            yield torch.stack(batch_rows)

    def state(self) -> dict[str, int]:
        """Byte-offset snapshot for M4.2 fault-tolerant resume."""
        return {"offset_bytes": self._offset}

    def restore(self, state: dict[str, int]) -> None:
        """Restore a prior ``state()`` snapshot (rewind the byte offset)."""
        offset = state["offset_bytes"]
        if offset < 0 or offset > self._mmap.size:
            raise ValueError(f"invalid restore offset {offset}")
        self._offset = offset

    @property
    def num_workers(self) -> int:
        """Hard contract: the memmap loader never spawns worker processes."""
        return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: ``python -m bayan_slm_engine.data.memmap_streamer ...``.

    Flags (M1.4 Step C): --input-dir, --synthetic, --tokenizer, --output-bin,
    --pad-id. Returns 0 on success, 1 on ``BinaryPackingError``/usage errors.
    """
    parser = argparse.ArgumentParser(description="M1.4 16-bit binary packer")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--synthetic", type=Path, default=None)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output-bin", required=True, type=Path)
    parser.add_argument("--pad-id", type=int, default=0)
    args = parser.parse_args(argv)

    config = PackConfig(
        input_dir=args.input_dir,
        tokenizer_path=args.tokenizer,
        output_bin=args.output_bin,
        synthetic_jsonl=args.synthetic,
        pad_id=args.pad_id,
    )
    try:
        report = BinaryPacker(config).pack()
    except BinaryPackingError as exc:
        print(f"packing aborted: {exc}", file=sys.stderr)
        return 1
    print(
        f"packed {report.total_tokens} tokens "
        f"({report.rows_raw} raw + {report.rows_synthetic} synthetic) -> {report.output_bin}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
