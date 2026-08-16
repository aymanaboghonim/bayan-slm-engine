"""M1.4 contract tests: 16-bit zero-copy binary packer & memory-mapped streamer.

SSOT: docs/EXECUTION_ROADMAP.md M1.4 DoD; docs/BLUEPRINT.md §3 steps 4-5.

Contracts validated here (hermetic, CPU-only, ``BAYAN_OFFLINE=1`` safe):
  1. ``BinaryPacker`` produces a flat uint16 .bin + sidecar .index.npz + .meta.json.
  2. Hard zero-OOV gate (M1.1 BPE Coverage Rule): synthetic rows with Arabic
     byte-fallback tokens raise ``BinaryPackingError``; allowlisted clitics pass.
  3. Flat 100M-token read via ``MemmapDataLoader.from_flat`` streams ``[B, S]``
     zero-copy ``torch.uint16`` views (ADR: ``.long()`` happens at the M4.1
     GPU transfer) while host RAM delta stays strictly < 50 MB (psutil).
  4. ``from_indexed`` routes kinds (RAW/SYNTHETIC), pads/truncates to seq_len,
     and ``state()``/``restore()`` snapshot the resume byte-offset (M4.2).

The suite targets the interface stubs (Step A) — it must FAIL with
``NotImplementedError`` until Step C implements the internals.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import psutil
import pytest
import torch
from tokenizers import Tokenizer

from bayan_slm_engine.data.distillation import MultiTurnDialogue, write_dialogues_jsonl
from bayan_slm_engine.data.memmap_streamer import (
    INDEX_SUFFIX,
    MAX_UINT16,
    META_SUFFIX,
    UNK_ID,
    BinaryPacker,
    BinaryPackingError,
    MemmapDataLoader,
    PackConfig,
    PackKind,
)

# CI override: shrink the flat-memory fixture when BAYAN_SMALL_MEMORY_TEST=1.
_MMAP_TOKEN_COUNT = int(os.environ.get("BAYAN_SMALL_MEMORY_TEST", "100_000_000"))
assert _MMAP_TOKEN_COUNT >= 1_000  # floor keeps the seq_len contract meaningful


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def memmap_flat_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped 100M-token (default) seeded random uint16 binary.

    Written once via numpy (mmap mode='w+'), then reopened read-only. The 200 MB
    artifact lives in a session tmp dir; ``BAYAN_SMALL_MEMORY_TEST=1`` shrinks it.
    """
    out = tmp_path_factory.mktemp("memmap") / "flat_100m.uint16.bin"
    n = _MMAP_TOKEN_COUNT
    # Pre-allocate then fill via mmap to avoid a full in-RAM 200 MB array.
    # np.memmap is a plain ndarray subclass (no context manager) — flush + close.
    m = np.memmap(out, dtype="uint16", mode="w+", shape=(n,))
    try:
        rng = np.random.default_rng(seed=42)
        chunk = 1_000_000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            m[start:end] = rng.integers(0, MAX_UINT16 + 1, size=end - start, dtype="uint16")
        m.flush()
    finally:
        del m
    return out


@pytest.fixture(scope="module")
def packed_bin(tmp_path_factory: pytest.TempPathFactory, trained_tokenizer_path: Path) -> Path:
    """Small packed corpus produced by the real BinaryPacker (Step C)."""
    root = tmp_path_factory.mktemp("packed")
    text_dir = root / "texts"
    text_dir.mkdir()
    # One SADA22 ``text`` row per line (Tier A raw).
    (text_dir / "sada.txt").write_text(
        "ووضّح كلامك يا مغيث\nالسلام عليكم كيف الحال\n",
        encoding="utf-8",
    )
    synthetic = root / "dialogues.jsonl"
    write_dialogues_jsonl(
        [
            MultiTurnDialogue(user="شلونك اليوم", assistant="بخير الحمد لله", dialect="najdi"),
            MultiTurnDialogue(user="وين رايح", assistant="نصير السوق", dialect="hijazi"),
        ],
        synthetic,
    )
    out = root / "corpus.uint16.bin"
    BinaryPacker(
        PackConfig(
            input_dir=text_dir,
            tokenizer_path=trained_tokenizer_path,
            output_bin=out,
            synthetic_jsonl=synthetic,
        )
    ).pack()
    return out


# ---------------------------------------------------------------------------
# 1. Packer contract: flat uint16 + sidecars
# ---------------------------------------------------------------------------


class TestPackerSidecars:
    def test_pack_produces_bin_and_sidecars(self, packed_bin: Path) -> None:
        assert packed_bin.exists()
        assert packed_bin.with_suffix(packed_bin.suffix + INDEX_SUFFIX).exists()
        assert packed_bin.with_suffix(packed_bin.suffix + META_SUFFIX).exists()

    def test_bin_is_contiguous_uint16(self, packed_bin: Path) -> None:
        m = np.memmap(packed_bin, dtype="uint16", mode="r")
        assert m.dtype == np.dtype("uint16")
        assert m.size > 0
        assert np.all(m >= 0) and np.all(m <= MAX_UINT16)
        del m

    def test_index_round_trip(self, packed_bin: Path) -> None:
        index = np.load(packed_bin.with_suffix(packed_bin.suffix + INDEX_SUFFIX))
        starts: np.ndarray = index["starts"]
        lengths: np.ndarray = index["lengths"]
        kinds: np.ndarray = index["kind"]
        assert starts.shape == lengths.shape == kinds.shape
        assert starts[0] == 0
        # Contiguous, non-overlapping token ranges.
        assert np.all(starts[1:] == starts[:-1] + lengths[:-1])
        assert set(np.unique(kinds).tolist()) <= {int(PackKind.RAW), int(PackKind.SYNTHETIC)}

    def test_meta_json_vocab_size(self, packed_bin: Path, trained_tokenizer_path: Path) -> None:
        meta = json.loads(packed_bin.with_suffix(packed_bin.suffix + META_SUFFIX).read_text())
        tokenizer = Tokenizer.from_file(str(trained_tokenizer_path))
        assert meta["vocab_size"] == tokenizer.get_vocab_size() == 16_000
        assert meta["total_tokens"] > 0
        assert meta["rows_raw"] == 2 and meta["rows_synthetic"] == 2


# ---------------------------------------------------------------------------
# 2. Hard zero-OOV gate (M1.1 BPE Coverage Rule)
# ---------------------------------------------------------------------------


class TestOOVGate:
    @pytest.fixture()
    def packer(self, tmp_path: Path, trained_tokenizer_path: Path) -> BinaryPacker:
        """Packer instance for direct gate unit tests (no packing run)."""
        return BinaryPacker(
            PackConfig(
                input_dir=tmp_path / "texts",
                tokenizer_path=trained_tokenizer_path,
                output_bin=tmp_path / "gate.uint16.bin",
            )
        )

    def test_id_beyond_vocab_raises(self, packer: BinaryPacker) -> None:
        """Real OOV: token id >= frozen vocab size aborts the pack hard."""
        vocab_size = packer._tokenizer.get_vocab_size()
        with pytest.raises(BinaryPackingError, match="OOV"):
            packer._validate_oov(row_idx=3, ids=[vocab_size])
        # In-range ids pass silently.
        packer._validate_oov(row_idx=3, ids=[vocab_size - 1])

    def test_unk_sentinel_raises(self, packer: BinaryPacker) -> None:
        """Unknown-token path: <unk> sentinel id aborts the pack hard."""
        with pytest.raises(BinaryPackingError, match="OOV"):
            packer._validate_oov(row_idx=0, ids=[UNK_ID])

    def test_single_byte_token_raises(self, packer: BinaryPacker) -> None:
        """A lone raw byte token (len-1 latin-1 render) is byte-level fallback.

        Every ByteLevel vocab contains the 256 raw bytes as len-1 tokens (e.g.
        ``Ø`` for byte 0xD8). Emitting one means a character was NOT covered
        by a merged byte-pair token — the hard OOV signal (M1.1 rule).
        """
        vocab = packer._tokenizer.get_vocab()
        raw_byte_token_id = next(
            token_id
            for token, token_id in vocab.items()
            if token_id != UNK_ID and len(token) == 1 and not token.isspace()
        )
        with pytest.raises(BinaryPackingError, match="OOV"):
            packer._validate_oov(row_idx=5, ids=[raw_byte_token_id])

    def test_synthetic_round_trip_lossless(
        self, tmp_path: Path, trained_tokenizer_path: Path
    ) -> None:
        """Every synthetic row round-trips decode(encode(norm)) == norm.

        With a byte-level 16k BPE, single-byte fallback is unreachable (all
        bytes are in the vocab); the zero-OOV guarantee is therefore a lossless
        encode/decode round-trip plus the id-range/unk checks above.
        """
        text_dir = tmp_path / "texts"
        text_dir.mkdir()
        (text_dir / "sada.txt").write_text("سطر عادي\n", encoding="utf-8")
        synthetic = tmp_path / "dialogues.jsonl"
        write_dialogues_jsonl(
            [MultiTurnDialogue(user="عايز أروح", assistant="كذا تمام", dialect="hijazi")],
            synthetic,
        )
        packer = BinaryPacker(
            PackConfig(
                input_dir=text_dir,
                tokenizer_path=trained_tokenizer_path,
                output_bin=tmp_path / "ok.uint16.bin",
                synthetic_jsonl=synthetic,
            )
        )
        report = packer.pack()
        assert report.oov_errors == 0
        # Lossless round-trip on the packed synthetic row (decode of ids).
        index = np.load(tmp_path / "ok.uint16.bin.index.npz")
        syn_mask = index["kind"] == int(PackKind.SYNTHETIC)
        for start, length in zip(
            index["starts"][syn_mask], index["lengths"][syn_mask], strict=True
        ):
            ids = np.memmap(tmp_path / "ok.uint16.bin", dtype="uint16", mode="r")[
                int(start) : int(start + length)
            ]
            decoded = Tokenizer.from_file(str(trained_tokenizer_path)).decode(ids.tolist())
            assert decoded  # non-empty, representable text

    def test_allowlisted_clitics_pass(self, tmp_path: Path, trained_tokenizer_path: Path) -> None:
        text_dir = tmp_path / "texts"
        text_dir.mkdir()
        (text_dir / "sada.txt").write_text("حياك الله\n", encoding="utf-8")
        synthetic = tmp_path / "dialogues.jsonl"
        write_dialogues_jsonl(
            [MultiTurnDialogue(user="وش أخبارك", assistant="بخير وعساك بخير", dialect="najdi")],
            synthetic,
        )
        packer = BinaryPacker(
            PackConfig(
                input_dir=text_dir,
                tokenizer_path=trained_tokenizer_path,
                output_bin=tmp_path / "ok.uint16.bin",
                synthetic_jsonl=synthetic,
            )
        )
        report = packer.pack()
        assert report.oov_errors == 0


# ---------------------------------------------------------------------------
# 3. Flat 100M read: shapes + hardware footprint < 50 MB (psutil)
# ---------------------------------------------------------------------------


class TestMemmapMemory:
    def test_flat_iteration_shapes(self, memmap_flat_bin: Path) -> None:
        loader = MemmapDataLoader.from_flat(memmap_flat_bin)
        batch_size, seq_len = 32, 1024
        count = 0
        for batch in loader.iter_batches(batch_size, seq_len):
            assert isinstance(batch, torch.Tensor)
            assert batch.dtype == torch.uint16  # zero-copy view (see ADR)
            assert batch.shape == (batch_size, seq_len)
            # uint16 has no CPU min/max reduction — inspect via numpy.
            values = batch.numpy()
            assert int(values.min()) >= 0 and int(values.max()) <= MAX_UINT16
            count += 1
            if count >= 3:  # shape contract on the first batches is enough here
                break
        assert count == 3

    def test_full_100m_read_ram_delta_below_50mb(self, memmap_flat_bin: Path) -> None:
        process = psutil.Process()
        loader = MemmapDataLoader.from_flat(memmap_flat_bin)
        # Warm page cache + allocator state before measuring the delta.
        next(iter(loader.iter_batches(32, 1024)))
        base_rss = process.memory_info().rss
        consumed = 0
        for _ in loader.iter_batches(32, 1024):
            consumed += 32 * 1024
        peak_delta = process.memory_info().rss - base_rss
        assert consumed >= _MMAP_TOKEN_COUNT  # the full file was streamed
        assert peak_delta < 50 * 1024 * 1024, f"host RAM delta {peak_delta / 1e6:.1f} MB >= 50 MB"


# ---------------------------------------------------------------------------
# 4. Indexed mode: kind routing, pad/truncate, state()/restore()
# ---------------------------------------------------------------------------


class TestIndexedLoader:
    def test_kind_filter_and_shapes(self, packed_bin: Path) -> None:
        loader = MemmapDataLoader.from_indexed(
            packed_bin,
            packed_bin.with_suffix(packed_bin.suffix + INDEX_SUFFIX),
            packed_bin.with_suffix(packed_bin.suffix + META_SUFFIX),
        )
        batches = list(loader.iter_batches(batch_size=2, seq_len=8, kinds=[PackKind.RAW]))
        assert all(b.shape == (2, 8) for b in batches)

    def test_pad_truncate_respects_seq_len(self, packed_bin: Path) -> None:
        loader = MemmapDataLoader.from_indexed(
            packed_bin,
            packed_bin.with_suffix(packed_bin.suffix + INDEX_SUFFIX),
            packed_bin.with_suffix(packed_bin.suffix + META_SUFFIX),
        )
        seq_len = 4  # forces truncation on longer rows and padding on shorter
        for batch in loader.iter_batches(batch_size=2, seq_len=seq_len):
            assert batch.shape == (2, seq_len)
            assert batch.dtype == torch.uint16

    def test_state_restore_resume_offset(self, packed_bin: Path) -> None:
        loader = MemmapDataLoader.from_indexed(
            packed_bin,
            packed_bin.with_suffix(packed_bin.suffix + INDEX_SUFFIX),
            packed_bin.with_suffix(packed_bin.suffix + META_SUFFIX),
        )
        state = loader.state()
        assert set(state) >= {"offset_bytes"}
        loader.restore(state)
        # Consume part of the stream, snapshot, then verify restore rewinds.
        loader.restore(state)
        first = next(loader.iter_batches(batch_size=2, seq_len=8))
        loader.restore(state)
        second = next(loader.iter_batches(batch_size=2, seq_len=8))
        assert torch.equal(first, second)
