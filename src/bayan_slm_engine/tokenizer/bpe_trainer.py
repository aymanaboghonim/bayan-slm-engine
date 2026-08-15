"""Custom 16k Arabic-native BPE trainer (Hugging Face ``tokenizers`` backend).

SSOT: docs/BLUEPRINT.md §2 — clitic-optimized BPE; V = 16,000 keeps every token
ID inside ``uint16`` for the M1.4 zero-copy binary packer.

Bootstrap corpus note (M1.1): ``make tokenize`` trains on
``tests/fixtures/dialect_corpus.txt`` (committed placeholder) until the real
SDAIA + synthetic-template corpus lands in M1.2. The trainer materializes the
normalized training split in RAM — fine at bootstrap scale; M1.2 swaps in a
streaming iterator for the full corpus.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer as HFBpeTrainer

from bayan_slm_engine.tokenizer.normalizer import ArabicNormalizer

SPECIAL_TOKENS: list[str] = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]

# Clitics kept as atomic tokens — dialect-native design (BLUEPRINT §2).
# Excluded from the R_char fallback metric in verify_vocab.
#
# 8-char allowlist (BLUEPRINT §2 ADR, 2026-08-14):
#   و ف ب ل ك س — standard MSA/Saudi proclitics (conjunction, prepositions, future marker).
#   ح          — Saudi dialectal future proclitic (حيجي، حنسوي).
#   ع          — Saudi dialectal truncated على (عالبيت، عالجبل).
# NOTE: ا is deliberately EXCLUDED — it is not a clitic; whitelisting the most
# frequent Arabic letter would mask genuine fallback (elongation ااا, byte-fallback).
CLITIC_ALLOWLIST: frozenset[str] = frozenset({"و", "ف", "ب", "ل", "ك", "س", "ح", "ع"})

VOCAB_SIZE = 16_000
VALIDATION_FRACTION = 0.02
SEED = 42


def is_validation_line(index: int, *, seed: int, fraction: float) -> bool:
    """Deterministic, stable seeded holdout predicate.

    Uses blake2b (immune to ``PYTHONHASHSEED``) so ``bpe_trainer`` and
    ``verify_vocab`` derive the SAME validation split from the same corpus
    + seed — the diagnostic set is a pure function of the corpus.
    """
    digest = hashlib.blake2b(f"{seed}:{index}".encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return (value % 10_000) < int(fraction * 10_000)


@dataclass(frozen=True)
class TrainedBPE:
    """Result of a BPE training run."""

    tokenizer: Tokenizer
    vocab_size: int
    seed: int
    validation_fraction: float
    train_lines: int
    validation_lines: int
    output_path: Path


class BPETrainer:
    """Trains a byte-level BPE tokenizer to exactly ``vocab_size`` tokens."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        *,
        seed: int = SEED,
        validation_fraction: float = VALIDATION_FRACTION,
        min_frequency: int = 2,
    ) -> None:
        if not 0 < vocab_size <= 65_536:
            raise ValueError("vocab_size must satisfy 0 < vocab_size <= 65_536 (uint16 range)")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
        self.vocab_size = vocab_size
        self.seed = seed
        self.validation_fraction = validation_fraction
        self.min_frequency = min_frequency
        self._normalizer = ArabicNormalizer()

    def train_from_path(self, corpus_path: Path, output_path: Path) -> TrainedBPE:
        """Train on a corpus file of raw lines; persist ``tokenizer.json``."""
        with Path(corpus_path).open(encoding="utf-8") as fh:
            lines = fh.readlines()
        return self.train(lines, output_path)

    def train(self, corpus: Iterable[str], output_path: Path) -> TrainedBPE:
        """Train on re-iterable raw lines; persist ``tokenizer.json`` at ``output_path``."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        train_lines: list[str] = []
        validation_lines: list[str] = []
        index = 0
        for line in corpus:
            normalized = self._normalizer.normalize(line)
            if not normalized:
                index += 1
                continue
            if is_validation_line(index, seed=self.seed, fraction=self.validation_fraction):
                validation_lines.append(normalized)
            else:
                train_lines.append(normalized)
            index += 1

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = HFBpeTrainer(  # type: ignore[no-untyped-call]
            vocab_size=self.vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=self.min_frequency,
            show_progress=False,
        )
        tokenizer.train_from_iterator(train_lines, trainer=trainer, length=len(train_lines))
        tokenizer.save(str(output_path))

        return TrainedBPE(
            tokenizer=tokenizer,
            vocab_size=len(tokenizer.get_vocab()),
            seed=self.seed,
            validation_fraction=self.validation_fraction,
            train_lines=len(train_lines),
            validation_lines=len(validation_lines),
            output_path=output_path,
        )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (invoked by ``make tokenize``)."""
    parser = argparse.ArgumentParser(
        description="Train the bayan 16k Arabic-native BPE tokenizer (M1.1)."
    )
    parser.add_argument(
        "--corpus", type=Path, required=True, help="Raw text corpus (one line per document)"
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validation-fraction", type=float, default=VALIDATION_FRACTION)
    args = parser.parse_args(argv)

    trainer = BPETrainer(
        vocab_size=args.vocab_size,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    result = trainer.train_from_path(args.corpus, args.output)
    print(
        f"BPE trained: vocab={result.vocab_size} "
        f"train_lines={result.train_lines} validation_lines={result.validation_lines}"
    )
    print(f"Tokenizer saved: {args.output}")


if __name__ == "__main__":
    main()
