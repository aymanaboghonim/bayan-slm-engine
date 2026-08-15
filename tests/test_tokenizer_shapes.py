"""M1.1 tokenizer shape & contract tests (normalizer + 16k BPE).

SSOT: docs/BLUEPRINT.md §2 — deterministic orthographic normalization and a
clitic-optimized 16,000-vocab BPE. All assertions run on dummy strings /
CPU tensors (hermetic CI, ``BAYAN_OFFLINE=1``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from bayan_slm_engine.tokenizer.bpe_trainer import VOCAB_SIZE, BPETrainer, is_validation_line
from bayan_slm_engine.tokenizer.normalizer import ArabicNormalizer

# Highly complex dialectal strings: mixed diacritics, Alef variants,
# Hamza positions, Ta-Marbuta, and Najdi clitics.
DIALECT_SAMPLES = [
    "وش تسوي الحين",
    "أبغى أكل سعودي اليوم",
    "إيش أخبارك مع الشغل",
    "الحين نروح للسوق ولا نجلس",
    "مُحَمَّدٌ يَلْعَبُ بِالكُرَةِ",
    "مدرسة الحَيِّ الجديدة",
    "عَلَى الجَبَلِ كَانَ البَيْتُ",
    "حيجي أخوي بكرة بالعصر",
    "عالبيت ولا عالسوق الحين",
    "لا تنسى تاخذ معك الجوال",
]


@pytest.fixture
def normalizer() -> ArabicNormalizer:
    return ArabicNormalizer()


class TestNormalizer:
    def test_alef_variants_unify_to_bare_alef(self, normalizer: ArabicNormalizer) -> None:
        assert normalizer.normalize("أحمد إبراهيم آدم") == "احمد ابراهيم ادم"

    def test_hamza_positions_normalize(self, normalizer: ArabicNormalizer) -> None:
        assert normalizer.normalize("مؤمن بئر") == "مومن بير"

    def test_ta_marbuta_resolves_to_ha(self, normalizer: ArabicNormalizer) -> None:
        assert normalizer.normalize("مدرسة جميلة") == "مدرسه جميله"

    def test_diacritics_stripped(self, normalizer: ArabicNormalizer) -> None:
        # Diacritization is strictly a Tier B TTS-frontend concern.
        assert normalizer.normalize("مُحَمَّدٌ") == "محمد"

    def test_whitespace_collapsed_and_stripped(self, normalizer: ArabicNormalizer) -> None:
        assert normalizer.normalize("  السلام \t عليكم  ") == "السلام عليكم"

    def test_control_characters_removed(self, normalizer: ArabicNormalizer) -> None:
        assert normalizer.normalize("سلام\u200bعليكم") == "سلامعليكم"

    def test_empty_and_whitespace_only_inputs(self, normalizer: ArabicNormalizer) -> None:
        assert normalizer.normalize("") == ""
        assert normalizer.normalize("   ") == ""


class TestBPEContract:
    def test_vocab_size_exactly_16000(self, trained_tokenizer: Tokenizer) -> None:
        assert len(trained_tokenizer.get_vocab()) == VOCAB_SIZE == 16_000

    def test_all_token_ids_fit_uint16(self, trained_tokenizer: Tokenizer) -> None:
        assert all(0 <= tid < 65_536 for tid in trained_tokenizer.get_vocab().values())

    def test_roundtrip_decode_encode_equals_normalized(
        self, trained_tokenizer: Tokenizer, normalizer: ArabicNormalizer
    ) -> None:
        for sample in DIALECT_SAMPLES:
            expected = normalizer.normalize(sample)
            encoded = trained_tokenizer.encode(expected, add_special_tokens=False)
            # ByteLevel BPE prepends a space token on encode; normalized text
            # never has surrounding whitespace, so lstrip() removes only that
            # encoding artifact (tokenizer decode is otherwise lossless).
            decoded = trained_tokenizer.decode(encoded.ids).lstrip(" ")
            assert decoded == expected, (
                f"round-trip failed for {sample!r}: {decoded!r} != {expected!r}"
            )

    def test_training_is_deterministic(self, tmp_path: Path, dialect_corpus_path: Path) -> None:
        first = BPETrainer().train_from_path(dialect_corpus_path, tmp_path / "a.json")
        second = BPETrainer().train_from_path(dialect_corpus_path, tmp_path / "b.json")
        assert first.train_lines == second.train_lines
        assert first.validation_lines == second.validation_lines
        assert first.vocab_size == second.vocab_size

    def test_validation_split_predicate_is_deterministic(self) -> None:
        # blake2b-based: immune to PYTHONHASHSEED, identical across calls.
        for index in range(10):
            assert is_validation_line(index, seed=42, fraction=0.02) == is_validation_line(
                index, seed=42, fraction=0.02
            )
