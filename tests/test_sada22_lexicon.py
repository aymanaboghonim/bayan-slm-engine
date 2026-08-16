"""M1.3 SADA22 lexicon dispersion validator contract tests.

SSOT: docs/EXECUTION_ROADMAP.md Milestone 1.3 (lexical-drift hardening).

Asserts the soft investigative layer:

* Dispersion counts DISTINCT ``audio`` rows per token (speaker proxy ADR —
  the verified SADA22 schema has no ``speaker_id`` column).
* Tokens with dispersion >= min_speakers pass; below are flagged.
* Unknown tokens (absent from the corpus) are flagged.
* Deterministic, no input mutation; hermetic (``BAYAN_OFFLINE=1`` safe).

One integration test reads the committed M1.2 fixture parquet; the rest use
hand-built rows/dictionaries for exact, collision-free assertions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bayan_slm_engine.data.distillation import MultiTurnDialogue
from bayan_slm_engine.data.ingestion import (
    SADA22MetadataRow,
    iter_validated_rows,
    read_sada_metadata,
)
from bayan_slm_engine.data.sada22_lexicon import Sada22LexiconValidator


def make_row(text: str, audio: str) -> SADA22MetadataRow:
    return SADA22MetadataRow(
        audio=audio,
        text=text,
        cleaned_text=text,
        speaker_age="Adult -- بالغ",
        speaker_gender="Male",
        speaker_dialect="Najdi",
    )


@pytest.fixture(scope="module")
def validated_rows() -> list[SADA22MetadataRow]:
    frame = read_sada_metadata(Path(__file__).parent / "fixtures" / "sada_metadata.parquet")
    return iter_validated_rows(frame)


class TestBuildDispersion:
    def test_counts_distinct_audio_rows(self) -> None:
        rows = [
            make_row("يا مغيث", "a1.wav"),
            make_row("يا صاحبي", "a2.wav"),
            make_row("يا مغيث", "a1.wav"),  # same audio must not double-count
            make_row("بس", "a3.wav"),
        ]
        dispersion = Sada22LexiconValidator().build_dispersion(rows)
        assert dispersion["يا"] == 2  # {a1, a2}
        assert dispersion["مغيث"] == 1  # {a1}
        assert dispersion["بس"] == 1  # {a3}

    def test_deterministic_and_no_mutation(self) -> None:
        rows = [make_row("يا مغيث", "a1.wav"), make_row("بس", "a2.wav")]
        first = Sada22LexiconValidator().build_dispersion(rows)
        second = Sada22LexiconValidator().build_dispersion(rows)
        assert first == second
        assert rows[0].text == "يا مغيث"  # input untouched

    def test_fixture_parquet_integration(self, validated_rows: list[SADA22MetadataRow]) -> None:
        dispersion = Sada22LexiconValidator().build_dispersion(validated_rows)
        assert len(dispersion) > 0
        # "يا" appears in the row-0 trio + 2 sentinel copies (>= 4 distinct audios).
        assert dispersion.get("يا", 0) >= 4


class TestValidateDialogue:
    def test_high_dispersion_tokens_pass(self) -> None:
        dispersion = {"يا": 5, "فلان": 6, "الحين": 6}
        dialogue = MultiTurnDialogue(user="يا فلان", assistant="الحين", dialect="hijazi")
        assert Sada22LexiconValidator().validate_dialogue(dialogue, dispersion) == []

    def test_low_dispersion_token_flagged(self) -> None:
        dispersion = {"يا": 5, "مغيث": 5, "بس": 3}
        dialogue = MultiTurnDialogue(user="بس يا مغيث", assistant="بخير", dialect="najdi")
        flagged = Sada22LexiconValidator().validate_dialogue(dialogue, dispersion)
        # "بس" has 3 < 5; "بخير" is absent (0 < 5) — both flagged, order preserved.
        assert flagged == ["بس", "بخير"]

    def test_unknown_tokens_flagged(self) -> None:
        dispersion = {"يا": 5}
        dialogue = MultiTurnDialogue(user="زين نعم", assistant="أعوذ بالله", dialect="najdi")
        flagged = Sada22LexiconValidator().validate_dialogue(dialogue, dispersion)
        assert set(flagged) == {"زين", "نعم", "أعوذ", "بالله"}

    def test_repeated_tokens_deduplicated(self) -> None:
        dispersion = {"يا": 5}
        dialogue = MultiTurnDialogue(user="يا يا", assistant="بخير", dialect="najdi")
        flagged = Sada22LexiconValidator().validate_dialogue(dialogue, dispersion)
        assert flagged == ["بخير"]
