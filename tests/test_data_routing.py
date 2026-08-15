"""M1.2 SADA22 metadata ingestion & modality-routing contract tests.

SSOT: docs/BLUEPRINT.md §"Dataset Specifications & Pipeline Mechanics" and
docs/EXECUTION_ROADMAP.md Milestone 1.2.

Asserts the routing contract against the committed hermetic fixture
(``tests/fixtures/sada_metadata.parquet``), which mirrors the REAL SADA22
schema verified via the HF datasets-server API (2026-08-15):

* ``text`` keeps diacritics/punctuation → Tier A; ``cleaned_text`` is stripped
  → Tier B (never ``text``).
* Multi-speaker rows (sentinel ``More than 1 speaker اكثر من متحدث``) are
  retained for Tier A but purged from Tier B.
* Dialects match case-insensitively (``Najdi``/``Hijazi``); ``Unknown`` excluded.
* Feeding a dirty ``cleaned_text`` (e.g., the raw ``text`` field) to Tier B
  raises a Pydantic ``ValidationError``.

Hermetic: fixture parquet only, ``BAYAN_OFFLINE=1`` safe, no network.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from bayan_slm_engine.data.ingestion import (
    MULTI_SPEAKER_SENTINEL,
    FieldRouter,
    MetadataFilter,
    SADA22MetadataRow,
    TierATrainingRow,
    TierBTrainingRow,
    is_clean_text,
    is_multi_speaker,
    is_target_dialect,
    iter_validated_rows,
    read_sada_metadata,
)


@pytest.fixture
def sada_frame() -> pd.DataFrame:
    return read_sada_metadata(Path(__file__).parent / "fixtures" / "sada_metadata.parquet")


@pytest.fixture
def validated_rows(sada_frame: pd.DataFrame) -> list[SADA22MetadataRow]:
    return iter_validated_rows(sada_frame)


class TestSchemaContract:
    def test_parquet_schema_matches_real_sada22(self, sada_frame: pd.DataFrame) -> None:
        assert set(sada_frame.columns) == {
            "audio",
            "text",
            "cleaned_text",
            "speaker_age",
            "speaker_gender",
            "speaker_dialect",
        }

    def test_all_rows_validate_against_pydantic(
        self, validated_rows: list[SADA22MetadataRow]
    ) -> None:
        assert len(validated_rows) == len(
            read_sada_metadata(Path(__file__).parent / "fixtures" / "sada_metadata.parquet")
        )
        assert all(row.text for row in validated_rows)
        assert all(row.cleaned_text for row in validated_rows)

    def test_unknown_column_rejected(self) -> None:
        # mypy flags the bogus kwarg (it's not a model field) — that's the
        # point: Pydantic `extra="forbid"` rejects it at runtime.
        with pytest.raises(ValidationError):
            SADA22MetadataRow(  # type: ignore[call-arg]
                audio="a.wav",
                text="سلام",
                cleaned_text="سلام",
                speaker_age="Adult -- بالغ",
                speaker_gender="Male",
                speaker_dialect="Najdi",
                bogus_column="nope",
            )


class TestDialectFilter:
    def test_najdi_hijazi_case_insensitive(self) -> None:
        assert is_target_dialect("Najdi")
        assert is_target_dialect("najdi")
        assert is_target_dialect("HIJAZI")
        assert is_target_dialect("hijazi")

    def test_unknown_and_khaliji_excluded(self) -> None:
        assert not is_target_dialect("Unknown")
        assert not is_target_dialect("Khaliji")
        assert not is_target_dialect("")

    def test_filter_accepts_only_najdi_hijazi(
        self, validated_rows: list[SADA22MetadataRow]
    ) -> None:
        filt = MetadataFilter()
        accepted = [r for r in validated_rows if filt.accepts(r)]
        rejected = [r for r in validated_rows if not filt.accepts(r)]
        assert all(r.speaker_dialect.lower() in {"najdi", "hijazi"} for r in accepted)
        assert all(r.speaker_dialect.lower() not in {"najdi", "hijazi"} for r in rejected)


class TestMultiSpeaker:
    def test_sentinel_detected(self, validated_rows: list[SADA22MetadataRow]) -> None:
        multi = [r for r in validated_rows if is_multi_speaker(r)]
        assert len(multi) > 0  # fixture guarantees sentinel rows
        assert all(
            r.speaker_dialect == MULTI_SPEAKER_SENTINEL
            and r.speaker_gender == MULTI_SPEAKER_SENTINEL
            and r.speaker_age == MULTI_SPEAKER_SENTINEL
            for r in multi
        )

    def test_multi_speaker_retained_in_tier_a_purged_from_tier_b(
        self, validated_rows: list[SADA22MetadataRow]
    ) -> None:
        """Decoupled modality ingestion (blueprint): multi-speaker rows are
        RETAINED for Tier A text distillation (conversational dialogue) but
        PURGED from Tier B TTS (acoustic crosstalk / MAS collapse)."""
        router = FieldRouter()
        multi = [r for r in validated_rows if is_multi_speaker(r)]
        assert multi, "fixture must contain multi-speaker rows"
        for row in multi:
            tier_a = router.route_tier_a(row)
            assert tier_a is not None, "multi-speaker must be retained for Tier A"
            assert tier_a.is_multi_speaker is True
            assert tier_a.text == row.text  # conversational text preserved
            assert router.route_tier_b(row) is None  # purged from Tier B


class TestTierRouting:
    def test_tier_a_preserves_text_verbatim(self, validated_rows: list[SADA22MetadataRow]) -> None:
        router = FieldRouter()
        najdi = next(r for r in validated_rows if r.speaker_dialect == "Najdi")
        tier_a = router.route_tier_a(najdi)
        assert isinstance(tier_a, TierATrainingRow)
        assert tier_a.text == najdi.text  # punctuation/diacritics preserved

    def test_tier_b_uses_cleaned_text_only(self, validated_rows: list[SADA22MetadataRow]) -> None:
        router = FieldRouter()
        for row in validated_rows:
            if is_multi_speaker(row) or not is_target_dialect(row.speaker_dialect):
                continue
            tier_b = router.route_tier_b(row)
            assert tier_b is not None
            assert isinstance(tier_b, TierBTrainingRow)
            assert tier_b.cleaned_text == row.cleaned_text
            assert is_clean_text(tier_b.cleaned_text)  # no diacritics/punct
            assert "؟" not in tier_b.cleaned_text and "،" not in tier_b.cleaned_text

    def test_tier_b_rejects_dirty_cleaned_text(
        self, validated_rows: list[SADA22MetadataRow]
    ) -> None:
        router = FieldRouter()
        # Take a Najdi row whose cleaned_text is genuinely clean; then simulate
        # a dirty row by copying the raw `text` (diacritics/punct) into it.
        clean = next(r for r in validated_rows if r.speaker_dialect == "Najdi")
        assert router.route_tier_b(clean) is not None  # control: clean passes
        dirty = clean.model_copy(update={"cleaned_text": clean.text})
        with pytest.raises(ValidationError):
            router.route_tier_b(dirty)

    def test_demographics_carried_into_tier_a(
        self, validated_rows: list[SADA22MetadataRow]
    ) -> None:
        router = FieldRouter()
        row = next(r for r in validated_rows if r.speaker_dialect == "Hijazi")
        tier_a = router.route_tier_a(row)
        assert tier_a is not None
        assert tier_a.speaker_dialect == row.speaker_dialect
        assert tier_a.speaker_gender == row.speaker_gender
        assert tier_a.speaker_age == row.speaker_age


class TestCleanText:
    def test_is_clean_text(self) -> None:
        assert is_clean_text("ووضح كلامك يا مغيث")
        assert not is_clean_text("ووضّح كلامك يا مغيث")  # diacritic
        assert not is_clean_text("وين رايح؟")  # Arabic question mark
        assert not is_clean_text("نروح للسوق،")  # Arabic comma
