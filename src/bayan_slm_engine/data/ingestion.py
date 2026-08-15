"""SADA22 metadata ingestion & modality routing (M1.2).

SSOT: docs/BLUEPRINT.md §"Dataset Specifications & Pipeline Mechanics" and
docs/EXECUTION_ROADMAP.md Milestone 1.2 — Metadata-First Ingestion.

The pipeline consumes the REAL ``MohamedRashad/SADA22`` parquet metadata
(667 h Saudi speech; cc-by-nc-sa-4.0) and performs **dynamic field routing**:

* **Tier A (SLM / TTS training):** the ``text`` column (GroundTruthText) —
  punctuation and diacritics preserved for intent parsing / prosodic modeling.
* **Tier B (STT encoder):** strictly the ``cleaned_text`` column (stripped) —
  feeding the raw ``text`` field raises a Pydantic ``ValidationError``.

Schema facts verified against the real dataset (HF datasets-server, 2026-08-15):

* Columns: ``audio``, ``text``, ``cleaned_text``, ``speaker_age``,
  ``speaker_gender``, ``speaker_dialect``.
* Dialects are CAPITALIZED (``Najdi``, ``Hijazi``; also ``Unknown``) — matching
  is case-insensitive. Only ``najdi`` / ``hijazi`` are retained downstream
  (blueprint scope); ``khaliji`` and ``unknown`` are excluded.
* Multi-speaker rows are NOT a separate column — they carry the literal
  sentinel ``More than 1 speaker اكثر من متحدث`` in ALL demographic columns.
  Such rows are retained for Tier A (turn-taking) but purged from Tier B TTS
  (cross-talk contamination).

Hermetic CI: tests read the committed fixture parquet
(``tests/fixtures/sada_metadata.parquet``); real SADA22 ingestion is local-only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

#: Real SADA22 sentinel marking multi-speaker rows (all demo columns).
MULTI_SPEAKER_SENTINEL = "More than 1 speaker اكثر من متحدث"

#: Blueprint M1.2 scope: retain only these dialects (case-insensitive).
TARGET_DIALECTS = frozenset({"najdi", "hijazi"})

#: Diacritics + Arabic punctuation stripped in SADA22's ``cleaned_text``.
_DIACRITICS_AND_PUNCT_RE = re.compile(
    r"[\u0640\u064b-\u065f\u0670\u0671\u06d6-\u06ed\u060c\u061b\u061f\u0660-\u0669\u066a-\u066d]"
)


class SADA22MetadataRow(BaseModel):
    """Pydantic v2 contract for one real SADA22 parquet row.

    ``extra="forbid"``: the fixture mirrors the verified 6-column schema, so any
    unexpected column is a contract violation (catches schema drift early).
    """

    model_config = {"extra": "forbid"}

    audio: str
    text: str = Field(min_length=1)
    cleaned_text: str = Field(min_length=1)
    speaker_age: str
    speaker_gender: str
    speaker_dialect: str

    @field_validator("text", "cleaned_text")
    @classmethod
    def _strip_whitespace(cls, value: str) -> str:
        return value.strip()


class TierATrainingRow(BaseModel):
    """Tier A (SLM/TTS) row: ``text`` preserved verbatim + demographics."""

    text: str = Field(min_length=1)
    speaker_age: str
    speaker_gender: str
    speaker_dialect: str
    is_multi_speaker: bool


class TierBTrainingRow(BaseModel):
    """Tier B (STT) row: strictly ``cleaned_text`` (no punctuation/diacritics)."""

    cleaned_text: str = Field(min_length=1)
    is_multi_speaker: bool


def is_multi_speaker(row: SADA22MetadataRow) -> bool:
    """True when the SADA22 multi-speaker sentinel appears in any demographic col."""
    return any(
        value == MULTI_SPEAKER_SENTINEL
        for value in (row.speaker_age, row.speaker_gender, row.speaker_dialect)
    )


def is_target_dialect(dialect: str) -> bool:
    """Case-insensitive Najdi/Hijazi check (blueprint M1.2 scope)."""
    return dialect.strip().lower() in TARGET_DIALECTS


def is_clean_text(text: str) -> bool:
    """True if ``text`` contains no diacritics or Arabic punctuation."""
    return not _DIACRITICS_AND_PUNCT_RE.search(text)


class MetadataFilter:
    """Deterministic dialect gate over SADA22 rows.

    Retains rows whose dialect is Najdi or Hijazi (case-insensitive). The
    multi-speaker sentinel and ``Unknown`` are deliberately NOT target
    dialects — but see ``FieldRouter.route_tier_a``, which retains
    multi-speaker rows for text distillation despite the sentinel dialect.
    """

    def accepts(self, row: SADA22MetadataRow) -> bool:
        return is_target_dialect(row.speaker_dialect)


class FieldRouter:
    """Routes a SADA22 row to Tier A or Tier B training rows.

    **Decoupled modality ingestion (blueprint §"Domain & Persona Focus"):**
    multi-speaker rows are RETAINED for Tier A text distillation (natural
    conversational turn-taking, Q&A, discourse markers) but PURGED from Tier B
    TTS exports — overlapping speech causes Monotonic Alignment Search (MAS)
    failure and cross-talk contamination in single-speaker acoustic models.

    Tier B is strictly ``cleaned_text`` — calling ``route_tier_b`` with a row
    whose ``cleaned_text`` is dirty (e.g., accidentally the raw ``text`` field)
    raises a Pydantic ``ValidationError``.
    """

    def __init__(self, filter_: MetadataFilter | None = None) -> None:
        self.filter = filter_ or MetadataFilter()

    def route_tier_a(self, row: SADA22MetadataRow) -> TierATrainingRow | None:
        """Tier A keeps ``text`` verbatim (punct/diacritics).

        Multi-speaker rows pass regardless of the sentinel dialect label — the
        transcript text still carries high-value conversational dialogue.
        Downstream M1.3 safeguards (demographic conditioning + lexical/MinHash
        filters) sanitize any unlabelled-dialect drift.
        """
        multi = is_multi_speaker(row)
        if not multi and not self.filter.accepts(row):
            return None
        return TierATrainingRow(
            text=row.text,
            speaker_age=row.speaker_age,
            speaker_gender=row.speaker_gender,
            speaker_dialect=row.speaker_dialect,
            is_multi_speaker=multi,
        )

    def route_tier_b(self, row: SADA22MetadataRow) -> TierBTrainingRow | None:
        """Tier B uses ``cleaned_text`` only; multi-speaker rows are purged.

        Multi-speaker audio produces acoustic crosstalk and MAS alignment
        collapse — such rows are excluded from STT/TTS training entirely.
        """
        if is_multi_speaker(row):
            return None  # cross-talk contamination — purge from STT/TTS training
        if not self.filter.accepts(row):
            return None
        if not is_clean_text(row.cleaned_text):
            raise ValidationError.from_exception_data(
                "TierBTrainingRow",
                [
                    {
                        "type": "value_error",
                        "loc": ("cleaned_text",),
                        "input": row.cleaned_text,
                        "ctx": {"error": "cleaned_text must be free of diacritics/punctuation"},
                    }
                ],
            )
        return TierBTrainingRow(cleaned_text=row.cleaned_text, is_multi_speaker=False)


def read_sada_metadata(path: Path, *, chunk_size: int = 10_000) -> pd.DataFrame:
    """Chunked parquet reader — RAM-safe for the ~28-shard SADA22 corpus.

    ``pyarrow`` (core dep) backs ``pandas.read_parquet``. Chunked iteration
    keeps WSL2 host RAM flat (< 12 GB ceiling) without materializing the full
    table in memory.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    reader = cast(
        pd.DataFrame,
        pd.read_parquet(path, engine="pyarrow", columns=None),  # type: ignore[call-overload]
    )
    return reader  # M1.2 slice: single-file read; sharded iteration lands in M1.4


def iter_validated_rows(frame: pd.DataFrame) -> list[SADA22MetadataRow]:
    """Validate every DataFrame row against the Pydantic contract."""
    return [SADA22MetadataRow.model_validate(rec) for rec in frame.to_dict("records")]


# Re-exported for ergonomic imports in tests / callers.
__all__ = [
    "MULTI_SPEAKER_SENTINEL",
    "TARGET_DIALECTS",
    "SADA22MetadataRow",
    "TierATrainingRow",
    "TierBTrainingRow",
    "is_multi_speaker",
    "is_target_dialect",
    "is_clean_text",
    "MetadataFilter",
    "FieldRouter",
    "read_sada_metadata",
    "iter_validated_rows",
    "ValidationError",
]
