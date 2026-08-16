"""SADA22 speaker-dispersion lexicon validator (M1.3 refinement).

SSOT: docs/BLUEPRINT.md §3 (Grounded Synthetic Distillation) and the M1.3
lexical-drift hardening ADR (docs/EXECUTION_ROADMAP.md Milestone 1.3).

**Role — soft investigative layer (never gates hermetic CI):** flags tokens
inside distilled dialogues whose speaker dispersion across the authentic
SADA22 corpus is below ``min_speakers``. Low dispersion means the token only
ever appeared in a handful of recordings — a signal of guest actors, radio
call-ins, or TV-drama code-switching — so the dialogue is a candidate for
regeneration. Local CLI runs only.

**Speaker proxy (ADR):** the verified SADA22 schema (HF datasets-server,
2026-08-15) has NO ``speaker_id`` column — only ``audio``, ``text``,
``cleaned_text`` and the demographic columns. Distinct ``audio`` rows are used
as the documented session proxy for speaker dispersion.

Hermetic CI: tests reuse the committed M1.2 fixture parquet
(``tests/fixtures/sada_metadata.parquet``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from bayan_slm_engine.data.distillation import MultiTurnDialogue
from bayan_slm_engine.data.ingestion import SADA22MetadataRow

#: Default minimum distinct recordings before a token is "native Saudi".
DEFAULT_MIN_SPEAKERS = 5

#: Diacritics + punctuation stripped from tokens before dispersion matching
#: (mirrors the drift regex's diacritic-insensitivity; deterministic).
_STRIP_RE = re.compile(
    r"[\u0640\u064b-\u065f\u0670\u0671\u06d6-\u06ed"
    r"\u060c\u061b\u061f.,!?؛:…\"'()\[\]{}]"
)


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenization with diacritics/punctuation stripped."""
    cleaned = _STRIP_RE.sub("", text)
    return [token for token in cleaned.split() if token]


class Sada22LexiconValidator:
    """Flags low-dispersion tokens against the authentic SADA22 corpus."""

    def __init__(self, min_speakers: int = DEFAULT_MIN_SPEAKERS) -> None:
        self._min_speakers = min_speakers

    def build_dispersion(self, rows: Iterable[SADA22MetadataRow]) -> dict[str, int]:
        """Token -> count of distinct audio rows containing it (speaker proxy).

        Distinct ``audio`` rows stand in for speakers (ADR): the verified
        SADA22 schema has no ``speaker_id`` column.
        """
        speakers: dict[str, set[str]] = {}
        for row in rows:
            for token in _tokenize(row.text):
                speakers.setdefault(token, set()).add(row.audio)
        return {token: len(audios) for token, audios in speakers.items()}

    def validate_dialogue(
        self, dialogue: MultiTurnDialogue, dispersion: dict[str, int]
    ) -> list[str]:
        """Low-dispersion tokens in the dialogue (order-preserving, deduped)."""
        flagged: list[str] = []
        seen: set[str] = set()
        for token in _tokenize(dialogue.user) + _tokenize(dialogue.assistant):
            if token in seen:
                continue
            seen.add(token)
            if dispersion.get(token, 0) < self._min_speakers:
                flagged.append(token)
        return flagged
