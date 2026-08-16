"""Deterministic orthographic normalization for Saudi-dialectal Arabic.

SSOT: docs/BLUEPRINT.md §2 (Custom Arabic Tokenization & Normalization Engine).

All rules are deterministic Unicode/regex transforms — never prompt heuristics:

* Unicode NFC canonical composition.
* Alef unification (أ إ آ ٱ → ا) and Hamza-position normalization (ؤ → و, ئ → ي).
* Ha/Ta-Marbuta resolution (ة → ه) per dialectal morphological bounds.
* Spoken spelling-variant canonicalization (كده/كدا → كذا) — M1.3 ADR: keeps
  synthetic distillation output orthographically consistent; variants are
  normalized, never rejected as drift downstream.
* Stripping of spurious diacritics — diacritization is strictly a Tier B
  TTS-frontend concern.
* Whitespace collapse (single ASCII space) and control-character removal.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator

# All Arabic diacritics / orthographic marks removed before text modeling.
_DIACRITICS_RE = re.compile(
    "[\u0640\u064b-\u065f\u0670\u0671\u06d6-\u06dc\u06df-\u06e4\u06e7-\u06e8\u06ea-\u06ed]"
)

# Control chars, zero-width joiners, and other invisible format characters.
_CONTROL_RE = re.compile(
    "[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\u200b-\u200f\u2060-\u206f]"
)

_WHITESPACE_RE = re.compile(r"[ \t\r\n\f\v]+")

_ALEF_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_HAMZA_MAP = str.maketrans({"ؤ": "و", "ئ": "ي"})
_TA_MARBUTA_MAP = str.maketrans({"ة": "ه"})

# Spoken spelling variants -> canonical Saudi orthography (M1.3 ADR).
# ``كده``/``كدا`` ("like this") canonicalize to ``كذا``. Clitic prefixes
# (و، ف، ب، …) are preserved (``وكدا`` -> ``وكذا``). Word-boundary anchored
# so ``كدها`` (different lemma) never matches; applied after diacritic
# stripping so ``كدّه`` -> ``كذا`` too.
_CLITICS = "وفبلكسحع"
_SPELLING_VARIANTS_RE = re.compile(rf"\b([{_CLITICS}]*)(كدا|كده)\b")


def _canonical_spelling(match: re.Match[str]) -> str:
    """Preserve any clitic prefix, canonicalize the variant root to كذا."""
    return f"{match.group(1)}كذا"


class ArabicNormalizer:
    """Deterministic orthographic normalizer for Saudi-dialectal Arabic."""

    def normalize(self, text: str) -> str:
        """Normalize a single string (NFC → diacritic strip → character maps)."""
        normalized = unicodedata.normalize("NFC", text)
        normalized = _DIACRITICS_RE.sub("", normalized)
        normalized = _CONTROL_RE.sub("", normalized)
        normalized = normalized.translate(_ALEF_MAP)
        normalized = normalized.translate(_HAMZA_MAP)
        normalized = normalized.translate(_TA_MARBUTA_MAP)
        normalized = _SPELLING_VARIANTS_RE.sub(_canonical_spelling, normalized)
        return _WHITESPACE_RE.sub(" ", normalized).strip()

    def normalize_batch(self, texts: Iterable[str]) -> Iterator[str]:
        """Lazily normalize an iterable of strings."""
        for text in texts:
            yield self.normalize(text)
