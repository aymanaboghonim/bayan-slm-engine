"""Deterministic SADA22-style metadata fixture generator (M1.2).

Writes ``tests/fixtures/sada_metadata.parquet`` — a committed, hermetic
placeholder mirroring the REAL SADA22 schema (verified via the HF
datasets-server API, 2026-08-15):

* ``audio``          (Audio)        — path column (fixture uses dummy paths)
* ``text``           (string)       — GroundTruthText, diacritics/punct preserved
* ``cleaned_text``   (string)       — diacritics/punct stripped (STT target)
* ``speaker_age``    (string)       — e.g. ``Elderly -- كبير في السن``
* ``speaker_gender`` (string)       — ``Male`` / ``Female``
* ``speaker_dialect``(string)       — ``Najdi`` / ``Hijazi`` / ``Unknown``

Real SADA22 encodes multi-speaker rows as the literal string
``More than 1 speaker اكثر من متحدث`` in ALL demographic columns (there is no
separate ``speaker_N`` column) — the fixture replicates that exact sentinel so
the router logic is tested against ground truth.

The output is deterministic for a given seed (``random.Random`` MT is stable
across CPython versions).

Usage: ``uv run python tests/fixtures/generate_sada_fixture.py``
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

MULTI_SPEAKER_SENTINEL = "More than 1 speaker اكثر من متحدث"
DIALECTS = ["Najdi", "Najdi", "Najdi", "Hijazi", "Unknown"]
GENDERS = ["Male", "Male", "Female", "Female"]
AGES = ["Elderly -- كبير في السن", "Adult -- بالغ", "Child -- طفل"]

# (text, cleaned_text) pairs exercising the real SADA22 behavior:
# text keeps diacritics/punctuation; cleaned_text strips them.
ROWS: list[tuple[str, str]] = [
    ("ووضّح كلامك يا مغيث", "ووضح كلامك يا مغيث"),
    ("وين رايح من بدري؟", "وين رايح من بدري"),
    ("الحين نروح للسوق، ولا نجلس؟", "الحين نروح للسوق ولا نجلس"),
    ("تعال عندنا بكرة بالعصر.", "تعال عندنا بكرة بالعصر"),
    ("أبغى أكل سعودي اليوم!", "ابغى اكل سعودي اليوم"),
    ("إيش أخبارك مع الشغل؟", "ايش اخبارك مع الشغل"),
    ("هذا البيت حق عمي.", "هذا البيت حق عمي"),
    ("لا تنسى تاخذ معك الجوال.", "لا تنسى تاخذ معك الجوال"),
    ("شلونك بعد ما طلعت من المستشفى؟", "شلونك بعد ما طلعت من المستشفى"),
    ("بس أنا ما أقدر أروح الحين.", "بس انا ما اقدر اروح الحين"),
]


def generate(output: Path, *, seed: int = 11, rows_per: int = 3) -> int:
    """Write the deterministic SADA22-style fixture; returns rows written."""
    rng = random.Random(seed)
    records: list[dict[str, str]] = [
        {
            "audio": f"audio/{rng.randrange(10000):05d}.wav",
            "text": text,
            "cleaned_text": cleaned,
            "speaker_age": rng.choice(AGES),
            "speaker_gender": rng.choice(GENDERS),
            "speaker_dialect": rng.choice(DIALECTS),
        }
        for text, cleaned in ROWS
        for _ in range(rows_per)
    ]
    # Always include the multi-speaker sentinel rows (all three demo cols).
    for text, cleaned in ROWS[:2]:
        records.append(
            {
                "audio": f"audio/{rng.randrange(10000):05d}.wav",
                "text": text,
                "cleaned_text": cleaned,
                "speaker_age": MULTI_SPEAKER_SENTINEL,
                "speaker_gender": MULTI_SPEAKER_SENTINEL,
                "speaker_dialect": MULTI_SPEAKER_SENTINEL,
            }
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the M1.2 SADA22-style metadata fixture.")
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/sada_metadata.parquet"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rows-per", type=int, default=3)
    args = parser.parse_args()
    n = generate(args.output, seed=args.seed, rows_per=args.rows_per)
    print(f"Wrote {n} rows -> {args.output}")


if __name__ == "__main__":
    main()
