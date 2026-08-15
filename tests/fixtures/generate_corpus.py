"""Deterministic bootstrap corpus generator for M1.1.

Writes ``tests/fixtures/dialect_corpus.txt`` — a committed placeholder corpus
until the real SDAIA + synthetic-template corpus lands in M1.2. Contains:

* hand-written Saudi-dialectal lines (dialect-representative clitics);
* seeded random Arabic-letter filler sized to saturate the 16,000-vocab BPE
  merge budget (each generated word appears exactly twice so every merge path
  meets the ``min_frequency=2`` requirement).

The output is deterministic for a given seed: words are sorted (set order is
hash-randomized) and ``random.Random``'s Mersenne-Twister stream is stable
across CPython versions.

Usage: ``uv run python tests/fixtures/generate_corpus.py``
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

ARABIC_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهويأإآءةىؤئ"

DIALECTAL_LINES = [
    "وش تسوي الحين",
    "تكفى لا تتأخر علينا",
    "الحين نروح للسوق ولا نجلس",
    "وين رايح من بدري",
    "أبغى أكل سعودي اليوم",
    "إيش أخبارك مع الشغل",
    "تعال عندنا بكرة بالعصر",
    "لا تنسى تاخذ معك الجوال",
    "بس أنا ما أقدر أروح الحين",
    "شلونك بعد ما طلعت من المستشفى",
    "هذا البيت حق عمي",
    "أول ما تخلص خلصني على طول",
    "المطر نزل البارحة كثير",
    "وش رايك نتعشى عند البيك",
    "أخوي الصغير دايما يلعب بالجوال",
    "الأكل عندهم مرة زين",
    "أبوي يبي يسافر للرياض الشهر الجاي",
    "خلك على طول معانا بالمجلس",
    "هذي السيارة جديدة مرة",
    "إنت من متى جاي من الشغل",
]

WORDS_PER_LINE = 12


def _word(rng: random.Random, min_len: int = 4, max_len: int = 10) -> str:
    return "".join(rng.choice(ARABIC_LETTERS) for _ in range(rng.randint(min_len, max_len)))


def generate(output: Path, *, seed: int = 7, distinct_words: int = 10_000) -> int:
    """Write the deterministic corpus; returns the number of lines written."""
    rng = random.Random(seed)
    words: set[str] = set()
    while len(words) < distinct_words:
        words.add(_word(rng))
    unique = sorted(words)  # sorted: set iteration order is hash-randomized

    corpus: list[str] = [line for line in DIALECTAL_LINES for _ in range(2)]
    pool = unique * 2  # every word twice → every merge path has frequency >= 2
    rng.shuffle(pool)
    corpus.extend(
        " ".join(pool[i : i + WORDS_PER_LINE]) for i in range(0, len(pool), WORDS_PER_LINE)
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(corpus) + "\n", encoding="utf-8")
    return len(corpus)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the M1.1 bootstrap dialect corpus.")
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/dialect_corpus.txt"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--distinct-words", type=int, default=10_000)
    args = parser.parse_args()
    lines = generate(args.output, seed=args.seed, distinct_words=args.distinct_words)
    print(f"Wrote {lines} lines -> {args.output}")


if __name__ == "__main__":
    main()
