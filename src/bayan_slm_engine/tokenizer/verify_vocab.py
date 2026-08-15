"""Phase 1 Tokenizer Quality Diagnostic Report — "Calculate & Report" paradigm.

SSOT: docs/BLUEPRINT.md §2 (Phase 1 Tokenizer Quality Diagnostic Report).

The report is a DIAGNOSTIC, never a quality gate: ``report_metrics`` always
returns exit code 0 so training pipelines proceed unblocked. Metrics are
emitted to three channels:

1. stdout                       — human-readable report
2. logs/tokenizer_metrics.json  — machine-auditable artifact
3. trackio                      — persistent local SQLite (tokenizer-diagnostics-* run)

Morphological F1 against MSA tools (Farasa/CamelTools) is INFORMATIONAL only
(optional, guarded by ``--with-morph-alignment``): MSA segmenters flag native
Saudi dialectal clitics as incorrect splits, so divergence from MSA is
expected and correct.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from bayan_slm_engine.tokenizer.bpe_trainer import CLITIC_ALLOWLIST, is_validation_line
from bayan_slm_engine.tokenizer.normalizer import ArabicNormalizer

RARE_TOKEN_THRESHOLD = 50  # < 50 hits = dead/rare (BLUEPRINT §2)
WARN_FERTILITY = 1.8  # provisional
WARN_R_CHAR = 5.0  # provisional
WARN_DEAD_PCT = 3.0  # provisional (target band < 2.0%)
NOTE_L_AVG = (3.5, 4.8)  # provisional informational band


class TokenizerMetricsReport(BaseModel):
    """Pydantic v2 contract for the tokenizer diagnostic report (BLUEPRINT §2)."""

    tokenizer_path: str
    vocab_size: int = Field(gt=0, le=65_536)
    corpus_lines: int = Field(ge=0)
    validation_lines: int = Field(ge=0)
    seed: int
    validation_fraction: float = Field(gt=0.0, lt=1.0)
    fertility: float = Field(ge=0.0)  # content tokens per word (F)
    r_char: float = Field(ge=0.0, le=100.0)  # % single-char fallback, excl. clitic allowlist
    l_avg: float = Field(ge=0.0)  # mean subword length in Arabic characters
    dead_pct: float = Field(ge=0.0, le=100.0)  # % vocab with < 50 hits (full corpus)
    morph_f1: float | None = None  # informational MSA-reference baseline (optional)
    timestamp: datetime


def is_char_fallback(token: str) -> bool:
    """True if ``token`` is a lone non-clitic, non-whitespace character.

    Single-char clitics (و ف ب ل ك س + dialectal ح ع) are intentionally atomic
    tokens for Saudi-dialectal Arabic (BLUEPRINT §2) and are NOT character
    fallbacks. ا is deliberately NOT whitelisted (see ADR in BLUEPRINT §2).
    """
    return len(token) == 1 and not token.isspace() and token not in CLITIC_ALLOWLIST


def fertility(total_content_tokens: int, total_words: int) -> float:
    """Fertility rate: content tokens per word."""
    return total_content_tokens / total_words if total_words else 0.0


def single_char_rate(char_fallback_tokens: int, content_tokens: int) -> float:
    """Single-char fallback rate, percent of content tokens."""
    return 100.0 * char_fallback_tokens / content_tokens if content_tokens else 0.0


def mean_subword_length(total_chars: int, content_tokens: int) -> float:
    """Mean subword length in Arabic characters (codepoints)."""
    return total_chars / content_tokens if content_tokens else 0.0


def dead_token_pct(rare_tokens: int, vocab_size: int) -> float:
    """Dead/rare token percentage of the total vocabulary."""
    return 100.0 * rare_tokens / vocab_size if vocab_size else 0.0


def _iter_normalized_lines(corpus_path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(raw_index, normalized)`` for every non-empty normalized raw line.

    The raw index matches ``BPETrainer``'s indexing so the validation split is
    derived identically in both processes.
    """
    normalizer = ArabicNormalizer()
    with Path(corpus_path).open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            normalized = normalizer.normalize(line)
            if normalized:
                yield index, normalized


def compute_metrics(
    tokenizer: Tokenizer,
    corpus_path: Path,
    *,
    tokenizer_path: Path,
    seed: int = 42,
    validation_fraction: float = 0.02,
    morph_f1: float | None = None,
) -> TokenizerMetricsReport:
    """Compute the full diagnostic report.

    Pass 1 (full corpus): dead/rare token utilization + corpus line count.
    Pass 2 (validation split): fertility, single-char fallback, mean subword
    length. Both passes stream line-by-line; only a vocab-bounded ``Counter``
    is retained, keeping host RAM flat regardless of corpus size.
    """
    vocab_size = len(tokenizer.get_vocab())
    token_counts: Counter[str] = Counter()
    corpus_lines = 0
    for _index, normalized in _iter_normalized_lines(corpus_path):
        corpus_lines += 1
        for tok in tokenizer.encode(normalized, add_special_tokens=False).tokens:
            token_counts[tok] += 1

    total_tokens = total_words = total_chars = char_fallback = 0
    validation_lines = 0
    for index, normalized in _iter_normalized_lines(corpus_path):
        if not is_validation_line(index, seed=seed, fraction=validation_fraction):
            continue
        validation_lines += 1
        total_words += len(normalized.split())
        for tok in tokenizer.encode(normalized, add_special_tokens=False).tokens:
            if tok.isspace():
                continue
            total_tokens += 1
            total_chars += len(tok)
            if is_char_fallback(tok):
                char_fallback += 1

    rare_tokens = sum(1 for count in token_counts.values() if count < RARE_TOKEN_THRESHOLD)
    return TokenizerMetricsReport(
        tokenizer_path=str(Path(tokenizer_path)),
        vocab_size=vocab_size,
        corpus_lines=corpus_lines,
        validation_lines=validation_lines,
        seed=seed,
        validation_fraction=validation_fraction,
        fertility=fertility(total_tokens, total_words),
        r_char=single_char_rate(char_fallback, total_tokens),
        l_avg=mean_subword_length(total_chars, total_tokens),
        dead_pct=dead_token_pct(rare_tokens, vocab_size),
        morph_f1=morph_f1,
        timestamp=datetime.now(UTC),
    )


def report_metrics(
    report: TokenizerMetricsReport,
    *,
    json_out: Path = Path("logs/tokenizer_metrics.json"),
    trackio_run_name: str | None = None,
) -> int:
    """Emit the diagnostic report. NEVER gates the pipeline — always returns 0."""
    print("=== BAYAN TOKENIZER DIAGNOSTIC REPORT ===")
    print(f"  tokenizer            : {report.tokenizer_path}")
    print(f"  vocab_size           : {report.vocab_size}")
    print(f"  corpus_lines         : {report.corpus_lines}")
    print(f"  validation_lines     : {report.validation_lines}")
    print(f"  seed / fraction      : {report.seed} / {report.validation_fraction}")
    print(f"  fertility (F)        : {report.fertility:.3f} tokens/word")
    print(f"  single-char (R_char) : {report.r_char:.2f}% of content tokens")
    print(f"  mean subword (L_avg) : {report.l_avg:.2f} chars")
    print(f"  dead/rare (<50 hits) : {report.dead_pct:.2f}% of vocab")
    print(
        f"  morph F1 (MSA ref.)  : {report.morph_f1 if report.morph_f1 is not None else 'not computed (optional --with-morph-alignment)'}"
    )

    json_out = Path(json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  artifact             : {json_out}")

    warnings: list[str] = []
    if report.fertility > WARN_FERTILITY:
        warnings.append(
            f"⚠️ WARNING: High fertility rate ({report.fertility:.2f} > {WARN_FERTILITY}). "
            "TPOT latency may increase during serving."
        )
    if report.r_char > WARN_R_CHAR:
        warnings.append(
            f"⚠️ WARNING: High single-char fallback rate ({report.r_char:.2f}% > {WARN_R_CHAR}%). "
            "Investigate normalization."
        )
    if report.dead_pct > WARN_DEAD_PCT:
        warnings.append(
            f"⚠️ WARNING: High dead token percentage ({report.dead_pct:.2f}% > {WARN_DEAD_PCT}%). "
            "Consider trimming vocabulary size."
        )
    low, high = NOTE_L_AVG
    if report.l_avg < low or report.l_avg > high:
        warnings.append(
            f"ℹ️ NOTE: Mean subword length {report.l_avg:.2f} outside provisional band "
            f"[{low}, {high}]. Recalibrate via ADR after the first real run."
        )
    for warning in warnings:
        print(warning)

    try:
        import trackio  # trackio ships py.typed (core dependency, always installed)
    except ImportError:
        print("  trackio              : skipped (trackio not installed)")
        return 0

    trackio.init(
        project="bayan-slm-engine",
        name=trackio_run_name or f"tokenizer-diagnostics-{report.timestamp:%Y%m%d-%H%M%S}",
    )
    trackio.log(
        {
            "tokenizer/vocab_size": report.vocab_size,
            "tokenizer/fertility": report.fertility,
            "tokenizer/r_char_pct": report.r_char,
            "tokenizer/l_avg": report.l_avg,
            "tokenizer/dead_pct": report.dead_pct,
            "tokenizer/validation_lines": report.validation_lines,
            "tokenizer/seed": report.seed,
        }
        | ({"tokenizer/morph_f1": report.morph_f1} if report.morph_f1 is not None else {})
    )
    return 0


def _compute_morph_f1() -> float | None:
    """Optional MSA-reference morphological F1 (Farasa/CamelTools).

    Informational baseline only — dialect divergence from MSA is expected
    (BLUEPRINT §2). Returns None when the optional tooling is unavailable;
    the full alignment pipeline lands with the eval suite (Phase 5).
    """
    try:
        import farasapy  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        print("  morph alignment      : skipped (farasapy not installed; optional extras group)")
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (invoked by ``make tokenize``)."""
    parser = argparse.ArgumentParser(
        description="Bayan tokenizer diagnostic report (M1.1; Calculate & Report paradigm)."
    )
    parser.add_argument("--tokenizer", type=Path, required=True, help="Trained tokenizer.json")
    parser.add_argument(
        "--corpus", type=Path, required=True, help="Corpus the tokenizer was trained on"
    )
    parser.add_argument("--json-out", type=Path, default=Path("logs/tokenizer_metrics.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument(
        "--with-morph-alignment",
        action="store_true",
        help="(optional) compute morphological F1 vs MSA tools — informational only",
    )
    args = parser.parse_args(argv)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    report = compute_metrics(
        tokenizer,
        args.corpus,
        tokenizer_path=args.tokenizer,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        morph_f1=_compute_morph_f1() if args.with_morph_alignment else None,
    )
    return report_metrics(report, json_out=args.json_out)


if __name__ == "__main__":
    raise SystemExit(main())
