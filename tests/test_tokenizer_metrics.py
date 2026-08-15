"""M1.1 tokenizer diagnostic-report contract tests (verify_vocab).

SSOT: docs/BLUEPRINT.md §2 — "Calculate & Report" paradigm. The report is a
diagnostic, NEVER a quality gate: ``report_metrics`` always exits 0. Tests
assert metric pure-functions, the 8-char clitic allowlist boundary, and the
triple emission channel (stdout, JSON artifact, trackio) — with trackio
stubbed so the suite stays hermetic (``BAYAN_OFFLINE=1``).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from bayan_slm_engine.tokenizer.bpe_trainer import CLITIC_ALLOWLIST
from bayan_slm_engine.tokenizer.verify_vocab import (
    TokenizerMetricsReport,
    compute_metrics,
    dead_token_pct,
    fertility,
    is_char_fallback,
    mean_subword_length,
    report_metrics,
    single_char_rate,
)
from bayan_slm_engine.tokenizer.verify_vocab import (
    main as verify_main,
)


class _TrackioStub:
    """Hermetic stand-in for trackio (local SQLite) — no disk/network side effects."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def init(self, **kwargs: object) -> None:
        self.calls.append({"init": kwargs})

    def log(self, payload: dict[str, object]) -> None:
        self.calls.append({"log": payload})


@pytest.fixture
def trackio_stub(monkeypatch: pytest.MonkeyPatch) -> _TrackioStub:
    stub = _TrackioStub()
    monkeypatch.setitem(sys.modules, "trackio", stub)
    return stub


class TestCharFallback:
    def test_all_allowlisted_clitics_are_not_fallback(self) -> None:
        assert frozenset({"و", "ف", "ب", "ل", "ك", "س", "ح", "ع"}) == CLITIC_ALLOWLIST
        for clitic in CLITIC_ALLOWLIST:
            assert is_char_fallback(clitic) is False

    def test_alef_is_fallback_despite_normalization(self) -> None:
        # ا is deliberately NOT whitelisted: it is not a clitic, and masking the
        # most frequent letter hides genuine fallback (elongation, byte-fallback).
        assert is_char_fallback("ا") is True

    def test_other_lone_letters_are_fallback(self) -> None:
        assert is_char_fallback("م") is True
        assert is_char_fallback("ن") is True

    def test_whitespace_and_multi_char_never_fallback(self) -> None:
        assert is_char_fallback(" ") is False
        assert is_char_fallback("\t") is False
        assert is_char_fallback("الحين") is False


class TestMetricPureFunctions:
    def test_fertility(self) -> None:
        assert fertility(100, 10) == 10.0
        assert fertility(0, 0) == 0.0

    def test_single_char_rate(self) -> None:
        assert single_char_rate(5, 100) == 5.0
        assert single_char_rate(0, 0) == 0.0

    def test_mean_subword_length(self) -> None:
        assert mean_subword_length(40, 10) == 4.0
        assert mean_subword_length(0, 0) == 0.0

    def test_dead_token_pct(self) -> None:
        assert dead_token_pct(320, 16_000) == 2.0
        assert dead_token_pct(0, 16_000) == 0.0


class TestReportEmission:
    def _make_report(self, tmp_path: Path, **overrides: object) -> TokenizerMetricsReport:
        fields: dict[str, object] = {
            "tokenizer_path": str(tmp_path / "tokenizer.json"),
            "vocab_size": 16_000,
            "corpus_lines": 100,
            "validation_lines": 2,
            "seed": 42,
            "validation_fraction": 0.02,
            "fertility": 1.4,
            "r_char": 1.0,
            "l_avg": 4.2,
            "dead_pct": 1.0,
            "morph_f1": None,
            "timestamp": datetime.now(UTC),
        }
        fields.update(overrides)
        return TokenizerMetricsReport.model_validate(fields)

    def test_report_metrics_exit_zero_and_writes_json(
        self, tmp_path: Path, trackio_stub: _TrackioStub, capsys: pytest.CaptureFixture[str]
    ) -> None:
        json_out = tmp_path / "logs" / "tokenizer_metrics.json"
        report = self._make_report(tmp_path)
        exit_code = report_metrics(report, json_out=json_out, trackio_run_name="test-diagnostics")
        captured = capsys.readouterr()

        assert exit_code == 0  # never gates the pipeline
        assert json_out.exists()
        artifact = json.loads(json_out.read_text(encoding="utf-8"))
        assert artifact["vocab_size"] == 16_000
        assert artifact["fertility"] == 1.4
        assert "BAYAN TOKENIZER DIAGNOSTIC REPORT" in captured.out
        # trackio stub received the tokenizer/* payload under the right run name.
        log_calls = [c["log"] for c in trackio_stub.calls if "log" in c]
        assert len(log_calls) == 1
        assert log_calls[0]["tokenizer/fertility"] == 1.4  # type: ignore[index]
        assert log_calls[0]["tokenizer/r_char_pct"] == 1.0  # type: ignore[index]

    def test_report_soft_warnings_not_exceptions(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        report = self._make_report(tmp_path, fertility=2.5, r_char=9.0, dead_pct=4.0)
        exit_code = report_metrics(
            report, json_out=tmp_path / "m.json", trackio_run_name="warn-run"
        )
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "WARNING" in captured.out  # soft warnings, never exceptions

    def test_cli_main_exits_zero(
        self,
        tmp_path: Path,
        trained_tokenizer_path: Path,
        dialect_corpus_path: Path,
        trackio_stub: _TrackioStub,
    ) -> None:
        json_out = tmp_path / "cli_metrics.json"
        exit_code = verify_main(
            [
                "--tokenizer",
                str(trained_tokenizer_path),
                "--corpus",
                str(dialect_corpus_path),
                "--json-out",
                str(json_out),
                "--seed",
                "42",
            ]
        )
        assert exit_code == 0
        assert json_out.exists()


class TestComputeMetrics:
    def test_end_to_end_report(
        self, tmp_path: Path, trained_tokenizer_path: Path, dialect_corpus_path: Path
    ) -> None:
        tokenizer = Tokenizer.from_file(str(trained_tokenizer_path))
        report = compute_metrics(
            tokenizer,
            dialect_corpus_path,
            tokenizer_path=trained_tokenizer_path,
        )
        assert report.vocab_size == 16_000
        assert report.corpus_lines > 0
        assert report.validation_lines > 0
        assert report.fertility > 0.0
        assert 0.0 <= report.r_char <= 100.0
        assert 0.0 <= report.dead_pct <= 100.0
        assert report.morph_f1 is None  # informational-only, uncomputed by default
        assert report.timestamp is not None
