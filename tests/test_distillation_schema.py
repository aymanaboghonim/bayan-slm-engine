"""M1.3 Frontier API distillation contract tests.

SSOT: docs/BLUEPRINT.md §3 (Grounded Synthetic Distillation) and
docs/EXECUTION_ROADMAP.md Milestone 1.3.

Asserts the distillation contract against the REAL DeepSeek API surface
(verified via api-docs.deepseek.com, 2026-08-16):

* ``deepseek-v4-flash`` via OpenAI-compatible ``https://api.deepseek.com``.
* Thinking mode is ON by default -> the client must explicitly send
  ``{"thinking": {"type": "disabled"}}`` (M1.3 is a schema-restructure task;
  chain-of-thought would add a 4-6x latency/cost tax for zero benefit).
* JSON Output mode: ``response_format={"type": "json_object"}`` + the word
  "json" in the prompt, with ``max_tokens`` set so payloads never truncate.
* Generated dialogues strictly validate against ``MultiTurnDialogue`` (keys
  ``user`` / ``assistant`` / ``dialect``), no non-JSON wrapper text survives.

Dedup guardrails (HARD constraints):

* MinHash LSH (Jaccard >= 0.75, word 5-grams) applies ONLY to synthetic
  dialogue documents — never Tier B acoustic rows, never raw M1.2 rows.
* Strings under 6 words bypass shingling -> exact-match handling only
  (protects زين، نعم، أعوذ بالله، إي نعم from empty-set false positives).
* The ``DedupPipeline`` is typed to accept ``MultiTurnDialogue`` only.

Hermetic: ``httpx.MockTransport`` only, ``BAYAN_OFFLINE=1`` safe, zero network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from bayan_slm_engine.data.distillation import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DedupPipeline,
    DeepSeekClient,
    DeepSeekPaymentError,
    DistillationConfig,
    DistillationInputRow,
    MultiTurnDialogue,
    PromptBuilder,
    ResumeRegistry,
    anchor_hash,
    compute_backoff_delay,
    extract_json,
    filter_todo,
    write_dialogues_jsonl,
)
from bayan_slm_engine.data.ingestion import TierATrainingRow

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_dialogue(user: str = "شلونك اليوم؟", dialect: str = "najdi") -> MultiTurnDialogue:
    return MultiTurnDialogue(user=user, assistant="بخير، الحمد لله", dialect=dialect)


@pytest.fixture
def config() -> DistillationConfig:
    return DistillationConfig(api_key="test-key")


@pytest.fixture
def input_row() -> DistillationInputRow:
    return DistillationInputRow(
        text="ووضّح كلامك يا مغيث",
        speaker_age="Elderly -- كبير في السن",
        speaker_gender="Male",
        speaker_dialect="Najdi",
    )


@pytest.fixture
def tier_a_row() -> TierATrainingRow:
    return TierATrainingRow(
        text="ووضّح كلامك يا مغيث",
        speaker_age="Elderly -- كبير في السن",
        speaker_gender="Male",
        speaker_dialect="Najdi",
        is_multi_speaker=False,
    )


# --------------------------------------------------------------------------- #
# Schema contract (Step A)
# --------------------------------------------------------------------------- #


class TestMultiTurnDialogueSchema:
    def test_valid_dialogue_validates(self) -> None:
        d = MultiTurnDialogue(user="شلونك", assistant="بخير", dialect="najdi")
        assert d.dialect == "najdi"

    def test_dialect_case_insensitive(self) -> None:
        assert MultiTurnDialogue(user="a", assistant="b", dialect="Hijazi").dialect == "hijazi"

    def test_required_keys_enforced(self) -> None:
        with pytest.raises(ValidationError):
            MultiTurnDialogue(user="a", assistant="b")  # type: ignore[call-arg]  # missing dialect

    def test_unknown_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            MultiTurnDialogue(user="a", assistant="b", dialect="najdi", extra="x")  # type: ignore[call-arg]

    def test_dialect_restricted_to_najdi_hijazi(self) -> None:
        with pytest.raises(ValidationError):
            MultiTurnDialogue(user="a", assistant="b", dialect="khaliji")

    def test_empty_turns_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            MultiTurnDialogue(user="", assistant="b", dialect="najdi")


class TestDistillationInputRow:
    def test_maps_from_tier_a_row(self, tier_a_row: TierATrainingRow) -> None:
        row = DistillationInputRow(
            text=tier_a_row.text,
            speaker_age=tier_a_row.speaker_age,
            speaker_gender=tier_a_row.speaker_gender,
            speaker_dialect=tier_a_row.speaker_dialect,
            is_multi_speaker=tier_a_row.is_multi_speaker,
        )
        assert row.text == "ووضّح كلامك يا مغيث"
        assert row.speaker_dialect == "Najdi"
        assert row.is_multi_speaker is False

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            DistillationInputRow(
                text="x",
                speaker_age="a",
                speaker_gender="Male",
                speaker_dialect="Najdi",
                bogus="nope",  # type: ignore[call-arg]
            )


# --------------------------------------------------------------------------- #
# Prompt builder (Step C) — red until implemented
# --------------------------------------------------------------------------- #


class TestPromptBuilder:
    def test_system_prompt_mentions_json_and_example(self) -> None:
        prompt = PromptBuilder().build_system_prompt()
        assert "json" in prompt.lower()
        assert '"user"' in prompt and '"assistant"' in prompt and '"dialect"' in prompt

    def test_user_prompt_conditions_on_demographics(self, input_row: DistillationInputRow) -> None:
        prompt = PromptBuilder().build_user_prompt(input_row)
        assert "Male" in prompt
        assert "Elderly -- كبير في السن" in prompt
        assert "Najdi" in prompt
        assert "ووضّح كلامك يا مغيث" in prompt


# --------------------------------------------------------------------------- #
# DeepSeek client (Step C) — red until implemented
# --------------------------------------------------------------------------- #


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


class TestDeepSeekClientRequestContract:
    def test_request_body_disables_thinking_and_requests_json(
        self, config: DistillationConfig, input_row: DistillationInputRow
    ) -> None:
        captured: list[dict[str, object]] = []
        payload = {"user": "شلونك", "assistant": "بخير", "dialect": "najdi"}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert request.url.path == "/chat/completions"
            assert str(request.url).startswith(DEEPSEEK_BASE_URL)
            captured.append(json.loads(request.content))
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
            )

        client = DeepSeekClient(config, transport=httpx.MockTransport(handler))
        result = run(client.distill(input_row))
        assert result == MultiTurnDialogue(**payload)
        body = captured[0]
        assert body["model"] == DEEPSEEK_MODEL
        assert body["thinking"] == {"type": "disabled"}  # thinking is ON by default — must be off
        assert body["response_format"] == {"type": "json_object"}
        assert isinstance(body["max_tokens"], int)

    def test_rate_limit_honors_retry_after_and_recovers(
        self, config: DistillationConfig, input_row: DistillationInputRow
    ) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            payload = {"user": "شلونك", "assistant": "بخير", "dialect": "najdi"}
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
            )

        client = DeepSeekClient(config, transport=httpx.MockTransport(handler))
        result = run(client.distill(input_row))
        assert result.dialect == "najdi"
        assert calls == 2  # one retry after the 429

    def test_empty_content_is_retried_once(
        self, config: DistillationConfig, input_row: DistillationInputRow
    ) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                # DeepSeek JSON mode may occasionally return empty content.
                return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
            payload = {"user": "شلونك", "assistant": "بخير", "dialect": "hijazi"}
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
            )

        client = DeepSeekClient(config, transport=httpx.MockTransport(handler))
        result = run(client.distill(input_row))
        assert result.dialect == "hijazi"
        assert calls == 2

    def test_non_json_prose_completion_rejected(
        self, config: DistillationConfig, input_row: DistillationInputRow
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "Sure! Here is your dialogue:"}}]}
            )

        client = DeepSeekClient(config, transport=httpx.MockTransport(handler))
        with pytest.raises(ValueError):
            run(client.distill(input_row))

    def test_batch_preserves_order(
        self, config: DistillationConfig, input_row: DistillationInputRow
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = {"user": "شلونك", "assistant": "بخير", "dialect": "najdi"}
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
            )

        client = DeepSeekClient(config, transport=httpx.MockTransport(handler))
        rows = [input_row, input_row.model_copy(deep=True)]
        dialogues, report = run(client.distill_batch(rows))
        assert len(dialogues) == 2
        assert report.succeeded == 2
        assert all(d.dialect == "najdi" for d in dialogues)


# --------------------------------------------------------------------------- #
# Wrapper extraction (Step C) — red until implemented
# --------------------------------------------------------------------------- #


class TestExtractJson:
    def test_strips_markdown_fence(self) -> None:
        raw = '```json\n{"user": "شلونك", "assistant": "بخير", "dialect": "najdi"}\n```'
        assert json.loads(extract_json(raw))["dialect"] == "najdi"

    def test_plain_json_passthrough(self) -> None:
        raw = '{"user": "شلونك", "assistant": "بخير", "dialect": "hijazi"}'
        assert json.loads(extract_json(raw))["dialect"] == "hijazi"

    def test_prose_without_json_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_json("Sure! Here is your dialogue: enjoy!")


# --------------------------------------------------------------------------- #
# Dedup pipeline + guardrails (Step D) — red until implemented
# --------------------------------------------------------------------------- #


class TestDedupGuardrails:
    def test_exact_duplicate_rejected(self) -> None:
        d = make_dialogue()
        kept = DedupPipeline().dedupe([d, d.model_copy(deep=True)])
        assert len(kept) == 1

    def test_near_duplicate_rejected_at_jaccard_0_75(self) -> None:
        # ~48-word doc (in scope for the 50-300 word dialogue guardrail). A
        # single-word change affects 5 shingles: doc=51 words -> S=47
        # shingles, Jaccard = (47-5)/(47+5) = 0.81 >= 0.75.
        base = (
            "قال أبو خالد إن السوق صار مزدحم من بدري والطريق طويل والجو حار اليوم "
            "والحراج مليان ناس والبنزين غالي وكل شيء ارتفع سعره هالأيام والله ما نعرف "
            "وش صاير بالسوق بس الحمد لله على كل حال والرزق على الله والأيام دولاب "
            "والدوامة تدور والمهم إننا نتحمل ونصبر شوي"
        )
        near = base.rsplit(" ", 1)[0] + " زين"  # single-word change
        kept = DedupPipeline().dedupe([make_dialogue(base), make_dialogue(near)])
        assert len(kept) == 1

    def test_distinct_dialogues_kept(self) -> None:
        a = make_dialogue("شلونك اليوم؟")
        b = make_dialogue("وين رايح هالأيام؟")
        kept = DedupPipeline().dedupe([a, b])
        assert len(kept) == 2

    def test_short_text_bypasses_shingling_exact_match_only(self) -> None:
        """<6 words must NOT collapse distinct particles (زين vs نعم)."""
        pipe = DedupPipeline()
        a = make_dialogue("زين")
        b = make_dialogue("نعم")
        kept = pipe.dedupe([a, b])
        assert len(kept) == 2  # no false-positive collision
        # Literally identical short text still dedups exactly:
        kept = pipe.dedupe([a, a.model_copy(deep=True)])
        assert len(kept) == 1

    def test_type_separation_excludes_tier_b_rows(self) -> None:
        """Dedup accepts MultiTurnDialogue only — Tier B rows are structurally excluded."""
        with pytest.raises(TypeError):
            DedupPipeline().dedupe(["cleaned_text row"])  # type: ignore[list-item]

    def test_drift_markers_flagged(self) -> None:
        d = make_dialogue("عايز أروح السوق بدّي أشوف")
        assert DedupPipeline().contains_drift(d) is True
        clean = make_dialogue("أبغى أروح السوق")
        assert DedupPipeline().contains_drift(clean) is False

    def test_native_saudi_particles_pass(self) -> None:
        """عشان/مين/ليش are native Najdi/Hijazi — must NOT flag (M1.3 refinement)."""
        pipe = DedupPipeline()
        assert pipe.contains_drift(make_dialogue("أنا عشان أروح السوق")) is False
        assert pipe.contains_drift(make_dialogue("مين اللي جا؟")) is False
        assert pipe.contains_drift(make_dialogue("ليش تأخرت؟")) is False

    def test_egyptian_levantine_drift_flagged(self) -> None:
        pipe = DedupPipeline()
        assert pipe.contains_drift(make_dialogue("عايز أروح السوق")) is True
        assert pipe.contains_drift(make_dialogue("إزاي حالك؟")) is True
        assert pipe.contains_drift(make_dialogue("بدّي أشوفك")) is True
        assert pipe.contains_drift(make_dialogue("هلق نشوف")) is True

    def test_word_boundaries_prevent_substring_false_positives(self) -> None:
        """بص in بصراحة / شو in شويت must NOT match (word-boundary anchors)."""
        pipe = DedupPipeline()
        assert pipe.contains_drift(make_dialogue("بصراحة الموضوع سهل")) is False
        assert pipe.contains_drift(make_dialogue("شويت تعب")) is False

    def test_kdeh_spelling_variant_not_flagged(self) -> None:
        """كده/كدا are canonicalized by the M1.1 normalizer (كذا) — not drift."""
        pipe = DedupPipeline()
        assert pipe.contains_drift(make_dialogue("كده وكدا")) is False


# --------------------------------------------------------------------------- #
# JSONL round-trip (Step E) — DoD: json.loads per line, zero errors
# --------------------------------------------------------------------------- #


class TestJsonlRoundTrip:
    def test_every_line_parses_with_json_loads(self, tmp_path: Path) -> None:
        dialogues = [
            make_dialogue("شلونك اليوم؟", "najdi"),
            make_dialogue("وين رايح؟", "hijazi"),
        ]
        out = tmp_path / "dialogues.jsonl"
        write_dialogues_jsonl(dialogues, out)
        with out.open(encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)  # DoD: zero decoding errors
            assert set(obj) == {"user", "assistant", "dialect"}


# --------------------------------------------------------------------------- #
# Batch resiliency & resumability (Refinement 2) — red until implemented
# --------------------------------------------------------------------------- #


class TestBackoffComputation:
    def test_missing_retry_after_never_sleeps_zero(self) -> None:
        """429 without Retry-After must fall back to backoff+jitter (> 0)."""
        delay = compute_backoff_delay(0, None, __import__("random").Random(0))
        assert delay >= 1.0

    def test_zero_retry_after_falls_back(self) -> None:
        """Retry-After: 0 must NOT sleep zero seconds (hot-loop guard)."""
        delay = compute_backoff_delay(0, "0", __import__("random").Random(0))
        assert delay >= 1.0

    def test_non_numeric_retry_after_falls_back(self) -> None:
        delay = compute_backoff_delay(0, "abc", __import__("random").Random(0))
        assert delay >= 1.0

    def test_valid_retry_after_honored(self) -> None:
        assert compute_backoff_delay(0, "2.5", __import__("random").Random(0)) == 2.5

    def test_backoff_grows_exponentially(self) -> None:
        rng = __import__("random").Random(0)
        assert compute_backoff_delay(2, None, rng) > compute_backoff_delay(0, None, rng)


class TestDeepSeekPaymentError:
    def test_402_raises_fatal_error(
        self, config: DistillationConfig, input_row: DistillationInputRow
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(402, json={"error": {"message": "insufficient balance"}})

        client = DeepSeekClient(config, transport=httpx.MockTransport(handler))
        with pytest.raises(DeepSeekPaymentError):
            run(client.distill(input_row))

    def test_402_halts_batch(
        self, config: DistillationConfig, input_row: DistillationInputRow
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(402, json={"error": {"message": "insufficient balance"}})

        client = DeepSeekClient(config, transport=httpx.MockTransport(handler))
        rows = [input_row, input_row.model_copy(deep=True)]
        with pytest.raises(DeepSeekPaymentError):
            run(client.distill_batch(rows))


class TestBatchFailureIsolation:
    def test_bad_sample_skipped_others_kept_order_preserved(
        self, config: DistillationConfig, input_row: DistillationInputRow
    ) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(
                    200, json={"choices": [{"message": {"content": "Sure! prose"}}]}
                )
            payload = {"user": f"u{calls['n']}", "assistant": "بخير", "dialect": "najdi"}
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
            )

        client = DeepSeekClient(config, transport=httpx.MockTransport(handler))
        rows = [input_row, input_row.model_copy(deep=True), input_row.model_copy(deep=True)]
        dialogues, report = run(client.distill_batch(rows))
        assert len(dialogues) == 2
        assert report.total == 3
        assert report.succeeded == 2
        assert report.failed == 1
        assert [d.user for d in dialogues] == ["u2", "u3"]  # order preserved
        assert len(report.error_samples) == 1

    def test_fail_fast_aborts_on_first_failure(
        self, config: DistillationConfig, input_row: DistillationInputRow
    ) -> None:
        config = config.model_copy(update={"fail_fast": True})

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "prose"}}]})

        client = DeepSeekClient(config, transport=httpx.MockTransport(handler))
        rows = [input_row, input_row.model_copy(deep=True)]
        with pytest.raises(ValueError):
            run(client.distill_batch(rows))

    def test_semaphore_bounds_in_flight(
        self, config: DistillationConfig, input_row: DistillationInputRow
    ) -> None:
        config = config.model_copy(update={"concurrency": 2})
        lock = asyncio.Lock()
        state = {"in_flight": 0, "max": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            async with lock:
                state["in_flight"] += 1
                state["max"] = max(state["max"], state["in_flight"])
            await asyncio.sleep(0.01)
            payload = {"user": "شلونك", "assistant": "بخير", "dialect": "najdi"}
            async with lock:
                state["in_flight"] -= 1
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps(payload)}}]}
            )

        client = DeepSeekClient(config, transport=httpx.MockTransport(handler))
        rows = [input_row.model_copy(deep=True) for _ in range(4)]
        dialogues, report = run(client.distill_batch(rows))
        assert state["max"] <= 2
        assert report.succeeded == 4


class TestResumeRegistry:
    def test_anchor_hash_deterministic(self, input_row: DistillationInputRow) -> None:
        assert anchor_hash(input_row) == anchor_hash(input_row.model_copy(deep=True))
        assert len(anchor_hash(input_row)) == 32  # blake2b digest_size=16 -> 32 hex

    def test_mark_and_load_roundtrip(self, tmp_path: Path) -> None:
        registry = ResumeRegistry(tmp_path / "dialogues.jsonl.anchors")
        registry.mark("abc")
        registry.mark("def")
        assert registry.load_anchors() == {"abc", "def"}

    def test_filter_todo_excludes_completed(self, input_row: DistillationInputRow) -> None:
        other = input_row.model_copy(update={"text": "مختلف نص"})
        rows = [input_row, other]
        todo = filter_todo(rows, {anchor_hash(input_row)})
        assert todo == [other]
