"""Frontier API distillation pipeline (M1.3).

SSOT: docs/BLUEPRINT.md §3 (Grounded Synthetic Distillation) and
docs/EXECUTION_ROADMAP.md Milestone 1.3 — Frontier API Distillation Pipeline.

Converts routed Tier A rows (``TierATrainingRow`` from M1.2) into structured
multi-turn JSON dialogue pairs via the DeepSeek API (``deepseek-v4-flash``,
OpenAI-compatible ``https://api.deepseek.com/chat/completions``).

**DeepSeek specifics (verified against api-docs.deepseek.com, 2026-08-16):**

* Thinking mode is **enabled by default** — M1.3 is a schema-restructure task
  that needs no chain-of-thought, so the request body explicitly disables it:
  ``{"thinking": {"type": "disabled"}}`` (avoids a 4–6x latency/cost tax).
* JSON Output mode: ``response_format={"type": "json_object"}`` plus the word
  "json" and an example in the prompt (per DeepSeek docs).
* ``max_tokens`` is set explicitly so the JSON payload is never truncated.
* The API may occasionally return empty content — one bounded retry handles it.

**Deduplication scope guardrails (HARD constraints):**

* MinHash LSH (Jaccard >= 0.75, word 5-grams) applies **only** to synthetic
  multi-turn JSON dialogues (full documents, 50–300+ words) to purge repetitive
  frontier API tropes, structural boilerplate, and hallucinated templates.
* **Never** applied to Tier B acoustic rows — speech models require identical
  phrases across diverse speakers to learn acoustic invariance.
* **Never** applied to raw M1.2 SADA22 rows — authentic turns stay intact.
* **Short-text guardrail:** strings under 6 words bypass word 5-gram shingling
  and fall back to exact-match handling only — protecting authentic short
  Arabic discourse particles (زين، نعم، أعوذ بالله، إي نعم) from collapsing into
  empty shingle sets or false-positive collisions.

**Lexical drift hardening (M1.3 refinement):** the drift denylist contains
only unambiguous Egyptian/Levantine particles (عايز، إزاي، بدّي، هلق، …) with
word-boundary anchors. Native Saudi particles (عشان، مين، ليش) and the spoken
spelling variants (كده/كدا) are never rejected — variants are canonicalized to
كذا by the M1.1 normalizer. ``هيك`` is retained per ADR (mild over-purge,
DMPR-friendly). A soft, local-only SADA22 dispersion validator
(``sada22_lexicon.py``) flags low-dispersion tokens; it never gates hermetic CI.

**Batch hardening (M1.3 refinement 2):** a single long-lived ``httpx`` client
preserves the TCP/TLS socket pool across requests; bounded concurrency
(``asyncio.Semaphore``, default 8) with per-sample failure isolation (skip +
``DistillationReport`` counts, or ``--fail-fast`` for CI/smoke runs); HTTP 429
honors ``Retry-After`` with exponential-backoff + jitter fallback (never a
zero-second hot loop); HTTP 402 (payment exhaustion) is fatal with an
actionable top-up message; output streams to JSONL in append mode with a
sidecar anchor registry for zero-loss resume; near-duplicate detection uses an
inverted MinHash LSH index (exact Jaccard >= 0.75 remains the arbiter).

The ``DedupPipeline`` is typed to accept ``MultiTurnDialogue`` objects only, so
Tier B rows and raw SADA22 rows are structurally excluded at the type level.

Hermetic CI: every test uses ``httpx.MockTransport`` — ``BAYAN_OFFLINE=1`` safe,
zero network. ``DEEPSEEK_API_KEY`` is read only for local CLI runs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import random
import re
import sys
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from bayan_slm_engine.data.ingestion import (
    FieldRouter,
    iter_validated_rows,
    read_sada_metadata,
)

#: Default OpenAI-compatible endpoint for DeepSeek (verified 2026-08-16).
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
#: Default model alias — resolves to DeepSeek-V4-Flash-0731.
DEEPSEEK_MODEL = "deepseek-v4-flash"

#: Dialects the distillation pipeline emits (normalized lowercase).
Dialect = Literal["najdi", "hijazi"]

#: Unambiguous non-Saudi lexical drift (Egyptian & Levantine particles).
#: Permitted native Saudi particles: عشان, علشان, ليش, مين, كذا/كده (handled
#: by the M1.1 normalizer, never rejected). Word-boundary anchors prevent
#: substring false positives (بص != بصراحة, شو != شويت).
#: ``هيك`` is deliberately retained in the hard denylist (M1.3 ADR): it has
#: documented Hijazi presence (هيك؟، هيك السالفة) but the mild over-purge is
#: accepted to keep dialect-purity metrics strict (M5.1 DMPR).
DRIFT_MARKERS_RE = re.compile(
    r"(?:^|\s)("
    # Egyptian markers
    r"عايز|عاوز|إزاي|ازاي|بص|بصّ|كويس خالص|أوي|اوي|دلوقتي|"
    # Levantine markers
    r"بدّي|بدي|شو|هلّق|هلق|هيك|شو بدك|مشان هيك|شلونك خاي"
    r")(?:\s|[.,،!؟]|$)"
)

#: Diacritics may survive in Tier A-anchored dialogue text — strip them before
#: drift matching so ``عايزْ`` still matches ``عايز`` (deterministic regex).
_DIACRITICS_RE = re.compile(r"[\u0640\u064b-\u065f\u0670\u0671\u06d6-\u06ed]")


def _strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics/orthographic marks (deterministic)."""
    return _DIACRITICS_RE.sub("", text)


def _canonical_text(dialogue: MultiTurnDialogue) -> str:
    """Diacritic-insensitive canonical form for exact-dup matching."""
    return " ".join(_strip_diacritics(f"{dialogue.user} {dialogue.assistant}").split())


def compute_backoff_delay(attempt: int, retry_after: str | None, rng: random.Random) -> float:
    """Backoff delay for a failed attempt — never zero.

    HTTP 429 ``Retry-After`` is honored when valid and positive; missing, zero,
    or non-numeric values fall back to exponential backoff + jitter (a ``0``
    or absent header must never cause a hot retry loop).
    """
    if retry_after is not None:
        try:
            delay = float(retry_after)
            if delay > 0:
                return delay
        except ValueError:
            pass
    return float(min(2**attempt, 8)) + rng.random() * 0.5


class DistillationInputRow(BaseModel):
    """Contract for one distilled input, mapped from ``TierATrainingRow``.

    ``extra="forbid"`` mirrors the M1.2 ingestion contracts — any unexpected
    field is a contract violation.
    """

    model_config = {"extra": "forbid"}

    text: str = Field(min_length=1)
    speaker_age: str
    speaker_gender: str
    speaker_dialect: str
    is_multi_speaker: bool = False

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()


class MultiTurnDialogue(BaseModel):
    """Structured 2-turn JSON dialogue distilled from one SADA22 anchor.

    Strict keys ``user`` / ``assistant`` / ``dialect`` (roadmap M1.3 DoD).
    ``dialect`` is normalized case-insensitively (``Najdi`` -> ``najdi``).
    """

    model_config = {"extra": "forbid"}

    user: str = Field(min_length=1)
    assistant: str = Field(min_length=1)
    dialect: Dialect

    @field_validator("dialect", mode="before")
    @classmethod
    def _lowercase_dialect(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class DeepSeekPaymentError(RuntimeError):
    """HTTP 402 — DeepSeek balance exhausted.

    Fatal and non-retryable: the batch halts cleanly (active workers are
    cancelled) and the operator receives an actionable top-up message.
    """


class DistillationReport(BaseModel):
    """Structured batch outcome for provenance tracking (50K-row runs).

    ``succeeded_indices`` maps output dialogues back to input row indices
    (order-preserving); ``error_samples`` keeps bounded failure provenance.
    """

    model_config = {"extra": "forbid"}

    total: int = Field(ge=0)
    succeeded: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    drift_purged: int = Field(default=0, ge=0)
    succeeded_indices: list[int] = Field(default_factory=list)
    error_samples: list[str] = Field(default_factory=list)


class DistillationConfig(BaseSettings):
    """DeepSeek client configuration, driven by the ``DEEPSEEK_*`` env block.

    ``DEEPSEEK_API_KEY`` is required for local CLI runs; tests construct the
    config directly with a dummy key and a mock transport (hermetic).
    """

    model_config = SettingsConfigDict(env_prefix="DEEPSEEK_")

    api_key: str = Field(min_length=1)
    base_url: str = DEEPSEEK_BASE_URL
    model: str = DEEPSEEK_MODEL
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    #: Thinking mode is enabled by default on DeepSeek — M1.3 disables it.
    thinking_disabled: bool = True
    #: Bounded retry budget for 429/5xx/empty-content responses.
    max_retries: int = Field(default=3, ge=0, le=10)
    #: Bounded async worker pool for batch distillation (1-16).
    concurrency: int = Field(default=8, ge=1, le=16)
    #: Abort the batch on the first failing sample (CI / smoke runs).
    fail_fast: bool = False
    #: Seed for the backoff-jitter RNG — deterministic, hermetic tests.
    backoff_seed: int = Field(default=0, ge=0)


class PromptBuilder:
    """Builds the dynamic demographic-conditioned JSON dialogue prompt.

    Restructuring-only instruction (BLUEPRINT §3 ADR): the model receives an
    authentic SADA22 anchor and demographic tags, and rephrases into a 2-turn
    JSON dialogue — it never invents facts or switches dialect. The word
    "json" plus an example appear in the prompt (DeepSeek JSON-mode docs).
    """

    _SYSTEM_TEMPLATE = """\
You are an Arabic dialogue restructurer. You receive one authentic SADA22 \
transcript anchor and its demographic metadata. Rephrase it into a short \
2-turn JSON dialogue in the same Saudi dialect. Never invent new facts, never \
switch dialect, never add English.

Output ONLY a JSON object with exactly three keys:
- "user": the first turn (question or statement)
- "assistant": the reply
- "dialect": the lowercase dialect label ("najdi" or "hijazi")

Example JSON:
{"user": "شلونك اليوم؟", "assistant": "بخير، الحمد لله", "dialect": "najdi"}"""

    def build_system_prompt(self) -> str:
        """System prompt: restructure SADA22 anchors, never generate from scratch."""
        return self._SYSTEM_TEMPLATE

    def build_user_prompt(self, row: DistillationInputRow) -> str:
        """Persona-conditioned user prompt (gender/age/dialect conditioning)."""
        persona = (
            f"Persona: {row.speaker_gender}, {row.speaker_age}, speaking {row.speaker_dialect}."
        )
        if row.is_multi_speaker:
            persona += " (Multi-speaker transcript — preserve the turn-taking.)"
        return f"{persona}\nAnchor transcript: {row.text}\nProduce the JSON dialogue."


class DeepSeekClient:
    """Raw OpenAI-compatible ``httpx`` client for ``deepseek-v4-flash``.

    **Long-lived connection lifecycle (M1.3 hardening):** one lazy
    ``httpx.AsyncClient`` is created per instance and reused across all
    requests — preserving the TCP/TLS socket pool (no per-call handshakes, no
    OS socket exhaustion on 50K-row batches). Use it as an async context
    manager (``async with DeepSeekClient(config) as client:``) or call
    ``aclose()`` explicitly. The injected ``transport`` keeps hermetic tests
    offline (``httpx.MockTransport``, ``BAYAN_OFFLINE=1``).

    Retry policy (bounded by ``config.max_retries``):

    * HTTP 429: honor ``Retry-After``; if missing/zero/non-numeric fall back
      to exponential backoff + jitter (never ``sleep(0)`` — hot-loop guard).
    * HTTP 402: fatal — raises ``DeepSeekPaymentError`` (payment exhaustion).
    * HTTP 5xx / transport errors: exponential backoff + jitter.
    * Empty content (a DeepSeek JSON-mode quirk): retried within the budget.
    * Malformed prose / schema failures: NOT retried — the sample fails fast
      (deterministic); the batch layer logs and skips it.
    """

    def __init__(
        self,
        config: DistillationConfig,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily create the long-lived client; reuse across requests."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url, transport=self._transport
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying connection pool (no-op if never opened)."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> DeepSeekClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _build_request(self, row: DistillationInputRow) -> tuple[dict[str, object], dict[str, str]]:
        """OpenAI-compatible chat body + auth headers (thinking off, JSON mode)."""
        body: dict[str, object] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": PromptBuilder().build_system_prompt()},
                {"role": "user", "content": PromptBuilder().build_user_prompt(row)},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self._config.max_tokens,
            "thinking": {"type": "disabled" if self._config.thinking_disabled else "enabled"},
        }
        return body, {"Authorization": f"Bearer {self._config.api_key}"}

    async def distill(self, row: DistillationInputRow) -> MultiTurnDialogue:
        """One row -> one validated ``MultiTurnDialogue`` (see class docstring)."""
        body, headers = self._build_request(row)
        rng = random.Random(self._config.backoff_seed)
        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                client = await self._get_client()
                response = await client.post("/chat/completions", json=body, headers=headers)
            except httpx.TransportError as exc:
                last_error = exc
                await asyncio.sleep(compute_backoff_delay(attempt, None, rng))
                continue
            if response.status_code == 402:
                raise DeepSeekPaymentError(
                    "DeepSeek API returned HTTP 402 (insufficient balance). Top up at "
                    "platform.deepseek.com and re-run — completed anchors are already "
                    "flushed to the sidecar registry, so the batch resumes at sample N+1."
                )
            if response.status_code == 429:
                await asyncio.sleep(
                    compute_backoff_delay(attempt, response.headers.get("Retry-After"), rng)
                )
                continue
            if response.status_code >= 500:
                last_error = httpx.HTTPStatusError(
                    f"DeepSeek HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
                await asyncio.sleep(compute_backoff_delay(attempt, None, rng))
                continue
            try:
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            except (KeyError, ValueError) as exc:  # malformed envelope — retryable
                last_error = exc
                await asyncio.sleep(compute_backoff_delay(attempt, None, rng))
                continue
            if not content.strip():  # DeepSeek JSON mode may return empty — retry
                await asyncio.sleep(compute_backoff_delay(attempt, None, rng))
                continue
            # Parse + validate: NON-retryable — malformed prose won't self-correct.
            return MultiTurnDialogue.model_validate_json(extract_json(content))
        if last_error is not None:
            raise last_error
        raise RuntimeError("distillation failed: max retries exhausted")

    async def distill_batch(
        self, rows: list[DistillationInputRow]
    ) -> tuple[list[MultiTurnDialogue], DistillationReport]:
        """Bounded-concurrency batch -> (dialogues, report), input order preserved.

        * ``asyncio.Semaphore(config.concurrency)`` caps in-flight requests.
        * Indexed ``asyncio.gather`` preserves input order (row i -> dialogue i).
        * Per-sample failures are logged and skipped unless ``fail_fast``.
        * ``DeepSeekPaymentError`` is ALWAYS fatal — payment exhaustion halts
          the batch cleanly (gather cancels remaining workers).
        """
        semaphore = asyncio.Semaphore(self._config.concurrency)
        report = DistillationReport(total=len(rows))

        async def _one(
            index: int, row: DistillationInputRow
        ) -> tuple[int, MultiTurnDialogue | Exception]:
            async with semaphore:
                try:
                    return index, await self.distill(row)
                except Exception as exc:
                    if self._config.fail_fast or isinstance(exc, DeepSeekPaymentError):
                        raise
                    return index, exc

        outcomes = await asyncio.gather(*(_one(i, row) for i, row in enumerate(rows)))

        dialogues: list[MultiTurnDialogue] = []
        succeeded_indices: list[int] = []
        error_samples: list[str] = []
        for index, outcome in outcomes:
            if isinstance(outcome, Exception):
                report.failed += 1
                error_samples.append(f"row {index}: {type(outcome).__name__}: {outcome}")
                continue
            dialogues.append(outcome)
            succeeded_indices.append(index)
        report.succeeded = len(dialogues)
        report.succeeded_indices = succeeded_indices
        report.error_samples = error_samples
        return dialogues, report


class DedupPipeline:
    """Regex drift purge + MinHash LSH near-duplicate filtering.

    Scope guardrails (BLUEPRINT §3 + roadmap M1.3): applies to synthetic
    dialogue documents only; short texts (< 6 words) use exact-match handling.
    """

    def __init__(self, jaccard_threshold: float = 0.75, ngram: int = 5, min_words: int = 6) -> None:
        self._jaccard_threshold = jaccard_threshold
        self._ngram = ngram
        self._min_words = min_words

    def contains_drift(self, dialogue: MultiTurnDialogue) -> bool:
        """True if the dialogue contains Levantine/Egyptian drift markers."""
        text = _strip_diacritics(f"{dialogue.user} {dialogue.assistant}")
        return DRIFT_MARKERS_RE.search(text) is not None

    def is_near_duplicate(self, dialogue: MultiTurnDialogue, seen: set[str]) -> bool:
        """Exact-dup check against a set of canonical dialogue texts.

        Near-duplicate (Jaccard) detection needs shingle sets and lives in
        ``dedupe``; this public method covers the exact-match path.
        """
        return _canonical_text(dialogue) in seen

    def dedupe(self, dialogues: list[MultiTurnDialogue]) -> list[MultiTurnDialogue]:
        """Order-preserving dedup: drift purge + exact + near-dup rejection.

        **Scope guardrails (HARD, BLUEPRINT §3):** synthetic dialogue
        documents only. Short texts (< ``min_words``) bypass shingling and use
        exact matching only. The input type (``MultiTurnDialogue``) excludes
        Tier B rows and raw M1.2 rows at the type level; a runtime
        ``TypeError`` guards against misuse (deterministic safety).

        **Inverted LSH index (M1.3 hardening):** near-duplicate candidates are
        retrieved via a hash-bucket inverted index (``hash -> [doc_id]``)
        instead of a full O(N) linear scan; exact Jaccard remains the
        deterministic final arbiter (Jaccard >= 0.75 contract).
        """
        kept: list[MultiTurnDialogue] = []
        seen_exact: set[str] = set()
        docs: list[tuple[str, set[bytes], list[int]]] = []
        index: dict[int, list[int]] = {}
        for dialogue in dialogues:
            if not isinstance(dialogue, MultiTurnDialogue):
                raise TypeError(f"dedupe expects MultiTurnDialogue, got {type(dialogue).__name__}")
            if self.contains_drift(dialogue):
                continue  # regex purge — hallucinated cross-dialect drift
            canonical = _canonical_text(dialogue)
            if canonical in seen_exact:
                continue
            seen_exact.add(canonical)
            shingles = self._shingles(canonical)
            signature = self._signature(shingles) if shingles else []
            if shingles and self._is_near_duplicate(shingles, signature, docs, index):
                continue
            doc_id = len(docs)
            docs.append((canonical, shingles, signature))
            for band_hash in signature:
                index.setdefault(band_hash, []).append(doc_id)
            kept.append(dialogue)
        return kept

    def _shingles(self, text: str) -> set[bytes]:
        """Word n-gram shingles; empty set = short-text guardrail (exact only)."""
        words = text.split()
        if len(words) < self._min_words:
            return set()
        return {
            " ".join(words[i : i + self._ngram]).encode("utf-8")
            for i in range(len(words) - self._ngram + 1)
        }

    def _signature(self, shingles: set[bytes], rows: int = 16) -> list[int]:
        """Min-hash LSH signature: one keyed blake2b min-hash per row.

        Deterministic (fixed per-row keys). Band size is 1 (each hash is its
        own inverted-index bucket), so a Jaccard >= 0.75 pair always shares at
        least one bucket — the exact-Jaccard contract never depends on
        probabilistic band agreement.
        """
        return [
            min(
                int.from_bytes(
                    hashlib.blake2b(shingle, digest_size=8, key=row.to_bytes(4, "little")).digest(),
                    "little",
                )
                for shingle in shingles
            )
            for row in range(rows)
        ]

    def _is_near_duplicate(
        self,
        shingles: set[bytes],
        signature: list[int],
        docs: list[tuple[str, set[bytes], list[int]]],
        index: dict[int, list[int]],
    ) -> bool:
        """True when exact Jaccard >= threshold against any candidate doc.

        Candidates are retrieved from the inverted index (docs sharing any
        LSH bucket) — O(1) per hash — then exact Jaccard is the deterministic
        final arbiter (Jaccard = 0.75 contract).
        """
        candidates: set[int] = set()
        for band_hash in signature:
            candidates.update(index.get(band_hash, ()))
        for doc_id in candidates:
            _, prior_shingles, _ = docs[doc_id]
            if not prior_shingles:
                continue
            intersection = len(shingles & prior_shingles)
            union = len(shingles | prior_shingles)
            if union and intersection / union >= self._jaccard_threshold:
                return True
        return False


def extract_json(content: str) -> str:
    """Strip non-JSON wrapper text (e.g., ```json fences) from a completion.

    Returns the extracted JSON object verbatim; raises ``ValueError`` when no
    balanced JSON object is present (defensive guard against prose-only
    completions). A hand-rolled brace matcher keeps it dependency-free and
    respects string literals containing braces.
    """
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in completion")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("unbalanced JSON object in completion")


def write_dialogues_jsonl(dialogues: list[MultiTurnDialogue], path: Path) -> None:
    """Append dialogues as UTF-8 JSONL, flushing after every line.

    Append mode + per-line flush is the zero-loss resume primitive: a crash
    mid-batch never loses completed samples, and the sidecar anchor registry
    (``ResumeRegistry``) lets a re-run resume at sample N+1. Pydantic v2
    ``model_dump_json`` writes non-ASCII as literal UTF-8, so Arabic text is
    human-readable on disk; every line round-trips via ``json.loads``
    (roadmap M1.3 DoD).
    """
    with path.open("a", encoding="utf-8") as handle:
        for dialogue in dialogues:
            handle.write(dialogue.model_dump_json() + "\n")
            handle.flush()


def anchor_hash(row: DistillationInputRow) -> str:
    """Deterministic resume anchor: blake2b over text + demographics.

    Mirrors the M1.1 ``blake2b`` hashing convention — immune to
    ``PYTHONHASHSEED``, deterministic across runs.
    """
    payload = "\n".join(
        (
            row.text,
            row.speaker_age,
            row.speaker_gender,
            row.speaker_dialect,
            str(row.is_multi_speaker),
        )
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


class ResumeRegistry:
    """Sidecar anchor registry for zero-loss resumable distillation.

    The output JSONL keeps its 3-key DoD (``user``/``assistant``/``dialect``);
    completed anchors live in a parallel ``<output>.anchors`` file
    (append-only, one blake2b hex per line) so a re-run can skip already
    distilled rows without re-calling the API.
    """

    def __init__(self, anchors_path: Path) -> None:
        self._path = anchors_path

    def load_anchors(self) -> set[str]:
        """All completed anchors (empty set when the file does not exist)."""
        if not self._path.exists():
            return set()
        with self._path.open(encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}

    def mark(self, anchor: str) -> None:
        """Record one completed anchor (append + flush — crash-safe)."""
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(anchor + "\n")
            handle.flush()


def filter_todo(rows: list[DistillationInputRow], anchors: set[str]) -> list[DistillationInputRow]:
    """Rows not yet distilled — resume from sample N+1 (order preserved)."""
    return [row for row in rows if anchor_hash(row) not in anchors]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: distill Tier A rows -> deduped JSONL (local runs only).

    Requires a real SADA22 metadata parquet and ``DEEPSEEK_API_KEY`` in the
    environment; hermetic CI never invokes this (``BAYAN_OFFLINE=1``).

    * **Resumable:** completed anchors live in a ``<output>.anchors`` sidecar;
      re-runs skip already-distilled rows (zero loss on crash).
    * **Failure semantics:** skip + log + ``DistillationReport`` counts by
      default; ``--fail-fast`` aborts on the first failing sample.
    * HTTP 402 (payment exhaustion) halts with an actionable top-up message.
    """
    parser = argparse.ArgumentParser(
        prog="bayan-distill",
        description="M1.3: distill Tier A SADA22 rows into deduped JSON dialogue pairs.",
    )
    parser.add_argument("--input", type=Path, required=True, help="SADA22 metadata parquet path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/dialogues.jsonl"),
        help="output JSONL path (append mode; sidecar .anchors tracks resume)",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="bounded async workers (1-16)")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="abort the batch on the first failing sample (CI/smoke runs)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=0, help="cap on routed Tier A rows (0 = all)"
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=0,
        help="if >0, run the soft SADA22 dispersion report (local-only)",
    )
    args = parser.parse_args(argv)

    frame = read_sada_metadata(args.input)
    validated = iter_validated_rows(frame)
    router = FieldRouter()
    rows: list[DistillationInputRow] = []
    for row in validated:
        routed = router.route_tier_a(row)
        if routed is None:
            continue
        rows.append(
            DistillationInputRow(
                text=routed.text,
                speaker_age=routed.speaker_age,
                speaker_gender=routed.speaker_gender,
                speaker_dialect=routed.speaker_dialect,
                is_multi_speaker=routed.is_multi_speaker,
            )
        )
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    registry = ResumeRegistry(args.output.with_name(args.output.name + ".anchors"))
    todo = filter_todo(rows, registry.load_anchors())
    print(f"rows: {len(rows)} total, {len(rows) - len(todo)} already distilled, {len(todo)} to go")

    config = DistillationConfig(concurrency=args.concurrency, fail_fast=args.fail_fast)

    async def _run_batch() -> tuple[list[MultiTurnDialogue], DistillationReport]:
        async with DeepSeekClient(config) as client:
            return await client.distill_batch(todo)

    try:
        dialogues, report = asyncio.run(_run_batch())
    except DeepSeekPaymentError as exc:
        print(f"fatal: {exc}", file=sys.stderr)
        return 2

    before_dedup = len(dialogues)
    dialogues = DedupPipeline().dedupe(dialogues)
    report.drift_purged = before_dedup - len(dialogues)
    write_dialogues_jsonl(dialogues, args.output)
    for index in report.succeeded_indices:
        registry.mark(anchor_hash(todo[index]))

    if args.min_speakers > 0:
        # Lazy import avoids a module-level cycle (sada22_lexicon imports
        # MultiTurnDialogue from this module).
        from bayan_slm_engine.data.sada22_lexicon import Sada22LexiconValidator

        validator = Sada22LexiconValidator(min_speakers=args.min_speakers)
        dispersion = validator.build_dispersion(validated)
        flagged: dict[str, int] = {}
        for dialogue in dialogues:
            for token in validator.validate_dialogue(dialogue, dispersion):
                flagged[token] = flagged.get(token, 0) + 1
        if flagged:
            print(f"[soft] low-dispersion tokens (< {args.min_speakers} recordings):")
            for token, count in sorted(flagged.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {token!r}: {count} dialogue(s)")

    print(
        f"total={report.total} succeeded={report.succeeded} failed={report.failed} "
        f"drift_purged={report.drift_purged} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
