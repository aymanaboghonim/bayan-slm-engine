# Agent Execution Rules & Engineering Standards — bayan-slm-engine

---

## PART 1: Universal Agent Execution & Human-Paced Audit Protocol

### 1. Communication Rules (Strict Low-Cognitive-Load Constraints)
* **Micro-Reasoning Only:** Total text explanation per turn MUST NOT exceed **60 words**. No long walls of text, no deep chain-of-thought dumps.
* **No Code Sprawls:** Output ONLY the specific modified function or block—never reprint entire unchanged files.
* **Explicit Assumptions:** State system/codebase assumptions *before* executing tools or proposing diffs.

### 2. Mandatory Audit Format
Every single turn MUST follow this exact 4-line format before taking action:
* 🎯 **Intent:** [1 sentence explaining what you are trying to achieve]
* ⚠️ **Key Assumption:** [Max 2 bullet points on what you assume about the codebase/state]
* 🛠️ **Single Action:** [1 line describing the exact tool call or file diff you will run]
* 🛑 **Checkpoint:** "Waiting for approval to execute."

### 3. Step-and-Steer Control Loop
* **Single-Step Limit:** You are strictly forbidden from executing multiple tool calls or multi-file edits in a single turn.
* **Mandatory Pause:** Execute ONE action, emit the Audit Format, and STOP immediately to wait for human confirmation.

---

## PART 2: Universal AI Engineering Standards

1. **No High-Level Monolithic Wrappers:** Do NOT suggest or use `transformers` higher-level trainer wrappers (`TRL`, `AutoModelForCausalLM` trainers), `langchain`, or `llama-index`. Write explicit, native PyTorch (`torch.nn`, `torch.optim`) architecture and training loops.
2. **Pydantic v2 Contracts:** All API payloads, dataset items, and system configuration structures must use Pydantic v2 `BaseModel` with explicit typing and validation.
3. **Hardware-Budgeted Memory Safety:** All PyTorch allocations and forward passes must strictly respect the 8 GB VRAM budget on the RTX 3070 ($\le 4.2\text{ GB}$ VRAM during pretraining, $\le 4.8\text{ GB}$ VRAM during DPO alignment, $\le 2.85\text{ GB}$ VRAM during serving).
4. **Deterministic Safety First:** Enforce schema validation, tensor shape checks, and input normalization via deterministic code (e.g., regex, PyTorch tensor asserts), never via prompt heuristics alone.
5. **Observability Native:** Every model inference step, STT/TTS routing call, and alignment evaluation pass must be wrapped in OpenTelemetry spans and exported to Arize Phoenix.

---

## PART 3: Project Architecture & Single Source of Truth (`bayan-slm-engine`)

* **Single Source of Truth (SSOT):** `docs/BLUEPRINT.md` at the repository root is the authoritative technical specification. Before implementing any module, streamer, or model class, read the relevant section in `docs/BLUEPRINT.md` to verify exact hyperparameters, tensor dimensions, and model checkpoints.

* **Execution Roadmap:** `docs/EXECUTION_ROADMAP.md` is the authoritative milestone-by-milestone sequence. Implementation must proceed strictly phase-by-phase (Phase 1 through Phase 6) using the Contract-First SDAI loop.

* **Project Identity:** `bayan-slm-engine` | Python 3.12+ via `uv` | PyTorch (`torch.nn`)
* **Hardware Limit:** Single RTX 3070 (8 GB VRAM) + WSL2 Ubuntu (12 GB System RAM limit).
* **Text Engine (Tier A):** `tiiuae/Falcon-H1-0.5B-Base` (Transformer + Mamba-2 hybrid, GQA, RoPE) adapted via domain-adaptive continued pretraining and SFT on ~15M–30M SDAIA tokens + synthetic pairs.
* **Acoustic Engine (Tier B):** Decoupled `oddadmix/whisper-small-arabic-dialectal` (244M STT) + CATT neural diacritizer + `wasmdashai/vits-ar-sa-huba` (145 MB Saudi TTS).
* **Alignment & Serving:** Native PyTorch DPO + 8-bit AdamW; FastAPI async streaming with static KV-cache; local Arize Phoenix telemetry via OpenTelemetry.

## PART 4: Spec-Driven Agent Iteration (SDAI) Workflow

Execute tasks using the SDAI framework. Never attempt unconstrained monolithic implementations.

1. **Scope Boundaries:**
   * **Plan / Read Mode:** UNLIMITED search, inspection, and read scope across the entire workspace.
   * **Phase 0 Exception:** Repo scaffolding and config setup (e.g., `pyproject.toml`, `Makefile`, `ci.yml`) are permitted to create/modify all necessary setup files in a single pass.
   * **Feature Implementation Mode:** Edits are restricted to **1 logical feature slice** (typically 1 source module + 1 interface contract + 1 test file).

* **Milestone Gating:** You are strictly restricted to the current active milestone defined in `docs/EXECUTION_ROADMAP.md`. Do not jump ahead to subsequent phases or milestones without explicit human approval.

2. **Contract-First Scaffolding (TDD):**
   Before writing internal implementation logic for any feature module:
   * **Step A:** Write the abstract class / module interface (Pydantic v2 schemas, signatures, and `raise NotImplementedError`).
   * **Step B:** Write the corresponding `pytest` suite asserting tensor shapes, hardware boundaries, and contract responses.
   * **Step C:** Pause. Only implement internal logic *after* the failing test is committed.

3. **Machine-Executable Definition of Done (DoD):**
   A feature task is strictly "Done" only when:
   * `uv run ruff check .` and `uv run mypy src/` pass with zero warnings.
   * `uv run pytest tests/` passes using dummy CPU tensors.
   * Host RAM and VRAM footprint assertions pass.
