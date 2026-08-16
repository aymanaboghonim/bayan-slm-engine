# Agent Execution Rules & Engineering Standards — bayan-slm-engine

## PART 1: Low-Cognitive-Load HITL Protocol

### 1. Autonomy & Boundary Separation
* **Silent Context Gathering (Unlimited):** You may run chained read, search, and directory inspection tools autonomously without asking for permission. Build your context completely before interacting.
* **Guarded Mutations (Strict HITL):** You MUST NOT edit files or invoke terminal commands without explicit user approval.
* **Direct Workspace Application:** Apply code modifications directly to target workspace files. Never dump full files or large code diffs into the chat UI. Chat is strictly for decision logic and execution gating.

### 2. Micro-Reasoning & Communication Limits
* **Word Limit:** Chat output must not exceed **60 words** per turn. Omit conversational pleasantries, chain-of-thought dumps, and unsolicited summaries.
* **Mandatory Audit Format:** Before requesting approval for any write, edit, or terminal action, emit this exact 4-line format:
  * 🎯 **Intent:** [1 concise sentence on the targeted goal]
  * ⚠️ **Key Assumption:** [Max 2 concise assumptions regarding codebase/state]
  * 🛠️ **Proposed Action:** [Exact file mutation path or terminal command to run]
  * 🛑 **Checkpoint:** "Waiting for approval to execute."

### 3. Step-and-Steer Loop
* Limit execution to **one mutating action per turn** (1 file edit or 1 terminal command).
* After approval, execute the single action and immediately pause for human review before proceeding to the next step.

---

## PART 2: Universal AI Engineering Standards

1. **Native Implementations Only:** Monolithic framework wrappers are strictly forbidden (`langchain`, `llama-index`, `autogen`, `TRL`, and high-level `AutoModelForCausalLM` trainers). Use explicit, native PyTorch (`torch.nn`, `torch.optim`) and custom async state machines.
2. **Pydantic v2 Data Contracts:** All dataset schemas, internal configurations, and API payloads must use Pydantic v2 `BaseModel` with explicit typing and validation.
3. **Hardware Budget Enforcement:** Respect the single RTX 3070 8 GB VRAM limit ($\le 4.2\text{ GB}$ pretraining, $\le 4.8\text{ GB}$ DPO, $\le 2.85\text{ GB}$ serving). Use explicit garbage collection, CUDA cache clearing, and static allocation patterns where required.
4. **Deterministic Validation:** Enforce tensor shapes, input sanitization, and interface validation via deterministic code assertions and regex, not prompt heuristics.
5. **Telemetry Native:** Wrap inference calls, audio pipeline stages, and alignment evaluations in OpenTelemetry spans targeting Arize Phoenix.

---

## PART 3: Source of Truth & Project Context

* **Authoritative Architecture:** `docs/BLUEPRINT.md` governs model configurations (Falcon-H1-0.5B-Base, STT/TTS routing, DPO alignment), tensor shapes, and serving specs. Always inspect this file before modifying model code.
* **Authoritative Sequence:** `docs/EXECUTION_ROADMAP.md` dictates the active phase and milestone. Do not implement features outside the current phase.
* **Environment:** Python 3.12+ managed via `uv`.

---

## PART 4: Spec-Driven Agent Iteration (SDAI) Workflow

Implement features strictly through the Test-Driven Development (TDD) lifecycle:

1. **State 1: Interface Contract**
   * Create or update abstract interface classes and Pydantic schemas.
   * Method stubs must raise `NotImplementedError`.
2. **State 2: Contract Test Suite**
   * Write `pytest` test suites validating interface contracts, tensor shapes, and memory constraints using dummy tensors on CPU.
3. **State 3: Internal Implementation**
   * Implement internal module logic only after failing tests are in place.
4. **Definition of Done (DoD) Verification:**
   * Propose running: `uv run ruff check .`
   * Propose running: `uv run mypy src/`
   * Propose running: `uv run pytest tests/`
   * Confirm zero errors/warnings before closing the task.
