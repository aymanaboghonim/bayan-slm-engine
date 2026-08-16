# Agent Execution Rules & Engineering Standards — bayan-slm-engine

## PART 1: Low-Cognitive-Load HITL Protocol

### 1. Autonomy & Boundary Separation

* **Autonomous Operations (Zero-Approval):** You may autonomously execute chained workspace searches, file reads, directory inspections, and non-destructive verification commands (`uv run ruff check .`, `uv run mypy src/`, `uv run pytest tests/`).
* **Guarded Mutations (Strict HITL):** File writes, code edits, git operations, and destructive shell commands require explicit human confirmation before execution.
* **Workspace-Direct Application:** Apply code modifications directly to target workspace files. Never dump full files or diffs into the chat UI. Chat is strictly for intent, assumptions, and execution gating.
* **Mutation Slice Limit:** Confine edits to **1 logical feature slice** per turn (maximum 1 implementation module + 1 corresponding test file).

### 2. Micro-Reasoning & Communication Limits

* **Standard Word Budget:** Text output MUST NOT exceed **60 words** per turn. Omit conversational filler, chain-of-thought dumps, and generic setups.
* **Failure / Debugging Exception:** If a test, type check, or runtime assertion fails, provide a concise root-cause breakdown (maximum **120 words**) before emitting the audit format.
* **Mandatory Audit Format:** Before requesting approval for any file edit or mutating terminal command, emit this exact 4-line format:
* 🎯 **Intent:** [1 concise sentence on the targeted goal]
* ⚠️ **Key Assumption:** [Max 2 concise bullets on codebase/system state]
* 🛠️ **Proposed Action:** [Exact file path(s) to modify or terminal command to execute]
* 🛑 **Checkpoint:** "Waiting for approval to execute."



### 3. Step-and-Steer Control Loop

* Execute strictly the single approved feature slice or action.
* Pause immediately after execution for human review before proceeding to the next step.

---

## PART 2: Universal AI Engineering Standards

1. **Native Implementations Only:** Monolithic framework wrappers are strictly forbidden (`langchain`, `llama-index`, `autogen`, `TRL`, and high-level `transformers` trainer wrappers). Write native PyTorch (`torch.nn`, `torch.optim`) and custom async state machines.
2. **Pydantic v2 Contracts:** Enforce explicit typing and schema validation using Pydantic v2 `BaseModel` across all data pipelines, configuration files, and API contracts.
3. **Hardware Budget Safety:** Strictly enforce the dynamic VRAM and RAM ceilings defined in `docs/BLUEPRINT.md` (tailored for single RTX 3070 8 GB VRAM). Use explicit tensor shape asserts, CUDA cache management, and CPU-offloaded fixtures for unit tests.
4. **Deterministic Verification:** Enforce schema compliance, tensor dimensions, and sanitization through deterministic code assertions, not prompt heuristics.
5. **Native Observability:** Wrap all inference passes, STT/TTS routing pipelines, and alignment loops in OpenTelemetry spans targeting Arize Phoenix.

---

## PART 3: Source of Truth & Project Context

* **Authoritative Architecture:** `docs/BLUEPRINT.md` is the Single Source of Truth (SSOT) for model choices (`Falcon-H1-0.5B-Base`, decoupled STT/TTS), tensor shapes, hyperparameters, and dynamic memory budgets. Always inspect it before proposing architectural code.
* **Authoritative Sequence:** `docs/EXECUTION_ROADMAP.md` dictates active milestone gating. Do not implement features beyond the current active phase.
* **Runtime Stack:** Python 3.12+ managed via `uv`.

---

## PART 4: Spec-Driven Agent Iteration (SDAI) Workflow

Implement all features following the Contract-First TDD lifecycle:

1. **State 1: Interface Contract**
* Define abstract interfaces, type signatures, and Pydantic v2 schemas.
* Method stubs must raise `NotImplementedError`.


2. **State 2: Contract Test Suite**
* Write `pytest` suites validating interface behavior, tensor shapes, and boundary constraints using dummy CPU tensors.


3. **State 3: Internal Implementation**
* Implement internal module logic only after the failing test suite is committed.


4. **Definition of Done (DoD) Verification:**
* Autonomously run `uv run ruff check .`
* Autonomously run `uv run mypy src/`
* Autonomously run `uv run pytest tests/`
* Verify zero warnings and zero errors before concluding the feature slice.
