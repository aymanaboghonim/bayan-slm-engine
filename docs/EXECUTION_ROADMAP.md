# EXECUTION_ROADMAP.md

This document strictly governs the Spec-Driven Agent Iteration (SDAI) implementation of `bayan-slm-engine`. It translates the `BLUEPRINT.md` architectural specifications into deterministic, verifiable engineering milestones.

**Execution Rule:** Phase 0 is exempt from the "single feature slice" constraint and permits multi-file scaffolding. All subsequent phases (1–6) strictly enforce atomic, single-PR execution and Contract-First (test-driven) development.

**README Lifecycle Rule:** Each phase's README checklist checkbox must be flipped to `- [x]` as part of that phase's Definition of Done, and any README metric tables/telemetry placeholders are populated only when the corresponding phase passes.

---

## Phase 0: Workspace Scaffolding & CI Guardrails

**Objective:** Establish the foundational repository structure, deterministic package management, and the programmatic Definition of Done (DoD) pipeline required for the AI coding agent to self-correct during future phases.

### Milestone 0.1: Project Initialization & Dependency Lock

* **Context:** Set up the Python 3.12+ workspace using `uv` to ensure absolute dependency determinism and fast resolution. Configure static analysis tools to enforce code quality before any ML logic is written.
* **Target Files:** `pyproject.toml`, `.pre-commit-config.yaml`, `.python-version`
* **Actionable Steps:**
1. Initialize the `uv` workspace for a standalone Python package (`src/bayan_slm_engine`).
2. Define core dependencies: `torch`, `torchaudio`, `bitsandbytes`, `pandas`, `numpy`, `tokenizers`, `huggingface-hub`, `fastapi`, `gradio`, `trackio`, `opentelemetry-api`.
3. Define development dependencies: `pytest`, `ruff`, `mypy`, `psutil`, `pre-commit`.
4. Configure `ruff` (line length 100, strict linting) and `mypy` (strict typing) within `pyproject.toml`.
5. Wire `.pre-commit-config.yaml` to run `ruff` and `mypy` on every commit.


* **Definition of Done (DoD):**
* `uv sync` successfully resolves and generates `uv.lock`.
* `uv run pre-commit run --all-files` executes with zero failures.



### Milestone 0.2: Execution Orchestration (Makefile)

* **Context:** Abstract complex, hardware-constrained PyTorch execution commands and WSL2 OS-level memory flushing into a single, reliable interface to prevent human/agent execution error.
* **Target Files:** `Makefile`
* **Actionable Steps:**
1. Define `setup:` to handle directory scaffolding (`mkdir -p data/raw_sdaia data/processed checkpoints logs`) and `pre-commit` installation.
2. Define `data-prep:` wrapping the binary streamer pipeline.
3. Define `train-slm:` embedding the exact 8-bit AdamW and micro-batching CLI flags.
4. Define `serve-ui:` wrapping the FastAPI/Gradio server with `torch.compile` flags.
5. Define `clean-wsl-ram:` implementing the `sudo sysctl -w vm.drop_caches=3` logic and clearing PyTest/Ruff caches.
6. Define `weights:` to pre-fetch the CATT `.pt` (pinned GitHub Release v2), Whisper, and VITS safetensors checkpoints into `checkpoints/weights/` (offline-ready; referenced by M3.x fallbacks). Invoke downloads via `uv run huggingface-hub` so the locked CLI version is used, never a global binary.
7. Define `verify:` encoding the machine-executable DoD gate (`ruff check` → `ruff format --check` → `mypy src/bayan_slm_engine` → `pytest tests/`).


* **Definition of Done (DoD):**
* Running `make -n <target>` (dry-run) for all targets (`setup`, `data-prep`, `train-slm`, `serve-ui`, `verify`, `clean-wsl-ram`, `weights`) yields no syntax or variable resolution errors.



### Milestone 0.3: CI Pipeline Assembly

* **Context:** Enforce the programmatic DoD automatically on GitHub to prevent merging malformed tensor math or memory-leaking data loaders.
* **Target Files:** `.github/workflows/ci.yml`
* **Actionable Steps:**
1. Create a GitHub Actions YAML leveraging `ubuntu-latest`.
2. Configure the action to use `actions/checkout@v7` (mutable major) followed by `astral-sh/setup-uv@v9.0.0` (setup-uv publishes only exact release tags — pin the exact release, not a bare major) with `enable-cache: true` and `cache-dependency-glob` (string input, multi-line `|` form).
3. Trigger on `pull_request` and `push` to `main` only (dedupes double runs per PR push).
4. Inject pipeline steps: `uv sync --frozen --group dev` (deterministic against the committed lock; no index override — the cu129 wheels are installed as locked), split Ruff lint + Ruff format steps, MyPy (two-tier strictness: paths + `tests.*` overrides in `pyproject.toml`), and the `pytest` suite.
5. Explicitly note in the YAML comments that tests are restricted to CPU-only Shape-Driven validation (no CUDA requirements).
6. Make CI hermetic: set `BAYAN_OFFLINE=1` on the pytest step and confirm no checkpoint downloads occur during tests (dummy CPU tensors only).
7. Add pre-commit hooks: `actionlint` (validates workflow grammar locally — catches errors the generic `check-yaml` misses) and a local `pytest` hook (reuses the project venv; pre-commit then covers tests automatically at commit).
8. Add a `ci-smoke` Makefile target: throwaway `.venv_ci_smoke` + `uv sync --frozen --group dev` + `ruff` + `mypy` + hermetic `pytest` — catches CI-only integration/lock errors locally before opening a PR.


* **Definition of Done (DoD):**
* YAML passes strict GitHub Actions syntax validation.

### Milestone 0.4: README Scaffolding (Professional, No-Hype)

* **Context:** The root `README.md` is currently empty. It is the project's primary professional interface and must state only verified, achievable facts — never hype. It is authored during Phase 0 scaffolding and maintained by the README Lifecycle Rule.
* **Target Files:** `README.md`
* **Actionable Steps:**
1. State system identity (`bayan-slm-engine` — ~500M Arabic SLM + decoupled acoustic engine), hardware constraints (RTX 3070 8 GB VRAM; WSL2 12 GB RAM cap), and the VRAM budgets (pretraining $\le 4.2\text{ GB}$, DPO $\le 4.8\text{ GB}$, serving $\le 2.89\text{ GB}$).
2. Describe the multi-modal topology: Tier A (Falcon-H1-0.5B-Base, domain-adaptive continued pretraining + SFT) and Tier B (Whisper STT 244M, CATT diacritizer, VITS `ar-sa` 145 MB).
3. Publish the model manifest table: Falcon-H1 0.5B · Whisper `oddadmix/whisper-small-arabic-dialectal` 244M · VITS `wasmdashai/vits-ar-sa-huba` 145 MB · CATT `abjadai/catt` ~18.9M — with data provenance (SDAIA ~15M–30M tokens + synthetic distillation pairs) and a publish note: OSS assets are reference-only (downloaded, never re-uploaded); our own artifacts publish to HF at end-of-project (deferred, placeholder `aymanaboghonim/bayan-slm-engine-*`).
4. Document the operational Quickstart via the Makefile, in execution order: `make setup` → `make weights` (offline weight sync) → `make data-prep` → `make train-slm` → `make serve-ui`.
5. Add the phased SDAI Execution Roadmap checklist (Phase 0 through Phase 6), all checkboxes initially `- [ ]` (unchecked); flips are governed by the README Lifecycle Rule.
6. Include a compact Demo pointer: the interactive Gradio inspection dashboard (`make serve-ui`; 3-stage pipeline Raw Text → CATT Diacritization → VITS Audio) and a reference to `BLUEPRINT.md` §4 for the full recording protocol — do **not** duplicate the full walkthrough.
7. Encode telemetry targets **only as targets** (TTFT $\le 60\text{ ms}$, TPOT $\le 15\text{ ms/token}$, RTF $< 0.15$, serving VRAM $\le 2.89\text{ GB}$); every metric must carry an explicit "(target)" label until measured in Phases 5–6.

* **Definition of Done (DoD):**
* `README.md` exists at repo root, contains zero hype/meta-commentary language (no "impressive/prove/senior" phrasing), and every number matches the blueprint SSOT (e.g., 2.89 GB, not 2.85).


## Phase 1: Tokenizer Surgery & Zero-Copy Data Engine

**Objective:** Build the foundational linguistic boundaries (clitic-optimized BPE) and the memory-safe data pipeline. This phase completely neutralizes the 12 GB WSL2 host RAM constraint by guaranteeing that data is pre-processed, packed into binary, and streamed with zero-copy overhead.

*Execution Reminder:* From this phase forward, the Contract-First SDAI loop is strictly enforced.

### Milestone 1.1: Arabic Normalizer & Custom 16k BPE Tokenizer

* **Context:** Standard tokenizers fragment Arabic clitics (ال, و, ب) and bloat sequence lengths. We must standardize characters (Alef/Hamza unification) and train a custom 16,000-vocabulary BPE. 16k ensures every token ID mathematically fits inside a `uint16` integer for downstream binary packing.
* **Target Files:**
  * `src/bayan_slm_engine/tokenizer/normalizer.py`
  * `src/bayan_slm_engine/tokenizer/bpe_trainer.py`
  * `src/bayan_slm_engine/tokenizer/verify_vocab.py`
  * `tests/test_tokenizer_shapes.py`
  * `tests/test_tokenizer_metrics.py`

* **Actionable Steps (Contract-First):**
  1. **Step A (Interface):** Stub the `ArabicNormalizer` and `BPETrainer` classes with strict input/output type hints. Define the `TokenizerMetricsReport` Pydantic v2 schema (fertility, `r_char`, `l_avg`, `dead_pct`, optional `morph_f1`) and the `report_metrics(report) -> int` diagnostic contract (BLUEPRINT §2 — Calculate & Report paradigm).
  2. **Step B (Tests):** Write `pytest` assertions that feed highly complex dialectal Arabic strings (mixed diacritics, varying Alefs) into the normalizer and assert exact expected string outputs. Write a round-trip test asserting `decode(encode(text)) == normalized_text`. Write metric unit tests on pure functions over dummy token lists — asserting `r_char` **excludes** the 8-char clitic allowlist $\{$و, ف, ب, ل, ك, س, ح, ع$\}$ (and that `ا` **is** flagged as fallback — deliberate ADR decision, BLUEPRINT §2) — and a report-emission test asserting exit code 0, the `logs/tokenizer_metrics.json` artifact, and the Trackio `tokenizer-diagnostics-*` run (soft warnings, never exceptions).
  3. **Step C (Implement):** Implement the Unicode NFC normalization, regex-based clitic handling, Hamza-position normalization, Ha/Ta-Marbuta (`ه` vs `ة`) resolution, and stripping of spurious diacritics (leaving diacritization strictly to the TTS frontend), plus the Hugging Face `tokenizers` training loop to hit exactly $V = 16,000$. Implement `verify_vocab.py`: seeded 2% validation holdout (`is_validation_line`, deterministic via `blake2b` — immune to `PYTHONHASHSEED`), streaming RAM-safe metric computation, and triple emission (stdout report + `logs/tokenizer_metrics.json` + Trackio run `tokenizer-diagnostics-*`). Metrics are computed on the validation split; dead/rare token utilization on the full corpus. Soft warnings per BLUEPRINT §2 provisional ranges — the report **never gates** the pipeline (always exit 0). Morph F1 is informational and guarded by `--with-morph-alignment`.

* **Definition of Done (DoD):**
  * `pytest tests/test_tokenizer_shapes.py` passes.
  * `pytest tests/test_tokenizer_metrics.py` passes.
  * The generated tokenizer `.json` file explicitly caps at 16,000 tokens.
  * `verify_vocab.py` emits the diagnostic report (`logs/tokenizer_metrics.json` + Trackio `tokenizer-diagnostics-*` run) with exit code 0 — report *presence*, not threshold pass, gates the milestone.

> **BPE Training & Token Coverage Rule:** The 16k BPE tokenizer in Milestone 1.1 must be trained on a representative bootstrap corpus combining raw normalized SDAIA text *and* synthetic JSON dialogue schema templates. In Milestone 1.4, before executing `uint16` binary packing, the packer must validate that all synthetic distillation pairs yield zero Out-Of-Vocabulary (OOV) errors against the frozen 16k vocabulary.

> **Verification Corpus Scope (M1.1):** `make tokenize` trains and verifies on the committed hermetic bootstrap corpus (`tests/fixtures/dialect_corpus.txt`) so CI stays offline (`BAYAN_OFFLINE=1`). Real SDAIA verification lands with M1.2; local ALMoST-style runs (e.g., the SDAIA ALMoST Saudi text split) are supported at any time via the existing `CORPUS` override — `make tokenize CORPUS=/path/to/almost-saudi.txt` — without any CI wiring (Dual-Benchmark Track 2, BLUEPRINT §3).


### Milestone 1.2: SADA22 Metadata Ingestion & Modality Routing

* **Context:** Blindly ingesting the 50GB audio dataset will cause an OOM crash. We must download only the metadata (Parquet) of the **SADA22** corpus (`MohamedRashad/SADA22` — Saudi Audio Dataset, SDAIA NCAI; ~667h; `cc-by-nc-sa-4.0`), filter it demographically (Najdi/Hijazi), and construct the dynamic field router to pass punctuation/diacritics to Tier A (Text SLM) while stripping them for Tier B (STT).
* **Verified SADA22 schema:** columns `audio`, `text`, `cleaned_text`, `speaker_age` (string), `speaker_gender`, `speaker_dialect` (capitalized `Najdi`/`Hijazi`/`Unknown`). Multi-speaker rows carry the literal sentinel `More than 1 speaker اكثر من متحدث` in all demographic columns — retained for Tier A (conversational turn-taking) and purged from Tier B TTS (acoustic cross-talk / MAS collapse), per blueprint §"Domain & Persona Focus".
* **Target Files:**
  * `src/bayan_slm_engine/data/ingestion.py`
  * `tests/test_data_routing.py`

* **Actionable Steps (Contract-First):**
  1. **Step A (Interface):** Define Pydantic v2 schemas for the expected raw Parquet rows and the output structured dataclasses (e.g., `TierATrainingRow`, `TierBTrainingRow`).
  2. **Step B (Tests):** Write assertions proving that when a multi-speaker row is passed, it is preserved for Tier A but actively purged/dropped for Tier B TTS. Assert that the STT router strictly outputs `cleaned_text` (no punctuation).
  3. **Step C (Implement):** Implement the Pandas metadata filtering logic strictly on the SADA22 `.parquet` files. (Synthetic deduplication is deferred to Milestone 1.3).

* **Definition of Done (DoD):**
  * `pytest tests/test_data_routing.py` passes.
  * Pydantic validation raises explicit errors if Tier B is accidentally fed the `text` field instead of `cleaned_text`.


### Milestone 1.3: Frontier API Distillation Pipeline

* **Context:** Convert raw SADA transcripts into multi-turn JSON dialogue pairs using the DeepSeek API (`deepseek-v4-flash`; OpenAI-compatible `https://api.deepseek.com` — Gemini is a drop-in via `base_url`/`model` config, deferred until the DeepSeek key/limits are exercised). The script applies dynamic metadata prompts based on demographic tags (`speaker_gender`, `speaker_age`, `speaker_dialect`) to enforce native Arabic conjugations and structured JSON schema formatting. Thinking mode is ON by default on DeepSeek and is explicitly disabled for this schema-restructure task (4–6x latency/cost saving); JSON Output mode is used with `max_tokens` set to avoid truncation.
* **Target Files:**
  * `src/bayan_slm_engine/data/distillation.py`
  * `src/bayan_slm_engine/data/sada22_lexicon.py` (soft speaker-dispersion validator)
  * `tests/test_distillation_schema.py`
  * `tests/test_sada22_lexicon.py`

* **Actionable Steps (Contract-First):**
  1. **Step A (Interface):** Define Pydantic v2 schemas for the input demographic row (`DistillationInputRow`), the resulting output JSON schema (`MultiTurnDialogue` — keys `user`/`assistant`/`dialect`), the `DistillationConfig` settings, and the `DistillationReport` outcome schema.
  2. **Step B (Tests):** Write unit tests with mocked API responses (`httpx.MockTransport` — hermetic, `BAYAN_OFFLINE=1` safe) to assert that generated dialogues strictly validate against the Pydantic schema, contain valid JSON keys (`user`, `assistant`, `dialect`), pass regex checks (no non-JSON wrapper text), and that the request body explicitly disables thinking and requests JSON output.
  3. **Step C (Implement):** Implement the long-lived `httpx` client (single socket pool), bounded async concurrency (`asyncio.Semaphore`, default 8, order-preserving via indexed `gather`), exponential backoff + jitter with `Retry-After` handling on 429, fatal `DeepSeekPaymentError` on HTTP 402, per-sample failure isolation (skip + `DistillationReport` counts; `--fail-fast` opt-in), append-mode JSONL streaming with a sidecar `.anchors` resume registry (zero-loss resume at sample N+1), and the regex + MinHash LSH (Jaccard = 0.75, 5-grams) deduplication on the output — via an inverted hash-bucket index (exact Jaccard remains the arbiter).
  4. **Step D (Lexical drift hardening):** restrict the drift denylist to unambiguous Egyptian/Levantine particles with word-boundary anchors (native Saudi `عشان`/`مين`/`ليش` pass); canonicalize `كده`/`كدا` → `كذا` in the M1.1 normalizer (clitic-preserving); add the soft local-only `Sada22LexiconValidator` (distinct-`audio` speaker proxy, `min_speakers=5`) that flags low-dispersion tokens for regeneration without gating CI.

* **Definition of Done (DoD):**
  * `pytest tests/test_distillation_schema.py tests/test_sada22_lexicon.py` passes.
  * The generated JSONL output successfully parses via `json.loads` line-by-line without throwing decoding errors.
  * Hermetic resilience tests pass: mocked HTTP 402 halts the batch fatally; 429 without `Retry-After` falls back to backoff+jitter (never `sleep(0)`); a bad sample is skipped while others succeed with input order preserved; the sidecar registry resumes at sample N+1.


### Milestone 1.4: 16-Bit Zero-Copy Binary Packer & Streamer

* **Context:** Loading tokens through standard Hugging Face `.map()` and dataset object wrappers multiplies host RAM past the 12 GB WSL2 ceiling due to internal caching and column duplication. We must pack both the tokenized SDAIA text and the synthetic JSON pairs into a flat `uint16` binary file and stream it directly to VRAM using memory-mapped arrays (`np.memmap`), keeping the host RAM footprint strictly $< 1\text{ GB}$.
* **Target Files:**
  * `src/bayan_slm_engine/data/memmap_streamer.py`
  * `tests/test_memmap_memory.py`

* **Actionable Steps (Contract-First):**
  1. **Step A (Interface):** Stub the `BinaryPacker` and `MemmapDataLoader` (which must yield PyTorch tensors of shape `[batch_size, seq_len]`).
  2. **Step B (Tests):** Create a dummy script that writes 100 million random integers to a `.bin` file. Write a test that iterates through this file using the `MemmapDataLoader`.
  3. **Step C (Implement):** Implement the `np.memmap(mode='r')` read logic. Ensure the dataloader strictly enforces `num_workers=1` and returns pre-sliced `torch.from_numpy().long()` batches.

* **Definition of Done (DoD):**
  * `pytest tests/test_memmap_memory.py` passes.
  * **Hardware Footprint Assertion:** The test suite monitors `psutil` memory during the dummy 100M token read and explicitly asserts that host RAM consumption delta remains $< 50\text{ MB}$.


## Phase 2: Tier A Core (500M Text SLM Architecture)

**Objective:** Construct the native PyTorch foundation for the 500M text SLM. This phase implements the `Falcon-H1-0.5B-Base` hybrid architecture locally to bypass heavy Hugging Face `transformers` wrappers, enabling exact control over SDPA memory kernels, Mamba-2 SSM blocks, gradient checkpointing, and custom 16k vocabulary injection without blowing up the 8 GB RTX 3070 VRAM.

*Execution Reminder:* Strict Contract-First SDAI loop required. Write failing shape-tests before any tensor math.

### Milestone 2.1: Abstract Interfaces & Shape-Driven Test Suite
*   **Context:** Before implementing complex hybrid Transformer/Mamba-2 blocks or Grouped-Query Attention, the exact mathematical contracts (tensor dimensionalities) must be locked in. This prevents deep graph errors during later training.
*   **Target Files:**
    *   `src/bayan_slm_engine/models/text_slm.py`
    *   `tests/test_text_slm_shapes.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub the `BayanTextSLM`, `AttentionBlock` (GQA), `Mamba2Block` (SSM), `MLPBlock` (SwiGLU), and `RoPE` classes using `torch.nn.Module`. Add strict type hints and docstrings detailing input shapes `(B, S, d_model)` and output shapes.
    2.  **Step B (Tests):** Write `pytest` assertions injecting dummy CPU tensors (e.g., `batch_size=2`, `seq_len=1024`, `d_model=1024`) into each stubbed module. Assert that the output tensor shapes for both GQA and Mamba-2 layers exactly match expectations and that no silent dimension broadcasting occurs.
    3.  **Step C (Implement):** Halt. Wait for test generation before executing implementation.
*   **Definition of Done (DoD):**
    *   Module signatures and Pydantic configuration schemas are defined.
    *   `pytest tests/test_text_slm_shapes.py` fails explicitly due to `NotImplementedError` (validating the test executes).

### Milestone 2.2: Native PyTorch Implementation & Checkpoint Translation
*   **Context:** Implement the internal math of the Falcon-H1 hybrid backbone natively. This requires explicitly wiring PyTorch 2.x `F.scaled_dot_product_attention` (SDPA) for FlashAttention-2 dispatch alongside native Mamba-2 state-space recurrence logic. We must also build a weight-loading utility to map the open-source TII `.safetensors` into our native classes.
*   **Target Files:**
    *   `src/bayan_slm_engine/models/text_slm.py` (Implementation)
    *   `src/bayan_slm_engine/models/checkpoint_loader.py`
    *   `tests/test_model_parity.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub `load_falcon_weights(native_model: nn.Module, safetensors_path: str)`.
    2.  **Step B (Tests):** Write a test that instantiates `BayanTextSLM`, asserts the total trainable parameter count is $\approx 500M$, and verifies that `load_falcon_weights` successfully mounts the state dictionary (including Mamba-2 SSM weights and GQA weights) without missing/unexpected key errors.
    3.  **Step C (Implement):** Implement GQA (16 Q-Heads, 4 KV-Heads), Mamba-2 SSM blocks, RoPE, and SwiGLU logic natively. Implement the state-dict mapping logic to translate Hugging Face key names to our native `bayan_slm_engine` module names.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_text_slm_shapes.py` passes (tensor math is correct for both Attention and Mamba blocks).
    *   `pytest tests/test_model_parity.py` passes (parameter count matches specification and checkpoint loads cleanly).

### Milestone 2.3: Embedding Surgery & 16k Vocab Injection
*   **Context:** The loaded Falcon-H1 weights contain a generic, massive multilingual vocabulary embedding matrix (~65k+). To specialize for Arabic and reduce sequence/parameter bloat, we must perform low-level tensor surgery to resize the embedding matrix and final LM head strictly to our custom $V = 16,000$ BPE size, preserving overlapping token weights where possible.
*   **Target Files:**
    *   `src/bayan_slm_engine/models/embedding_surgery.py`
    *   `tests/test_embedding_surgery.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub `resize_and_initialize_embeddings(model: nn.Module, old_tokenizer, new_tokenizer, new_vocab_size: int = 16000)`.
    2.  **Step B (Tests):** Write a test that loads the native model, runs the surgery function, and explicitly asserts that `model.embedding.weight.shape == (16000, 1024)` and `model.lm_head.weight.shape == (16000, 1024)`. Assert that intersecting tokens retain identical embedding values.
    3.  **Step C (Implement):** Implement the matrix resizing. For every token in the new 16k Arabic vocabulary that exists in Falcon-H1's original vocabulary, directly copy its weight vector; apply mean/variance warm-start initialization to remaining new clitic tokens to prevent catastrophic loss spikes during continued pretraining.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_embedding_surgery.py` passes.
    *   A forward pass with token IDs in the range `[0, 15999]` succeeds, while passing a token ID of `16000` throws an explicit `IndexError`.


## Phase 3: Tier B Micro-Speech Subsystems (Decoupled Audio)

**Objective:** Integrate the lightweight open-weight acoustic checkpoints and build the critical Arabic morphological frontend. This phase ensures that speech recognition strictly routes native undiacritized text to the SLM, and speech synthesis explicitly restores diacritics and phoneme durations before generating audio, all while keeping the combined acoustic VRAM footprint strictly under ~690 MB (500 MB STT + 150 MB VITS + ~38 MB CATT).

*Execution Reminder:* Strict Contract-First SDAI loop required. Validate tensor shapes and memory footprints before implementing model loading.

### Milestone 3.0: Checkpoint Parity Spikes (Gating M3.1 & M3.2 DoD)
*   **Context:** Weight translation (HF safetensors → native `torchaudio` Whisper; CATT `.pt` → native `torch.nn`) is the riskiest Tier B integration. Run these spikes first so M3.1/M3.2 DoDs are committed only after conversion is proven.
*   **Target Files:**
    *   `tests/test_stt_parity.py`
    *   `tests/test_catt_translation.py`
    *   `tests/fixtures/` (committed minimal reference tensors — e.g., a 2-layer Whisper decoder slice and a CATT head stub — plus CATT `.pt` metadata; never downloaded during CI)
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub `whisper_safetensors_to_torchaudio()` and `catt_pt_to_native()` conversion utilities with explicit state-dict remap tables.
    2.  **Step B (Tests):** Write parity tests asserting `|logits_native − logits_reference| < 1e-3` on a dummy sample using the **committed reference fixtures** (`tests/fixtures/`) as ground truth, and that the CATT `.pt` loads with zero unexpected/missing key errors.
    3.  **Step C (Implement):** Implement both converters; if logit parity fails, record an ADR and adopt the documented alternate loading path before M3.1 DoD is met.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_stt_parity.py tests/test_catt_translation.py` passes (hermetic, dummy CPU tensors, `BAYAN_OFFLINE=1`).

### Milestone 3.1: 244M STT Encoder (`whisper-small-arabic-dialectal`)
*   **Context:** The STT engine must map 16kHz audio to undiacritized Arabic text representing colloquial Saudi speech. We initialize from the 244M parameter `oddadmix/whisper-small-arabic-dialectal` checkpoint to natively handle Gulf dialects without fine-tuning, operating safely within a ~500 MB VRAM budget.
*   **Target Files:**
    *   `src/bayan_slm_engine/models/stt_encoder.py`
    *   `tests/test_stt_shapes.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub the `ArabicSTTEncoder` class with strict type hints for 16kHz audio array inputs and string outputs.
    2.  **Step B (Tests):** Write `pytest` assertions injecting a dummy CPU log-mel spectrogram tensor (e.g., `batch=1, features=80, frames=3000`). Assert that the model outputs a valid string type. Write a test asserting that the VRAM/RAM allocation delta remains $\le 550\text{ MB}$ upon instantiation. **SPIKE GATE (before M3.1 DoD is considered met):** run a spike verifying `whisper_safetensors_to_torchaudio` achieves logit parity on a dummy sample (`|logits_native − logits_reference| < 1e-3`); if parity fails, record an ADR and adopt the documented alternate loading path before committing the DoD.
    3.  **Step C (Implement):** Implement the Whisper architecture purely natively using `torchaudio.models.whisper` (strictly NO Hugging Face `transformers` dependencies). **Whisper Weight-Translation Rule:** implement an explicit conversion utility (`whisper_safetensors_to_torchaudio`) to remap state dictionary keys and adjust tensor layouts, accompanied by a parity test (e.g., `tests/test_stt_parity.py`) verifying the converted model loads cleanly without unexpected or missing key errors. **Note:** The model natively generates undiacritized output; implement a standard greedy decode loop without hallucinating unnecessary suppression parameters.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_stt_shapes.py` passes.
    *   Hardware footprint assertion strictly passes (STT checkpoint consumes $< 550\text{ MB}$ VRAM).

### Milestone 3.2: Arabic TTS Frontend (CATT Diacritizer & Duration G2P)
*   **Context:** Undiacritized Arabic cannot be blindly routed to a TTS engine. We must implement the ~18.9M parameter Character-based Arabic Tashkeel Transformer (CATT) to restore short vowels, and an explicit Grapheme-to-Phoneme (G2P) engine that maps gemination (Shaddah) to temporal durations and handles emphatic consonants (ص ض ط ظ).
*   **Target Files:**
    *   `src/bayan_slm_engine/models/tts_frontend/diacritizer.py`
    *   `src/bayan_slm_engine/models/tts_frontend/g2p_aligner.py`
    *   `tests/test_tts_frontend.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub the `CATTFrontend` (Transformer/CBHG sequence tagger) and `ArabicG2P` classes. Define Pydantic/Dataclass schemas for the output `PhonemeDurationMap`.
    2.  **Step B (Tests):** Write explicit assertions for the G2P rules: assert that an input string containing a Shaddah (`ّ`) results in a duration multiplier (e.g., `duration * 2.0`), and that emphatics map to distinct phoneme IDs compared to non-emphatics (e.g., `ص` vs `س`).
    3.  **Step C (Implement):** **CATT Initialization Directive:** adopt the canonical GitHub release of **`abjadai/catt`** (Abjad Ltd; paper arXiv:2407.03236; Apache-2.0) — Encoder-Only variant, ~18.9M params, 6 layers, $d_{\text{model}}=512$, character-level 18-class tashkeel head. Explicitly load the raw `.pt` weights (GitHub Release v2; fallback mirror `niobures/CATT` on Hugging Face) to achieve the required DER $\le 4.5\%$. Implement a state-dict translation utility remapping the checkpoint keys into the native `torch.nn` classes, with a parity test verifying clean load (no unexpected/missing keys). **Fallback:** if weights are unavailable during CI, use a deterministic seeded initialization matching the exact shape contract, pending local sync via `make weights`. Code the deterministic G2P alignment rules to translate diacritized Arabic into the phoneme ID and duration tensors expected by the VITS renderer.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_tts_frontend.py` passes.
    *   Strings missing diacritics are correctly assigned Fatha/Damma/Kasra/Shaddah tags natively without throwing dimension errors.
    *   Hardware footprint assertion strictly passes (CATT checkpoint consumes $\le 50\text{ MB}$ VRAM in bf16).

### Milestone 3.3: VITS Acoustic Renderer (`vits-ar-sa-huba`)
*   **Context:** Render the diacritized phonemes and duration mappings into a 16kHz audio waveform. We initialize from the 145 MB `wasmdashai/vits-ar-sa-huba` VITS checkpoint (explicitly fine-tuned for Saudi Arabic, `ar-sa`).
*   **Target Files:**
    *   `src/bayan_slm_engine/models/tts_frontend/vits_renderer.py`
    *   `tests/test_vits_renderer.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub the `SaudiVITSRenderer` class, defining an input signature that accepts the `PhonemeDurationMap` from the G2P engine and outputs a 1D `torch.Tensor` waveform.
    2.  **Step B (Tests):** Write assertions injecting dummy phoneme IDs and duration integers. Assert the output is a valid waveform tensor `(1, seq_len)` and check that the static weight VRAM allocation remains strictly $\le 150\text{ MB}$.
    3.  **Step C (Implement):** Implement the native PyTorch VITS forward-pass logic, loading the 145 MB `.safetensors` checkpoint. Ensure it bypasses internal text normalization (relying exclusively on our upstream CATT/G2P inputs to prevent code-switching or phoneme collapse).
*   **Definition of Done (DoD):**
    *   `pytest tests/test_vits_renderer.py` passes.
    *   Hardware footprint assertion strictly passes (TTS checkpoint consumes $\le 150\text{ MB}$ VRAM).


## Phase 4: Training Engine Mechanics & DPO Alignment

**Objective:** Construct the VRAM-constrained training engines natively in PyTorch. This phase builds the continued-pretraining/SFT loop and the Direct Preference Optimization (DPO) alignment pipeline. By manually wiring 8-bit optimizers, BFloat16, gradient checkpointing, and sequence-length bucketing, we mathematically guarantee that training a 500M model and dual-model alignment both fit within the 8 GB RTX 3070 limit.

*Execution Reminder:* Strict Contract-First SDAI loop required. Absolutely NO Hugging Face `Trainer` or `TRL` wrappers are permitted.

### Milestone 4.1: VRAM-Constrained Continued-Pretraining & SFT Loop
*   **Context:** Standard AdamW consumes ~6GB VRAM for a 500M model's states alone. We must implement a custom loop using `bitsandbytes.optim.AdamW8bit`, `torch.bfloat16` (Ampere native), gradient accumulation (16 steps of micro-batch size 2), and `torch.utils.checkpoint` to cap peak VRAM at $\le 4.2\text{ GB}$.
*   **Target Files:**
    *   `src/bayan_slm_engine/engine/trainer.py`
    *   `tests/test_trainer_memory.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub the `BayanTrainer` class with methods for `training_step()`, `backward_pass()`, and `optimizer_step()`.
    2.  **Step B (Tests):** Write a strict memory-profiling test using a dummy 500M model. Execute a mocked forward/backward pass with `batch_size=2, seq_len=1024` and explicitly assert via `torch.cuda.memory_allocated()` that peak VRAM does not exceed $4.2\text{ GB}$.
    3.  **Step C (Implement):** Implement the native PyTorch training loop. Wire in the `bitsandbytes` 8-bit AdamW optimizer, apply weight decay strictly to 2D matrices (excluding RoPE/RMSNorm), enforce gradient checkpointing across the Transformer/Mamba blocks, apply `clip_grad_norm_(…, 1.0)`, and wire the Cosine Warmup LR schedule (state tracked by M4.2). **Continued-Pretraining vs. SFT Data Routing:** within the training step, route packed raw SDAIA text tokens through standard causal LM cross-entropy loss (broad domain adaptation) and route synthetic multi-turn JSON dialogue pairs through a masked-SFT loss (applying `-100` target masking to user prompt tokens so gradient updates calculate exclusively on assistant response tokens).
*   **Definition of Done (DoD):**
    *   `pytest tests/test_trainer_memory.py` passes.
    *   Hardware footprint assertion strictly passes ($\le 4.2\text{ GB}$ VRAM during simulated backward pass).

### Milestone 4.2: Fault-Tolerant Atomic Checkpointing & Trackio Logging
*   **Context:** WSL2 edge environments are volatile. We must implement atomic state checkpointing to prevent corruption and seamlessly resume. We must also integrate `trackio` for local-first SQLite logging of loss curves and GPU telemetry to prove convergence mathematically without cloud bloat.
*   **Target Files:**
    *   `src/bayan_slm_engine/engine/checkpointing.py`
    *   `tests/test_checkpoint_resume.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub `save_atomic_checkpoint()` and `load_checkpoint_state()`. Stub the `trackio` telemetry dispatcher.
    2.  **Step B (Tests):** Write a test that simulates a training state (model weights, 8-bit optimizer momentum, Cosine scheduler state, CPU/GPU RNG states, and a `.memmap` binary dataloader offset), saves it, mutates the active states, loads the checkpoint, and explicitly asserts `current_state == saved_state` for all components.
    3.  **Step C (Implement):** Implement the atomic save logic (write to `.tmp`, then `os.rename`). Hook in `trackio.log()` to capture `train/loss`, `train/lr`, gradient norms, and `torch.cuda.memory_allocated()` during the training loop.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_checkpoint_resume.py` passes.
    *   The checkpointing mechanism strictly maintains a rolling window of the last $K=3$ checkpoints to prevent SSD bloat.

### Milestone 4.3: Sequence-Length Bucketed DPO Alignment Pipeline
*   **Context:** DPO requires juggling an active policy network ($\pi_\theta$) and a frozen reference network ($\pi_{\text{ref}}$) simultaneously. Variable-length padding causes PyTorch memory fragmentation, triggering OOM crashes. We must implement **Sequence-Length Bucketing** in the data sampler and write the DPO loss math natively to cap peak VRAM at $\le 4.8\text{ GB}$.
*   **Target Files:**
    *   `src/bayan_slm_engine/engine/dpo_trainer.py`
    *   `src/bayan_slm_engine/data/dpo_sampler.py`
    *   `tests/test_dpo_mechanics.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub `BucketedDPOSampler` and the `dpo_loss_fn(policy_logprobs, ref_logprobs, beta: float = 0.1)` mathematical function.
    2.  **Step B (Tests):** Write a sampler test asserting that every yielded batch contains preference triples $(x, y_w, y_l)$ of the exact same sequence length (zero padding). Write a DPO memory test executing a forward pass of both models, asserting total VRAM stays $\le 4.8\text{ GB}$.
    3.  **Step C (Implement):** Implement the sequence bucketing logic. Implement the native DPO loss formula. Ensure the reference model is loaded strictly under `torch.no_grad()` in `bfloat16` to lock its footprint at $1.0\text{ GB}$, while the policy model updates via the 8-bit optimizer. **DPO Alignment Targets & Preference Pairing:** the preference data loader and loss computation must ingest paired triples $(x, y_w, y_l)$ mapped to the blueprint's dual alignment targets — (1) **Dialect Purity:** $y_w$ sourced from grounded, authentic SDAIA transcripts exhibiting native Najdi/Hijazi markers, $y_l$ capturing API-hallucinated cross-dialect drift (Egyptian/Levantine particles); (2) **Schema Rigidity:** $y_w$ enforcing valid, parseable JSON dictionaries, $y_l$ targeting malformed, non-JSON text or broken syntax.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_dpo_mechanics.py` passes.
    *   Hardware footprint assertion strictly passes ($\le 4.8\text{ GB}$ VRAM during dual-model forward/backward pass).


## Phase 5: Evaluation & Custom Benchmarking Suite

**Objective:** Implement the Dual-Benchmark Strategy (BLUEPRINT §3) — **Track 1** internal component engineering suite (deterministic, hermetic metrics: dialect purity, schema rigidity, acoustic accuracy, hardware constraints) and **Track 2** external SDAIA ALMoST subset (sliced Saudi dialectal text, local-only). Standard open-domain LLM metrics (like Perplexity or MMLU) fail to capture the operational targets of this system. On passing M5.1/M5.2 (Track 1) and M5.3 (Track 2), populate the README evaluation table with measured DMPR / JSON compliance / DER / WER / RTF / ALMoST-subset values, replacing the target placeholders.

*Execution Reminder:* Strict Contract-First SDAI loop required. Build the mathematical metric evaluators as pure functions before integrating them into a benchmarking script.

### Milestone 5.1: Dialect Purity & Schema Validation Engine (Track 1 — Internal Suite)
*   **Context:** We must evaluate the SLM's post-DPO performance using two strict operational metrics: Dialect Marker Preference Ratio (DMPR) to track regional particle accuracy (e.g., suppressing Levantine `بدّي`), and Zero-Shot JSON Schema Compliance (testing raw `json.loads()` success without constrained-decoding crutches).
*   **Target Files:**
    *   `src/bayan_slm_engine/eval/text_metrics.py`
    *   `tests/test_eval_text.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub the functions `calculate_dmpr(responses: list[str], target_dialect: str) -> float` and `calculate_json_compliance(responses: list[str]) -> float`.
    2.  **Step B (Tests):** Write unit tests injecting known arrays. (e.g., For DMPR, inject a list containing Egyptian `عايز` and Najdi `أبى`, asserting the formula strictly penalizes the Egyptian token. For JSON compliance, inject one valid JSON string and one with a trailing comma, asserting exactly 50.0% compliance).
    3.  **Step C (Implement):** Implement the metric functions. DMPR uses a predefined dictionary of dialect-exclusive particles (mapped from the SDAIA metadata). JSON compliance uses a strict `try/except json.loads` block.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_eval_text.py` passes.
    *   The benchmarking script successfully evaluates 500 generated prompt responses, targeting DMPR $\ge 88.0\%$ and JSON Compliance $\ge 94.5\%$.

### Milestone 5.2: Acoustic Frontend Verification (DER, WER, & RTF) (Track 1 — Internal Suite)
*   **Context:** The TTS bottleneck is missing diacritics. We must quantify the CATT neural diacritizer's accuracy using Diacritization Error Rate (DER). Furthermore, we need to track the STT Word Error Rate (WER) and the VITS Real-Time Factor (RTF) to prove the pipeline runs smoothly on the RTX 3070.
*   **Target Files:**
    *   `src/bayan_slm_engine/eval/audio_metrics.py`
    *   `tests/test_eval_audio.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub `calculate_der(preds, targets)`, `calculate_wer(preds, targets)`, `calculate_rtf(synthesis_time, audio_duration)`, and `calculate_diacritization_latency()` (timing decorator over the CATT frontend; DoD target $\le 35\text{ ms/sentence}$).
    2.  **Step B (Tests):** Write deterministic assertions. For DER/WER, inject strings with known character/word mismatches and assert the output matches standard Levenshtein distance calculations. **RTF Inequality Test Assertion Rule:** dummy inputs must yield a value strictly below the blueprint threshold — set synthesis time to `1.49s` over `10.0s` of audio to yield `0.149`, ensuring the strict inequality `RTF < 0.15` passes.
    3.  **Step C (Implement):** Implement the Levenshtein-based error rate logic natively (or via lightweight `editdistance`). Hook the RTF calculation into a timing decorator that can be wrapped around the VITS forward pass.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_eval_audio.py` passes.
    *   The acoustic benchmarking suite yields metrics proving the architecture hits blueprint targets: DER $\le 4.5\%$, STT WER $\le 18.5\%$, and TTS RTF $< 0.15$.

### Milestone 5.3: Track 2 — SDAIA ALMoST Saudi-Subset Harness
*   **Context:** Provide external, industry-recognized credibility on authentic Saudi dialectal text by benchmarking the post-alignment model against a **sliced subset** of the SDAIA ALMoST evaluation suite (BLUEPRINT §3 Dual-Benchmark Strategy). The slice keeps Saudi regional dialect comprehension (Najdi/Hijazi), cultural/geographical QA, and task-oriented intent routing; it excludes 70B-scale multi-step reasoning, multi-hop formal logic, and open-ended essay generation (architectural-mismatch ADR). Dataset download is **local-only** — never in hermetic CI (`BAYAN_OFFLINE=1`).
*   **Target Files:**
    *   `src/bayan_slm_engine/eval/almost_harness.py`
    *   `tests/test_almost_harness.py`
    *   `Makefile` — `eval-almost` target (local-only; documents ALMoST access/license gate)
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub the `ALMoSTSlice` dataclass (subset paths, task labels, license gate flag) and `evaluate_almost_slice(model, slice) -> dict[str, float]`.
    2.  **Step B (Tests):** Write unit tests over a committed minimal ALMoST-style fixture (dummy Saudi-slice items) asserting the slicing filter excludes reasoning/essay categories, and that the license gate raises when the ALMoST dataset is absent or unlicensed (hermetic — no real download).
    3.  **Step C (Implement):** Implement the slicing filter (keep: dialect comprehension, cultural/geo QA, intent routing; drop: 70B-scale math/logic/essays), the local `make eval-almost` harness, and the per-task accuracy aggregation. **Feasibility gate:** verify ALMoST access/license terms and dialect-split integrity before wiring real data; record an ADR if the dataset is inaccessible.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_almost_harness.py` passes (hermetic fixtures only).
    *   `make eval-almost` runs locally against the sliced ALMoST Saudi subset and reports per-task accuracy + aggregate (target ≥ sliced baseline + 5.0 pts; baseline measured at first run).


## Phase 6: High-Throughput Serving & Observability

**Objective:** Deploy the production serving tier and instrumentation layer. This phase connects the custom PyTorch models to standard web protocols via an asynchronous FastAPI server, implements pre-allocated static KV-caching and `torch.compile`, wraps execution in OpenTelemetry spans exported to local Arize Phoenix, and mounts an interactive Gradio engineering dashboard—all while capping peak serving VRAM strictly at $\le 2.89\text{ GB}$. After M6.x passes, update the README telemetry and demo sections with measured TTFT / TPOT / RTF / serving-VRAM values, flip the Phase 6 checkbox, and mark the project status as operational.

*Execution Reminder:* Strict Contract-First SDAI loop required. Write failing contract and shape-tests before implementing server routes or inference loops.

### Milestone 6.1: Autoregressive Generation & Static KV-Cache
*   **Context:** Dynamic tensor concatenation during generation fragments the RTX 3070's VRAM and creates latency spikes. We must pre-allocate a static `bfloat16` KV-cache tensor up to max sequence length (2048), perform in-place decode slice updates, and compile the forward pass via `torch.compile(mode="reduce-overhead")` to hit target latencies (TTFT $\le 60\text{ ms}$, TPOT $\le 15\text{ ms/token}$).
*   **Target Files:**
    *   `src/bayan_slm_engine/engine/inference.py`
    *   `tests/test_inference_kv.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub the `BayanInferenceEngine` class with methods for `prefill_cache()`, `decode_step()`, and `generate_stream()`.
    2.  **Step B (Tests):** Write a test injecting a prompt token sequence, verifying that the pre-allocated static KV-cache tensor dimensions remain invariant throughout generation, and asserting that inference memory delta stays within the $\le 2.89\text{ GB}$ ceiling.
    3.  **Step C (Implement):** Implement the static KV-cache pre-allocation logic, wire in `torch.compile` on the model forward pass, and write the autoregressive generation loop yielding tokens asynchronously.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_inference_kv.py` passes.
    *   Inference benchmarking script validates TTFT $\le 60\text{ ms}$ and TPOT $\le 15.0\text{ ms/token}$ using dummy inputs.

### Milestone 6.2: FastAPI Async Server & OpenTelemetry Instrumentation
*   **Context:** Expose the text and audio subsystems via standard HTTP/SSE endpoints. We must instrument raw PyTorch forward passes and multi-modal handoffs using vendor-neutral OpenTelemetry (OTel) standards, exporting trace hierarchies to a local Arize Phoenix collector.
*   **Target Files:**
    *   `src/bayan_slm_engine/serving/app.py`
    *   `src/bayan_slm_engine/serving/telemetry.py`
    *   `tests/test_serving_api.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub FastAPI route handlers (`/v1/chat`, `/v1/stt`, `/v1/tts`) using Pydantic v2 request/response schemas. Stub the full OTel span set: `bayan.normalize_and_tokenize`, `bayan.slm_prefill_and_decode`, `bayan.kv_cache_alloc`, `bayan.autoregressive_step`, `bayan.arabic_tts_pipeline`, `bayan.catt_diacritize`, `bayan.g2p_duration_map`, and `bayan.vits_render`.
    2.  **Step B (Tests):** Write `pytest` test client assertions simulating asynchronous requests to the endpoints. Assert that responses conform to expected Pydantic schemas and that mock OTel spans correctly record execution durations.
    3.  **Step C (Implement):** Implement the FastAPI app with Server-Sent Events (SSE) streaming for token outputs. Wire in the OTel instrumentation to trace custom PyTorch execution spans and export to Arize Phoenix.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_serving_api.py` passes.
    *   API endpoints successfully stream responses and export structured trace spans to local Phoenix without dropping connections.

### Milestone 6.3: Interactive Gradio Engineering Dashboard
*   **Context:** Provide a local web interface that functions as an Engineering Inspection Tool rather than a generic chat box. The UI must expose multi-modal inputs, stream SLM outputs, display the intermediate CATT diacritization text (Stage 2) to prove morphological rigor, render VITS audio, and monitor live hardware telemetry. To streamline developer iteration and smoke-testing without manual typing or peripheral audio capture, the dashboard must include pre-loaded click-to-test examples for both text and audio pipelines. The dashboard lives inside `app.py` (sequenced after M6.2's route/OTel work on the same file) and is mounted via `gradio.mount_gradio_app()` so the headless API routes and the UI share one process and port.
*   **Target Files:**
    *   `src/bayan_slm_engine/serving/app.py` (Gradio layout mounted alongside the headless API routes; `make serve-ui` launches `app.py`)
    *   `tests/test_dashboard_ui.py`
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub the Gradio layout generator function `create_bayan_dashboard()` and mount helper functions.
    2.  **Step B (Tests):** Write a test verifying that the Gradio blocks instantiate successfully, components bind to the correct backend FastAPI routes, and metric cards read hardware telemetry without throwing binding errors.
    3.  **Step C (Implement):** Build the Gradio interface featuring Panel A (controls, input modes, dialect/JSON toggles, and pre-loaded `gr.Examples` for instant text/audio evaluation) and Panel B (streaming text window, CATT diacritization display box, VITS audio player, live VRAM/telemetry cards). Mount the dashboard directly onto the FastAPI app via `gradio.mount_gradio_app()`.
*   **Definition of Done (DoD):**
    *   `pytest tests/test_dashboard_ui.py` passes.
    *   Running `make serve-ui` successfully launches the unified FastAPI + Gradio inspection interface on edge hardware while maintaining peak VRAM $\le 2.89\text{ GB}$.

### Milestone 6.4: HF Artifact Publishing (Deferred Gate)
*   **Context:** Per the blueprint's Artifact & Publishing Strategy, OSS checkpoints are reference-only (never re-uploaded), while our own artifacts (synthetic JSON corpus, 16k Arabic BPE tokenizer, domain-adapted Falcon-H1 SFT/DPO checkpoint) are published to Hugging Face at the end of the project. This milestone is intentionally deferred; it runs only after the full pipeline is verified reproducible.
*   **Target Files:**
    *   `scripts/publish_hf.py` (upload-only; no upload code in earlier phases)
    *   `HF_TOKEN` environment variable (required at publish time only)
*   **Actionable Steps (Contract-First):**
    1.  **Step A (Interface):** Stub `publish_artifacts(repo_ids: dict[str, str])` that pushes the synthetic dataset, tokenizer, and checkpoint via `huggingface_hub.upload_folder`.
    2.  **Step B (Tests):** Mock `upload_folder` and assert correct repo IDs under the placeholder namespace `aymanaboghonim/bayan-slm-engine-*`; assert the script fails fast (no silent skip) when `HF_TOKEN` is missing.
    3.  **Step C (Implement):** Implement upload logic. Final HF repo names are decided at publish time; gate checks: all Phases 0–6 checkboxes complete, `make setup → weights → data-prep → train-slm` reproducible, license audit passed (SDAIA terms; Apache-2.0).
*   **Definition of Done (DoD):**
    *   `pytest tests/test_publish_hf.py` passes (hermetic, mocked uploads).
    *   Manual gate: full `make` chain reproducible + all roadmap checkboxes `- [x]` before any real push executes.
