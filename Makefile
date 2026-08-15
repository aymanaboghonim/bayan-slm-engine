# Makefile for bayan-slm-engine (WSL2 / RTX 3070 constraints applied)
# SSOT: mirrors the makefile block in docs/BLUEPRINT.md

.PHONY: setup weights tokenize data-prep train-slm serve-ui verify ci-smoke clean-wsl-ram

# 1. Environment & Pre-commit Initialization
setup:
	mkdir -p data/raw_sdaia data/processed checkpoints logs
	uv sync
	uv run pre-commit install

# 2. Zero-Copy Binary Data Packing
data-prep:
	uv run python src/bayan_slm_engine/data/memmap_streamer.py \
		--input-dir data/raw_sdaia/ \
		--output-bin data/processed/arabic_corpus.uint16.bin

# 3. 500M SLM Pretraining (Enforcing 8GB VRAM limits)
train-slm:
	CUDA_VISIBLE_DEVICES=0 uv run python src/bayan_slm_engine/engine/trainer.py \
		--micro-batch-size 2 \
		--grad-accum-steps 16 \
		--use-8bit-adam \
		--gradient-checkpointing

# 4. Asynchronous Multi-Modal Serving & Telemetry
serve-ui:
	CUDA_VISIBLE_DEVICES=0 uv run python src/bayan_slm_engine/serving/app.py \
		--enable-torch-compile \
		--kv-cache-max-seq 2048

# 5. Machine-Executable Definition of Done (DoD) gate
verify:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest tests/

# 6. CI smoke test (catches CI-only integration errors locally, pre-PR)
ci-smoke:
	@echo "Simulating clean CI environment (frozen sync, throwaway venv)..."
	@rm -rf .venv_ci_smoke
	@UV_PROJECT_ENVIRONMENT=.venv_ci_smoke uv sync --frozen --group dev
	@UV_PROJECT_ENVIRONMENT=.venv_ci_smoke uv run --no-sync ruff check .
	@UV_PROJECT_ENVIRONMENT=.venv_ci_smoke uv run --no-sync ruff format --check .
	@UV_PROJECT_ENVIRONMENT=.venv_ci_smoke uv run --no-sync mypy
	@UV_PROJECT_ENVIRONMENT=.venv_ci_smoke BAYAN_OFFLINE=1 uv run --no-sync pytest tests/
	@rm -rf .venv_ci_smoke
	@echo "CI simulation passed."

# 7. WSL2 Host RAM Survival (Flushes OS pagecache to prevent 12GB OOM)
clean-wsl-ram:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
	sudo sysctl -w vm.drop_caches=3

# 8. Model Weight Sync (offline-ready; CATT .pt + Whisper/VITS safetensors)
#    Download-only: OSS checkpoints are referenced, never re-uploaded to Hugging Face.
#    Fixtures referenced by hermetic M3.0 parity tests are committed under tests/fixtures/ (never downloaded).
weights:
	mkdir -p checkpoints/weights
	uv run huggingface-hub download oddadmix/whisper-small-arabic-dialectal --local-dir checkpoints/weights/whisper
	uv run huggingface-hub download wasmdashai/vits-ar-sa-huba --local-dir checkpoints/weights/vits
	curl -fL -o checkpoints/weights/catt_eo.pt \
		https://github.com/abjadai/catt/releases/download/v2/best_eo_mlm_ns_epoch_193.pt

# 9. Tokenizer Training + Diagnostic Report (M1.1; Calculate & Report paradigm)
#    Trains the 16k clitic-optimized Arabic BPE, then emits the diagnostic report
#    (stdout + logs/tokenizer_metrics.json + Trackio run). The report NEVER gates
#    the pipeline. Bootstrap corpus until the real SDAIA corpus lands in M1.2.
CORPUS ?= tests/fixtures/dialect_corpus.txt
tokenize:
	mkdir -p checkpoints logs
	uv run python -m bayan_slm_engine.tokenizer.bpe_trainer --corpus $(CORPUS) --output checkpoints/tokenizer.json
	uv run python -m bayan_slm_engine.tokenizer.verify_vocab --tokenizer checkpoints/tokenizer.json --corpus $(CORPUS)
