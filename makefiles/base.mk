# =========================
# ironsquishy.ai Makefile
# =========================

SHELL := /bin/bash

# -------------------------
# Core config (env-aware)
# -------------------------
PYTHON ?= python
DEVICE ?= cpu
PROMPT ?= Explain caching in distributed systems

TRAIN_CONFIG ?= configs/training/$(DEVICE).yaml
INFER_CONFIG ?= configs/inference/$(DEVICE).yaml

TRAIN_INPUT ?= data/raw/sample_conversations.jsonl
TRAIN_OUTPUT ?= data/processed/train.jsonl
EVAL_FILE ?= data/eval/eval_prompts.jsonl

BASE_MODEL ?= google/gemma-4-E2B-it
ADAPTER_DIR ?= adapters/latest
MERGED_OUTPUT ?= merged_model

# -------------------------
# Targets
# -------------------------
.PHONY: help doctor prepare-data train infer base-infer eval merge export clean clean-all

# -------------------------
# Help
# -------------------------
help:
	@echo ""
	@echo "=== ironsquishy.ai Makefile ==="
	@echo ""
	@echo "Core:"
	@echo "  make doctor            Check environment"
	@echo "  make prepare-data      Prepare dataset"
	@echo "  make train             Train LoRA adapter"
	@echo "  make infer             Run inference (adapter)"
	@echo "  make base-infer        Run inference (no adapter)"
	@echo "  make eval              Evaluate model"
	@echo "  make merge             Merge adapter"
	@echo "  make export            Export GGUF"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             Remove temp files"
	@echo "  make clean-all         Remove ALL artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make train DEVICE=cuda"
	@echo "  make infer PROMPT='Explain caching'"
	@echo ""

# -------------------------
# Doctor (Python-based)
# -------------------------
doctor:
	@echo "[make] Running doctor script..."
	@if [ -f scripts/doctor.py ]; then \
		$(PYTHON) scripts/doctor.py; \
	else \
		echo "[ERROR] scripts/doctor.py not found"; \
		echo "Make sure the doctor script exists."; \
		exit 1; \
	fi

# -------------------------
# Data
# -------------------------
prepare-data:
	@echo "[make] Preparing dataset..."
	$(PYTHON) scripts/prepare_data.py \
		--input $(TRAIN_INPUT) \
		--output $(TRAIN_OUTPUT)

# -------------------------
# Training
# -------------------------
train:
	@echo "[make] Training adapter..."
	@echo "DEVICE=$(DEVICE)"
	@echo "CONFIG=$(TRAIN_CONFIG)"
	$(PYTHON) scripts/train_lora.py \
		--config $(TRAIN_CONFIG) \
		--device $(DEVICE)

# -------------------------
# Inference
# -------------------------
infer:
	@echo "[make] Running inference..."
	@echo "DEVICE=$(DEVICE)"
	@echo "PROMPT=$(PROMPT)"
	$(PYTHON) scripts/run_local_inference.py \
		--config $(INFER_CONFIG) \
		--prompt "$(PROMPT)" \
		--device $(DEVICE)

base-infer:
	@echo "[make] Running base model inference..."
	$(PYTHON) scripts/run_local_inference.py \
		--config $(INFER_CONFIG) \
		--prompt "$(PROMPT)" \
		--device $(DEVICE) \
		--base-only

# -------------------------
# Evaluation
# -------------------------
eval:
	@echo "[make] Evaluating model..."
	$(PYTHON) scripts/evaluate.py \
		--config $(INFER_CONFIG) \
		--eval-file $(EVAL_FILE) \
		--device $(DEVICE)

# -------------------------
# Merge
# -------------------------
merge:
	@echo "[make] Merging adapter..."
	$(PYTHON) scripts/merge_adapter.py \
		--base-model $(BASE_MODEL) \
		--adapter $(ADAPTER_DIR) \
		--output $(MERGED_OUTPUT)

# -------------------------
# Export
# -------------------------
export:
	@echo "[make] Exporting GGUF..."
	bash scripts/export_gguf.sh

# -------------------------
# Clean (safe)
# -------------------------
clean:
	@echo "[make] Cleaning temp files..."
	rm -rf data/processed/*.jsonl || true
	find . -type d -name "__pycache__" -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete || true

# -------------------------
# Clean All (destructive)
# -------------------------
clean-all:
	@echo "[make] Removing ALL artifacts..."
	rm -rf data/processed || true
	rm -rf adapters || true
	rm -rf $(MERGED_OUTPUT) || true
	rm -rf .cache || true
	rm -rf outputs || true