# Default shell
SHELL := /bin/bash

# Default variables
PYTHON ?= python
DEVICE ?= cuda
PROMPT ?= Explain caching in distributed systems
TRAIN_CONFIG ?= configs/training/$(DEVICE).yaml
INFER_CONFIG ?= configs/inference/$(DEVICE).yaml
TRAIN_INPUT ?= data/raw/sample_conversations.jsonl
TRAIN_OUTPUT ?= data/processed/train.jsonl
EVAL_FILE ?= data/eval/eval_prompts.jsonl
BASE_MODEL ?= google/gemma-4-E2B-it
ADAPTER_DIR ?= adapters/latest
MERGED_OUTPUT ?= merged_model

.PHONY: help prepare-data train infer base-infer eval merge export clean

help:
	@echo "Available targets:"
	@echo "  make prepare-data         Prepare raw JSONL training data"
	@echo "  make train                Train LoRA adapter"
	@echo "  make infer                Run inference with adapter"
	@echo "  make base-infer           Run inference with base model only"
	@echo "  make eval                 Evaluate model"
	@echo "  make merge                Merge adapter into base model"
	@echo "  make export               Export model to GGUF"
	@echo "  make clean                Remove generated artifacts"
	@echo ""
	@echo "Common variables:"
	@echo "  DEVICE=cuda|mps|cpu"
	@echo "  PROMPT='Your prompt here'"
	@echo "  TRAIN_CONFIG=path/to/config.yaml"
	@echo "  INFER_CONFIG=path/to/config.yaml"
	@echo ""
	@echo "Examples:"
	@echo "  make train DEVICE=cuda"
	@echo "  make infer DEVICE=mps PROMPT='Explain REST vs GraphQL'"
	@echo "  make base-infer DEVICE=cpu PROMPT='What is caching?'"

prepare-data:
	@echo "[make] Preparing training data..."
	$(PYTHON) scripts/prepare_data.py \
		--input $(TRAIN_INPUT) \
		--output $(TRAIN_OUTPUT)

train:
	@echo "[make] Training adapter..."
	@echo "[make] DEVICE=$(DEVICE)"
	@echo "[make] TRAIN_CONFIG=$(TRAIN_CONFIG)"
	$(PYTHON) scripts/train_lora.py \
		--config $(TRAIN_CONFIG) \
		--device $(DEVICE)

infer:
	@echo "[make] Running inference with adapter..."
	@echo "[make] DEVICE=$(DEVICE)"
	@echo "[make] INFER_CONFIG=$(INFER_CONFIG)"
	@echo "[make] PROMPT=$(PROMPT)"
	$(PYTHON) scripts/run_local_inference.py \
		--config $(INFER_CONFIG) \
		--prompt "$(PROMPT)" \
		--device $(DEVICE)

base-infer:
	@echo "[make] Running base-model-only inference..."
	@echo "[make] DEVICE=$(DEVICE)"
	@echo "[make] INFER_CONFIG=$(INFER_CONFIG)"
	@echo "[make] PROMPT=$(PROMPT)"
	$(PYTHON) scripts/run_local_inference.py \
		--config $(INFER_CONFIG) \
		--prompt "$(PROMPT)" \
		--device $(DEVICE) \
		--base-only

eval:
	@echo "[make] Evaluating model..."
	@echo "[make] DEVICE=$(DEVICE)"
	@echo "[make] INFER_CONFIG=$(INFER_CONFIG)"
	@echo "[make] EVAL_FILE=$(EVAL_FILE)"
	$(PYTHON) scripts/evaluate.py \
		--config $(INFER_CONFIG) \
		--eval-file $(EVAL_FILE) \
		--device $(DEVICE)

merge:
	@echo "[make] Merging adapter into base model..."
	@echo "[make] BASE_MODEL=$(BASE_MODEL)"
	@echo "[make] ADAPTER_DIR=$(ADAPTER_DIR)"
	@echo "[make] MERGED_OUTPUT=$(MERGED_OUTPUT)"
	$(PYTHON) scripts/merge_adapter.py \
		--base-model $(BASE_MODEL) \
		--adapter $(ADAPTER_DIR) \
		--output $(MERGED_OUTPUT)

export:
	@echo "[make] Exporting model to GGUF..."
	bash scripts/export_gguf.sh

clean:
	@echo "[make] Cleaning generated files..."
	rm -rf data/processed/*.jsonl
	rm -rf $(MERGED_OUTPUT)

doctor:
	@echo "========== ironsquishy.ai Doctor =========="
	@echo ""
	@echo "[system] Python"
	@which $(PYTHON)
	@$(PYTHON) --version
	@echo ""

	@echo "[system] Pip"
	@pip --version
	@echo ""

	@echo "[python] Checking core libraries..."
	@$(PYTHON) -c "import torch; print('torch:', torch.__version__)"
	@$(PYTHON) -c "import transformers; print('transformers:', transformers.__version__)"
	@$(PYTHON) -c "import accelerate; print('accelerate:', accelerate.__version__)"
	@$(PYTHON) -c "import peft; print('peft:', peft.__version__)"
	@$(PYTHON) -c "import datasets; print('datasets:', datasets.__version__)"
	@echo ""

	@echo "[hardware] Torch backend"
	@$(PYTHON) -c "import torch; print('CUDA available:', torch.cuda.is_available())"
	@$(PYTHON) -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
	@echo ""

	@echo "[cuda] Details (if available)"
	@$(PYTHON) -c "import torch; print('CUDA device count:', torch.cuda.device_count())"
	@$(PYTHON) -c "import torch; print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
	@echo ""

	@echo "[cuda] nvcc check"
	@which nvcc || echo "nvcc not found"
	@nvcc --version || echo "CUDA toolkit not installed"
	@echo ""

	@echo "[mps] Apple Silicon check"
	@$(PYTHON) -c "import platform; print('Platform:', platform.platform())"
	@echo ""

	@echo "[project] Checking directories"
	@test -d configs && echo "configs/ OK" || echo "configs/ missing"
	@test -d scripts && echo "scripts/ OK" || echo "scripts/ missing"
	@test -d utils && echo "utils/ OK" || echo "utils/ missing"
	@test -d data && echo "data/ OK" || echo "data/ missing"
	@echo ""

	@echo "[project] Checking key files"
	@test -f configs/base/training.yaml && echo "training config OK" || echo "training config missing"
	@test -f configs/base/inference.yaml && echo "inference config OK" || echo "inference config missing"
	@test -f scripts/train_lora.py && echo "train script OK" || echo "train script missing"
	@test -f scripts/run_local_inference.py && echo "inference script OK" || echo "inference script missing"
	@echo ""

	@echo "[doctor] Done"
	@echo "==========================================="