# ironsquishy.ai

A modular framework for training, evaluating, and deploying custom LLM agents using LoRA / QLoRA across CUDA, macOS (MPS), and CPU environments.

---

## 🚀 Overview

This project enables you to:

* Fine-tune open-source LLMs (Gemma-based currently)
* Run local inference with or without adapters
* Support multiple environments:

  * NVIDIA CUDA (recommended)
  * Apple Silicon (MPS)
  * CPU fallback
* Export models to GGUF (llama.cpp / edge deployment)

---

## 📦 Repository Structure

```id="repo-structure"
ironsquishy.ai/
├── configs/        # Base + environment-specific configs
├── scripts/        # Training, inference, evaluation, export
├── utils/          # Shared logic (runtime, model loading, prompts)
├── data/           # Raw + processed datasets
├── prompts/        # System prompts / personalities
├── deploy/         # Deployment configs
└── adapters/       # Saved LoRA adapters
```

---

## ⚙️ Installation

### 1. Create virtual environment

```id="setup-venv"
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

---

### 2. Install dependencies

#### ✅ Base install (works everywhere)

```id="install-base"
pip install -r requirements.txt
```

---

#### 🚀 CUDA (recommended for training)

```id="install-cuda"
pip install -r requirements-cuda.txt
```

---

#### 🧪 Dev tools (optional)

```id="install-dev"
pip install -r requirements-dev.txt
```

---

## 🧠 Dependency Strategy

This repo uses a **split dependency model**:

| File                    | Purpose                          |
| ----------------------- | -------------------------------- |
| `requirements.txt`      | Cross-platform core dependencies |
| `requirements-cuda.txt` | CUDA-only extras (bitsandbytes)  |
| `requirements-dev.txt`  | Testing / linting                |

### Why this matters

* Avoids CUDA-only install failures on Mac
* Keeps environments lightweight
* Makes debugging easier

---

## ⚡ Quick Start

---

### 🔹 Test base model (no adapter)

Always verify this first.

#### macOS / MPS

```id="run-base-mps"
python scripts/run_local_inference.py \
  --config configs/inference/mps.yaml \
  --prompt "Explain caching in distributed systems" \
  --device mps \
  --base-only
```

---

#### CUDA

```id="run-base-cuda"
python scripts/run_local_inference.py \
  --config configs/inference/cuda.yaml \
  --prompt "Explain caching in distributed systems" \
  --device cuda \
  --base-only
```

---

## 🏋️ Training

---

### Prepare data

```id="prepare-data"
python scripts/prepare_data.py \
  --input data/raw/sample_conversations.jsonl \
  --output data/processed/train.jsonl
```

---

### Train (CUDA)

```id="train-cuda"
python scripts/train_lora.py \
  --config configs/training/cuda.yaml \
  --device cuda
```

---

### Train (Mac debug only)

```id="train-mps"
python scripts/train_lora.py \
  --config configs/training/mps.yaml \
  --device mps
```

---

## 🤖 Inference (with adapter)

```id="run-inference"
python scripts/run_local_inference.py \
  --config configs/inference/cuda.yaml \
  --prompt "How do I design a scalable API?" \
  --device cuda
```

---

## 🧪 Evaluation

```id="evaluate"
python scripts/evaluate.py \
  --config configs/inference/cuda.yaml \
  --eval-file data/eval/eval_prompts.jsonl
```

---

## 🔧 Merge Adapter

```id="merge-adapter"
python scripts/merge_adapter.py \
  --base-model google/gemma-4-E2B-it \
  --adapter adapters/latest \
  --output merged_model/
```

---

## 📦 Export (GGUF)

```id="export-gguf"
bash scripts/export_gguf.sh
```

---

## 🧠 Config System

This repo uses **config inheritance**.

### Base configs

```id="base-configs"
configs/base/training.yaml
configs/base/inference.yaml
```

### Environment overrides

```id="env-configs"
configs/training/cuda.yaml
configs/training/mps.yaml
configs/training/cpu.yaml

configs/inference/cuda.yaml
configs/inference/mps.yaml
configs/inference/cpu.yaml
```

Loaded via:

```id="load-config"
utils/load_config.py
```

---

## 🧩 Utilities

Key shared modules:

```id="utils"
resolve_runtime.py
load_base_model.py
load_tokenizer.py
build_prompt.py
validate_adapter.py
get_generation_kwargs.py
```

---

## 🧪 Debugging

---

### Test without adapter

```id="debug-base"
--base-only
```

---

### Common issues

#### Adapter gives empty output

* Adapter likely mismatched with base model
* Fix: retrain with same base model

---

#### CUDA not found (WSL)

```id="check-cuda"
nvcc --version
```

Install CUDA toolkit if missing.

---

#### Slow Mac inference

* Expected (no CUDA / no 4-bit)
* Reduce tokens:

```id="reduce-tokens"
max_new_tokens: 64
```

---

#### NaN / invalid generation

```id="safe-generation"
do_sample=False
remove_invalid_values=True
```

---

## 🔥 Recommended Workflow

```id="workflow"
1. Run base model (no adapter)
2. Verify prompt + output
3. Train adapter (CUDA)
4. Test adapter inference
5. Evaluate model
6. Merge + export
7. Deploy
```

---

## ⚠️ Important Rules

* Adapter must match base model
* Prompt format must match training
* CUDA required for efficient training
* Mac is for debugging, not heavy training

---

## 🧭 Future Goals

* GGUF optimization
* OpenClaw integration
* Multi-model support

---

## 👨‍💻 Author

Allen Space
Project: ironsquishy.ai

---
