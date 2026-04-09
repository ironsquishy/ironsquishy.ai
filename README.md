# ironsquishy.ai

A modular framework for training, evaluating, and deploying custom LLM agents using LoRA / QLoRA across CUDA, macOS (MPS), and CPU environments.

---

## 🚀 Overview

This project enables you to:

* Fine-tune open-source LLMs (Gemma-based)
* Run local inference with or without adapters
* Support multiple environments:

  * NVIDIA CUDA (recommended)
  * Apple Silicon (MPS)
  * CPU fallback
* Export models to GGUF (llama.cpp / edge deployment)

---

## 📦 Repository Structure

```text id="repo-structure"
ironsquishy.ai/
├── Makefile        # Workflow shortcuts + environment checks
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

```bash id="setup-venv"
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

---

### 2. Install dependencies

#### ✅ Base (all platforms)

```bash id="install-base"
pip install -r requirements.txt
```

#### 🚀 CUDA (recommended for training)

```bash id="install-cuda"
pip install -r requirements-cuda.txt
```

#### 🧪 Dev tools (optional)

```bash id="install-dev"
pip install -r requirements-dev.txt
```

---

## 🩺 Environment Check (`make doctor`)

Before running anything, verify your environment:

```bash id="run-doctor"
make doctor
```

### What it checks

* Python + pip setup
* Core libraries (`torch`, `transformers`, etc.)
* CUDA availability and GPU detection
* Apple MPS support
* CUDA toolkit (`nvcc`)
* Project structure (configs, scripts, utils)

---

### When to use it

Run this if:

* training fails
* inference is slow or broken
* CUDA is not detected
* adapter behaves incorrectly
* switching machines (WSL ↔ Mac ↔ CPU)

---

## ⚡ Quick Start (Makefile)

### See all commands

```bash id="make-help"
make help
```

---

## 🧪 Test Base Model (No Adapter)

Always do this first.

```bash id="base-infer"
make base-infer DEVICE=mps PROMPT="Explain caching in distributed systems"
```

---

## 🏋️ Training

### Prepare data

```bash id="prepare-data"
make prepare-data
```

---

### Train (CUDA)

```bash id="train-cuda"
make train DEVICE=cuda
```

---

### Train (Mac debug only)

```bash id="train-mps"
make train DEVICE=mps
```

---

## 🤖 Inference (with adapter)

```bash id="infer"
make infer DEVICE=cuda PROMPT="How do I design a scalable API?"
```

---

## 🧪 Evaluation

```bash id="eval"
make eval DEVICE=cuda
```

---

## 🔧 Merge Adapter

```bash id="merge"
make merge
```

---

## 📦 Export to GGUF

```bash id="export"
make export
```

---

## 🧠 Config System

This repo uses config inheritance.

### Base configs

```text id="base-configs"
configs/base/training.yaml
configs/base/inference.yaml
```

### Environment configs

```text id="env-configs"
configs/training/cuda.yaml
configs/training/mps.yaml
configs/training/cpu.yaml

configs/inference/cuda.yaml
configs/inference/mps.yaml
configs/inference/cpu.yaml
```

Loaded via:

```text id="load-config"
utils/load_config.py
```

---

## 🧩 Utilities

Core shared modules:

```text id="utils"
resolve_runtime.py
load_base_model.py
load_tokenizer.py
build_prompt.py
validate_adapter.py
get_generation_kwargs.py
```

---

## 🧪 Debugging

### Run without adapter

```bash id="debug-base"
make base-infer DEVICE=cpu PROMPT="What is caching?"
```

---

### Common issues

#### Adapter produces bad or empty output

* Adapter likely mismatched with base model
* Fix: retrain with same base model

---

#### CUDA not detected

```bash id="check-cuda"
nvcc --version
```

If missing → install CUDA toolkit in WSL/Linux.

---

#### Slow Mac inference

* Expected (no CUDA / no 4-bit)
* Reduce tokens in config:

```yaml id="reduce-tokens"
max_new_tokens: 64
```

---

#### NaN / invalid generation

```yaml id="safe-gen"
do_sample: false
remove_invalid_values: true
```

---

## 🔥 Recommended Workflow

```text id="workflow"
1. make doctor
2. make base-infer
3. make prepare-data
4. make train DEVICE=cuda
5. make infer DEVICE=cuda
6. make eval
7. make merge
8. make export
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
