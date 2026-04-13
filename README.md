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
├── Makefile        # Primary workflow commands
├── Dockerfile      # Containerized environment
├── configs/        # Base + environment configs
├── scripts/        # Training, inference, evaluation
├── utils/          # Shared runtime + helpers
├── data/           # Raw + processed datasets
├── prompts/        # System prompts
├── adapters/       # LoRA adapters
└── requirements*.txt
```

---

# ⚙️ Local Setup (Recommended First)

## 1. Create virtual environment

```bash id="venv"
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

---

## 2. Install dependencies

### Base

```bash id="install-base"
pip install -r requirements.txt
```

### CUDA (recommended)

```bash id="install-cuda"
pip install -r requirements-cuda.txt
```

---

# 🩺 Environment Check

```bash id="doctor"
make doctor
```

Use this before running anything.

---

# ⚡ Makefile Workflow

## Show commands

```bash id="make-help"
make help
```

---

## Test base model (IMPORTANT)

```bash id="base-infer"
make base-infer DEVICE=mps PROMPT="Explain caching"
```

---

## Prepare data

```bash id="prepare-data"
make prepare-data
```

---

## Train

```bash id="train"
make train DEVICE=cuda
```

---

## Inference (adapter)

```bash id="infer"
make infer DEVICE=cuda PROMPT="How do I design a scalable API?"
```

---

## Evaluate

```bash id="eval"
make eval DEVICE=cuda
```

---

## Merge adapter

```bash id="merge"
make merge
```

---

## Export GGUF

```bash id="export"
make export
```

---

# 🧠 Config System

Configs use inheritance.

```text id="configs"
configs/base/training.yaml
configs/base/inference.yaml

configs/training/cuda.yaml
configs/training/mps.yaml
configs/training/cpu.yaml

configs/inference/cuda.yaml
configs/inference/mps.yaml
configs/inference/cpu.yaml
```

---

# 🧪 Debugging Tips

### Run without adapter

```bash id="debug"
make base-infer DEVICE=cpu PROMPT="What is caching?"
```

---

### Common issues

* Adapter mismatch → retrain
* CUDA missing → install toolkit
* Mac slow → expected (no 4-bit)
* NaN outputs → disable sampling

---

# 🐳 Docker (Optional but Recommended for Consistency)

Docker provides a reproducible environment and avoids dependency issues.

---

## 1. Create `.env`

```dotenv id="env"
DEVICE=cpu
PROMPT=Explain caching in distributed systems
BASE_MODEL=google/gemma-4-E2B-it
TRANSFORMERS_CACHE=/app/.cache
```

---

## 2. Build image

```bash id="docker-build"
docker build -t ironsquishy-ai .
```

---

## 3. Run environment check

```bash id="docker-doctor"
docker run --rm -it \
  --env-file .env \
  -v "$(pwd):/app" \
  ironsquishy-ai \
  make doctor
```

---

## 4. Run inference

```bash id="docker-infer"
docker run --rm -it \
  --env-file .env \
  -v "$(pwd):/app" \
  ironsquishy-ai \
  make infer
```

---

## 5. Train

```bash id="docker-train"
docker run --rm -it \
  --env-file .env \
  -v "$(pwd):/app" \
  ironsquishy-ai \
  make train
```

---

## 6. Override variables

```bash id="docker-override"
docker run --rm -it \
  --env-file .env \
  -e DEVICE=cuda \
  -v "$(pwd):/app" \
  ironsquishy-ai \
  make infer PROMPT="Explain vector databases"
```

---

## 🧠 Notes on Docker

* `.env` is passed via `--env-file`
* `-v $(pwd):/app` mounts your project (no rebuild needed)
* Use CUDA Docker separately if needed (`--gpus all`)

---

# 🔥 Recommended Workflow

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

# ⚠️ Important Rules

* Adapter must match base model
* Prompt format must match training
* CUDA required for real training
* Mac is for debugging only

---

# 🧭 Future Goals

* GGUF optimization
* OpenClaw integration
* Multi-model support

---

# 👨‍💻 Author

Allen Space
Project: ironsquishy.ai

---
