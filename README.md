# ironsquishy.ai

A modular framework for training, evaluating, and deploying custom LLM agents using LoRA/QLoRA with flexible runtime support across CUDA, macOS (MPS), and CPU.

---

## 🚀 Overview

This project enables you to:

* Fine-tune open-source LLMs (currently Gemma-based) using LoRA/QLoRA
* Run inference locally with or without adapters
* Support multiple hardware environments:

  * NVIDIA CUDA (recommended)
  * Apple Silicon (MPS)
  * CPU fallback
* Export models for edge deployment (llama.cpp / GGUF)

---

## 📦 Repository Structure

```
ironsquishy.ai/
├── configs/        # Base configs + environment-specific overrides
├── scripts/        # Entry points (train, inference, eval, export)
├── utils/          # Shared runtime, model loading, prompting logic
├── data/           # Raw, processed, and evaluation datasets
├── prompts/        # System prompts / personality definitions
├── deploy/         # Deployment artifacts (optional)
└── adapters/       # Saved LoRA adapters
```

---

## ⚙️ Supported Environments

| Environment | Training   | Inference     | Notes       |
| ----------- | ---------- | ------------- | ----------- |
| CUDA GPU    | ✅ Best     | ✅ Fast        | Recommended |
| macOS (MPS) | ⚠️ Limited | ✅ OK          | No 4-bit    |
| CPU         | ❌ Slow     | ⚠️ Debug only | Fallback    |

---

## 🧠 Config System (IMPORTANT)

This repo uses **config inheritance**:

### Base configs:

```
configs/base/training.yaml
configs/base/inference.yaml
```

### Environment overrides:

```
configs/training/cuda.yaml
configs/training/mps.yaml
configs/training/cpu.yaml

configs/inference/cuda.yaml
configs/inference/mps.yaml
configs/inference/cpu.yaml
```

Configs are merged automatically via `utils/load_config.py`.

---

## ⚡ Quick Start

### 1. Setup environment

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🧪 Test Base Model (No Adapter)

Always do this first.

### macOS / MPS

```
python scripts/run_local_inference.py \
  --config configs/inference/mps.yaml \
  --prompt "Explain caching in distributed systems" \
  --device mps \
  --base-only
```

---

### CUDA

```
python scripts/run_local_inference.py \
  --config configs/inference/cuda.yaml \
  --prompt "Explain caching in distributed systems" \
  --device cuda \
  --base-only
```

---

## 🏋️ Training

### Prepare data

```
python scripts/prepare_data.py \
  --input data/raw/sample_conversations.jsonl \
  --output data/processed/train.jsonl
```

---

### Train (CUDA recommended)

```
python scripts/train_lora.py \
  --config configs/training/cuda.yaml \
  --device cuda
```

---

### Train (Mac - small/debug only)

```
python scripts/train_lora.py \
  --config configs/training/mps.yaml \
  --device mps
```

---

## 🤖 Run Inference (With Adapter)

```
python scripts/run_local_inference.py \
  --config configs/inference/cuda.yaml \
  --prompt "How do I design a scalable API?" \
  --device cuda
```

---

## 🧪 Evaluate Model

```
python scripts/evaluate.py \
  --config configs/inference/cuda.yaml \
  --eval-file data/eval/eval_prompts.jsonl
```

---

## 🔧 Merge Adapter (Optional)

```
python scripts/merge_adapter.py \
  --base-model google/gemma-4-E2B-it \
  --adapter adapters/latest \
  --output merged_model/
```

---

## 📦 Export to GGUF (llama.cpp)

```
bash scripts/export_gguf.sh
```

---

## 🧠 Prompting

* System prompt defined in:

  ```
  utils/get_system_prompt.py
  ```

* Prompt formatting:

  ```
  utils/build_prompt.py
  ```

Uses model-native chat templates (Gemma-compatible).

---

## 🧩 Utilities

All shared logic is in `utils/`:

* `resolve_runtime.py` → device + dtype selection
* `load_base_model.py` → model loading (CUDA vs MPS vs CPU)
* `load_tokenizer.py`
* `build_prompt.py`
* `validate_adapter.py`
* `get_generation_kwargs.py`

---

## 🧪 Debugging

### Test base model only

```
--base-only
```

---

### Common issues

#### ❌ Adapter produces empty or bad output

* Adapter likely trained on different base model
* Fix: retrain adapter using same base model

---

#### ❌ CUDA not found in WSL

```
nvcc --version
```

If missing, install CUDA toolkit inside WSL.

---

#### ❌ Slow inference on Mac

* Expected (no CUDA / no 4-bit)
* Reduce tokens:

```
max_new_tokens: 64
```

---

#### ❌ NaN / Inf generation

Use safer generation:

```
do_sample=False
remove_invalid_values=True
```

---

## 🔥 Recommended Workflow

```
1. Run base model (no adapter)
2. Verify prompt + output
3. Train adapter (CUDA)
4. Test with adapter
5. Evaluate
6. Merge + export
7. Deploy (llama.cpp / Ollama)
```

---

## ⚠️ Important Rules

* Adapter MUST match base model
* Prompt format must match training
* CUDA required for efficient training
* Mac is for debugging, not heavy training

---

## 🧭 Future Direction

* GGUF optimization
* OpenClaw integration
* Multi-model support

---

## 🧑‍💻 Author

Built by Allen Space
Project: ironsquishy.ai

---
