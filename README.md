# ironsquishy-phi-agent

Starter repo for a compact custom assistant built around Phi and tuned for:

- OpenClaw workflows
- Tailscale networking
- reverse proxy configs
- Jetson / local LLM troubleshooting
- secure deployment defaults

## Project goals

- Start with app-layer customization first
- Fine-tune a small Phi instruct model with LoRA
- Evaluate behavior on infrastructure prompts
- Export for compact inference later
- Deploy to the orin server as a private LLM service

## Suggested base model

- microsoft/Phi-3-mini-4k-instruct


## Helpful commands and tools:

### Run it as prompt preview / scaffolding project:
Run `app/server.py` does not require downloading the model. Use to test API structure, prompt formatting, and local app wiring immediately with FastAPI + Uvicorn. 

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.server:app --host 127.0.0.1 --port 8080
```

Test example:
```
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/prompt-preview \
  -H "Content-Type: application/json" \
  -d '{"prompt":"How should I expose OpenClaw securely?"}'
```

### Run it as a local inference project on a stronger machine
It loads the base Phi model plus a LoRA adapter directory and generates text locally. This matches how Transformers and PEFT are intended to work together: load the base model in PyTorch, then attach the PEFT adapter with
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_local_inference.py \
  --config configs/inference.yaml \
  --prompt "How should I publish OpenClaw on ironsquishy.ai?"
```

### Run it as a LoRA fine-tuning project
```
# Prep data
python scripts/prepare_data.py \
  --input data/raw/sample_conversations.jsonl \
  --output data/processed/train.jsonl

# Train
python scripts/train_lora.py --config configs/training.yaml

#Eval
python scripts/evaluate.py \
  --config configs/inference.yaml \
  --eval-file data/eval/eval_prompts.jsonl
```

