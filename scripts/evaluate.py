import argparse
import json
import yaml
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def render_phi_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{user_prompt}\n"
        f"<|assistant|>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--eval-file", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    eval_rows = load_jsonl(args.eval_file)

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, cfg["adapter_dir"])
    model.eval()

    system_prompt = (
        "You are a concise infrastructure assistant for ironsquishy.ai. "
        "Prefer secure defaults, use the names steve server and orin server, "
        "prefer Tailscale for private traffic, and do not recommend exposing "
        "private model APIs publicly."
    )

    for row in eval_rows:
        prompt = render_phi_prompt(system_prompt, row["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.get("max_new_tokens", 300),
                temperature=cfg.get("temperature", 0.2),
                top_p=cfg.get("top_p", 0.9),
                do_sample=cfg.get("do_sample", True),
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("=" * 80)
        print(f"ID: {row['id']}")
        print(decoded)
        print("=" * 80)


if __name__ == "__main__":
    main()