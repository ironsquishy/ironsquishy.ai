import argparse
import json
import yaml
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.get_system_prompt import get_system_prompt


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

    base_model = cfg["base_model"]
    adapter_dir = cfg["adapter_dir"]

    dtype = torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        quantization_config=quantization_config,
        device_map={"":0}
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    system_prompt = get_system_prompt()

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