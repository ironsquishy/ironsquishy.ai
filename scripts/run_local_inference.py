import argparse
import yaml
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.get_system_prompt import get_system_prompt


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_phi_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{user_prompt}\n"
        f"<|assistant|>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    base_model = cfg["base_model"]
    adapter_dir = cfg["adapter_dir"]

    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    print("Finished tokanizing...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        quantization_config=quantization_config,
        device_map={"":0}
    )

    print("Finished getting model...")

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    system_prompt = get_system_prompt()

    prompt = render_phi_prompt(system_prompt, args.prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Finished model load...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )
    print("Finished generating output")

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # response = decoded.split("<|assistant|>")[-1].strip()
    # print(tokenizer.decode(output[0], skip_special_tokens=True))
    print(decoded)


if __name__ == "__main__":
    main()