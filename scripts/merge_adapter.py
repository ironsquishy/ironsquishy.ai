import argparse
import yaml
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    base_model = cfg["base_model"]

    dtype = torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Finished tokanizing...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("Finished bnb setup...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        quantization_config=quantization_config,
        device_map={"":0}
    )

    print("Finished pretrain model load...")
    
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    merged = model.merge_and_unload()

    merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()