import argparse
import yaml
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.get_system_prompt import get_system_prompt
from utils.get_device import get_device


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    device = get_device()

    print(f"Using device: {device}")

    base_model = cfg["base_model"]
    adapter_dir = cfg["adapter_dir"]

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config=None

    if device == "cuda":
        # CUDA
        print("Loading model for CUDA device...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",
        )

    elif device == "mps":
        # Mac
        print("Loading model for Mac device...")

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float32,
        ).to("mps")

    else:
        # CPU
        print("Loading model for CPU...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float32,
        )

    print("Model eval....")
    # model = PeftModel.from_pretrained(base, adapter_dir)
    # model.eval()
    model = base
    model.eval()

    print("Finished model eval...")

    system_prompt = get_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": args.prompt},
    ]

    print("Prepare prompt...")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("PROMPT:")
    print(prompt)
    print("*********")

    inputs = tokenizer(prompt, return_tensors="pt")

    print("Finished preparing prompt...")

    if device != "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        inputs = inputs.to(model.device)

    print("Generating model...")
    print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("bos_token:", tokenizer.bos_token, tokenizer.bos_token_id)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            remove_invalid_values=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    print("output shape:", output.shape)
    
    print("first 20 generated ids:", output[0][input_len:input_len+20].tolist())
    print("token 0 decodes to:", tokenizer.decode([0], skip_special_tokens=False))
    print("*******")

    print("Retrieved response...")
    print("*********")

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output[0][input_len:]

    print("input len:", input_len)
    print("Generated ids:", generated_ids[:20].tolist())
    print("Token 0 decodes to:", tokenizer.decode([0], skip_special_tokens=False))
    print("*********")
    

    response = tokenizer.batch_decode(
        output[:, input_len:],
        skip_special_tokens=True
    )[0].strip()

    print(response)
    print("*********")
    


if __name__ == "__main__":
    main()