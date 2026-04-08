import argparse
import yaml
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils.get_device import get_device


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_example(tokenizer, messages: list[dict]) -> str:
    # Gemma IT models should use the tokenizer chat template rather than
    # hand-built special tokens.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    device = get_device()

    cfg = load_config(args.config)
    base_model = cfg["base_model"]
    train_file = cfg["train_file"]
    output_dir = cfg["output_dir"]

    use_fp16 = cfg.get("fp16", True)
    compute_dtype = torch.float16 if use_fp16 else torch.float32

    device_cfg = cfg.get("device_overrides", {}).get(device, {})

    max_length = device_cfg.get("max_length", cfg.get("max_length", 512))
    batch_size = device_cfg.get("per_device_train_batch_size", 1)

    print("Loaded configurations...")
    print(f"base model: {base_model}\n Using fp16: {use_fp16}\n Comput type: {compute_dtype}\n")
    print(f"Using device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    # Gemma tokenizers typically have an eos token; set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Finished tokenizer...")

    if device == "cuda":
        # QLoRA path (your current setup)
        print("Using CUDA...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print("Finished quantization configuration...")

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
        )

    elif device == "mps":
        # Mac path (NO 4-bit)
        print("Using Mac...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16 if use_fp16 else torch.float32,
        ).to("mps")

    else:
        # CPU fallback
        print("Using CPU Fallback...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float32,
        )

    print("Finished loading model...")

    if device != "cuda":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    print("Finished additional model settings...")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        target_modules="all-linear",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=train_file)["train"]

    def format_chat(example: dict) -> dict:
        text = format_example(tokenizer, example["messages"])
        return {"text": text}

    dataset = dataset.map(format_chat)

    def tokenize_fn(example: dict) -> dict:
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        learning_rate=cfg.get("learning_rate", 2e-4),
        num_train_epochs=cfg.get("num_train_epochs", 2),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        fp16=use_fp16,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    torch.cuda.empty_cache()
    trainer.train()

    print("Finished training model...")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved adapter to {output_dir}")


if __name__ == "__main__":
    main()