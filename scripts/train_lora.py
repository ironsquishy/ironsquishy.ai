import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils.load_config import load_config
from utils.resolve_runtime import resolve_runtime
from utils.resolve_device_overrides import resolve_device_overrides
from utils.load_tokenizer import load_tokenizer
from utils.load_base_model import load_base_model


def format_example(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def main() -> None:
    print("[main] Starting train_lora.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()

    print(f"[main] Loading config from: {args.config}")
    cfg = load_config(args.config)
    print("[main] Config loaded successfully")

    base_model = cfg["base_model"]
    train_file = cfg["train_file"]
    output_dir = cfg["output_dir"]

    print("[runtime] Resolving runtime")
    runtime = resolve_runtime(args.device)
    print(f"[runtime] Resolved runtime: {runtime}")

    print(f"[config] Resolving device overrides for device={runtime['device']}")
    overrides = resolve_device_overrides(cfg, runtime["device"])
    print(f"[config] Resolved overrides: {overrides}")

    if overrides["use_4bit"] is not None:
        runtime["use_4bit"] = overrides["use_4bit"]

    if runtime["device"] == "cuda":
        runtime["dtype"] = torch.float16 if overrides["fp16"] else torch.float32
    else:
        runtime["dtype"] = torch.float32

    max_length = overrides["max_length"]
    batch_size = overrides["per_device_train_batch_size"]
    grad_accum = overrides["gradient_accumulation_steps"]

    print("\n========== EFFECTIVE TRAINING CONFIG ==========")
    print(f"Base model:                 {base_model}")
    print(f"Train file:                 {train_file}")
    print(f"Output dir:                 {output_dir}")
    print(f"Resolved device:            {runtime['device']}")
    print(f"Resolved dtype:             {runtime['dtype']}")
    print(f"Use 4-bit:                  {runtime['use_4bit']}")
    print(f"Max length:                 {max_length}")
    print(f"Batch size:                 {batch_size}")
    print(f"Gradient accumulation:      {grad_accum}")
    print(f"Num train epochs:           {cfg.get('num_train_epochs', 2)}")
    print(f"Learning rate:              {cfg.get('learning_rate', 2e-4)}")
    print(f"LoRA r:                     {cfg.get('lora_r', 8)}")
    print(f"LoRA alpha:                 {cfg.get('lora_alpha', 16)}")
    print(f"LoRA dropout:               {cfg.get('lora_dropout', 0.05)}")
    print(f"LoRA target modules:        {cfg.get('target_modules', 'all-linear')}")
    print("===============================================\n")

    if not Path(train_file).exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    print("[tokenizer] Loading tokenizer")
    tokenizer = load_tokenizer(base_model)
    print("[tokenizer] Tokenizer loaded successfully")

    print("[model] Loading base model")
    model = load_base_model(base_model, runtime)
    print("[model] Base model loaded successfully")

    print("[model] Applying training-specific model settings")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if runtime["use_4bit"]:
        print("[model] Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)

    print("[lora] Building LoRA config")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        target_modules=cfg.get("target_modules", "all-linear"),
    )

    print("[lora] Applying LoRA adapters")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"[data] Loading dataset from: {train_file}")
    dataset = load_dataset("json", data_files=train_file)["train"]
    print(f"[data] Dataset loaded with {len(dataset)} rows")

    print("[data] Formatting examples with chat template")

    def format_chat(example: dict) -> dict:
        text = format_example(tokenizer, example["messages"])
        return {"text": text}

    dataset = dataset.map(format_chat)
    print("[data] Formatting complete")

    print("[data] Tokenizing dataset")

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
    print("[data] Tokenization complete")

    print("[collator] Creating data collator")
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print("[trainer] Building training arguments")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.get("learning_rate", 2e-4),
        num_train_epochs=cfg.get("num_train_epochs", 2),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        fp16=(runtime["device"] == "cuda" and runtime["dtype"] == torch.float16),
        gradient_checkpointing=True,
        report_to="none",
    )

    print("[trainer] Creating trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    if runtime["device"] == "cuda":
        print("[cuda] Clearing CUDA cache before training")
        torch.cuda.empty_cache()

    print("[train] Starting training")
    trainer.train()
    print("[train] Training finished successfully")

    print(f"[save] Saving adapter to: {output_dir}")
    model.save_pretrained(output_dir)

    print(f"[save] Saving tokenizer to: {output_dir}")
    tokenizer.save_pretrained(output_dir)

    print(f"[done] Saved adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()