import argparse
import yaml
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_model = cfg["base_model"]
    train_file = cfg["train_file"]
    output_dir = cfg["output_dir"]

    use_fp16 = cfg.get("fp16", True)
    dtype = torch.float16 if use_fp16 else torch.float32

    bnb_config = BitsAndBytesConfig( 
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=dtype, 
        bnb_4bit_use_double_quant=True 
    )

    print("Finished setup bnb config...")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    print("Finished tokenizer...")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("Finished getting model...")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    print("Finished other model configurations...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=train_file)["train"]

    def format_chat(example: dict) -> dict:
        messages = example["messages"]
        text = ""
        for msg in messages:
            role = msg["role"].strip().lower()
            content = msg["content"].strip()
            text += f"<|{role}|>\n{content}\n"
        text += "<|end|>"
        return {"text": text}

    dataset = dataset.map(format_chat)

    def tokenize_fn(example: dict) -> dict:
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=cfg.get("max_length", 2048),
            padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        learning_rate=cfg.get("learning_rate", 2e-4),
        num_train_epochs=cfg.get("num_train_epochs", 2),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.01),
        fp16=cfg.get("fp16", True),
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
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved adapter to {output_dir}")


if __name__ == "__main__":
    main()