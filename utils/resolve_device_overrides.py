def resolve_device_overrides(cfg: dict, device: str) -> dict:
    print(f"[config] Resolving device overrides for: {device}")

    device_cfg = cfg.get("device_overrides", {}).get(device, {})

    resolved = {
        "max_length": device_cfg.get("max_length", cfg.get("max_length", 512)),
        "per_device_train_batch_size": device_cfg.get(
            "per_device_train_batch_size",
            cfg.get("per_device_train_batch_size", 1),
        ),
        "gradient_accumulation_steps": device_cfg.get(
            "gradient_accumulation_steps",
            cfg.get("gradient_accumulation_steps", 16),
        ),
        "use_4bit": device_cfg.get("use_4bit", None),
        "fp16": device_cfg.get("fp16", cfg.get("fp16", True)),
    }

    print(f"[config] Resolved overrides: {resolved}")
    return resolved