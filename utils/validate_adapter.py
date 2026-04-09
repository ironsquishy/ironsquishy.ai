import os
import json
from pathlib import Path


def validate_adapter(adapter_dir: str, base_model: str) -> None:
    adapter_path = Path(adapter_dir)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")

    with adapter_config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_in_adapter = cfg.get("base_model_name_or_path")
    if base_in_adapter and base_in_adapter != base_model:
        raise ValueError(
            f"Adapter/base mismatch:\n"
            f"  adapter expects: {base_in_adapter}\n"
            f"  script uses:     {base_model}"
        )