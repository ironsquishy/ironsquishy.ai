from copy import deepcopy
from pathlib import Path
import yaml


def deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str) -> dict:
    path_obj = Path(path)

    with path_obj.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    inherits = cfg.pop("inherits", None)
    if not inherits:
        return cfg

    parent_path = Path(inherits)
    if not parent_path.is_absolute():
        parent_path = Path.cwd() / inherits

    with parent_path.open("r", encoding="utf-8") as f:
        parent_cfg = yaml.safe_load(f)

    return deep_merge(parent_cfg, cfg)