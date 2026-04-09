def get_generation_kwargs(device: str, cfg: dict) -> dict:
    if device == "cuda":
        return {
            "max_new_tokens": cfg.get("max_new_tokens", 300),
            "temperature": cfg.get("temperature", 0.3),
            "top_p": cfg.get("top_p", 0.9),
            "do_sample": cfg.get("do_sample", True),
            "remove_invalid_values": True,
        }

    return {
        "max_new_tokens": min(cfg.get("max_new_tokens", 300), 128),
        "do_sample": False,
        "remove_invalid_values": True,
    }