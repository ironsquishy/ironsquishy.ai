import torch


def resolve_runtime(preferred_device: str | None = None) -> dict:
    print("[runtime] Resolving runtime device...")

    if preferred_device == "cuda" or (
        preferred_device is None and torch.cuda.is_available()
    ):
        runtime = {
            "device": "cuda",
            "dtype": torch.float16,
            "use_4bit": True,
        }
        print("[runtime] Selected CUDA runtime")
        return runtime

    if preferred_device == "mps" or (
        preferred_device is None and torch.backends.mps.is_available()
    ):
        runtime = {
            "device": "mps",
            "dtype": torch.float32,  # safer default for Mac
            "use_4bit": False,
        }
        print("[runtime] Selected MPS runtime")
        return runtime

    runtime = {
        "device": "cpu",
        "dtype": torch.float32,
        "use_4bit": False,
    }
    print("[runtime] Selected CPU runtime")
    return runtime