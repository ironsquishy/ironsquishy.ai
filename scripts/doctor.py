#!/usr/bin/env python3

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def print_header(title: str) -> None:
    print(f"\n[{title}]")


def check_command(name: str) -> None:
    path = shutil.which(name)
    if path:
        print(f"[OK] {name}: {path}")
    else:
        print(f"[WARN] {name}: not found")


def check_package(name: str, import_name: str | None = None) -> None:
    mod_name = import_name or name
    try:
        module = importlib.import_module(mod_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {name}: {version}")
    except Exception as exc:
        print(f"[WARN] {name}: not importable ({exc})")


def run_command(cmd: list[str]) -> None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            output = result.stdout.strip() or result.stderr.strip() or "[OK]"
            print(output)
        else:
            output = result.stderr.strip() or result.stdout.strip() or "command failed"
            print(f"[WARN] {' '.join(cmd)} -> {output}")
    except Exception as exc:
        print(f"[WARN] failed to run {' '.join(cmd)}: {exc}")


def check_paths() -> None:
    required = [
        REPO_ROOT / "configs",
        REPO_ROOT / "scripts",
        REPO_ROOT / "utils",
        REPO_ROOT / "data",
        REPO_ROOT / "Makefile",
        REPO_ROOT / "requirements.txt",
    ]

    for path in required:
        rel = path.relative_to(REPO_ROOT)
        if path.exists():
            print(f"[OK] {rel}")
        else:
            print(f"[WARN] missing: {rel}")


def check_config_files() -> None:
    expected = [
        "configs/base/training.yaml",
        "configs/base/inference.yaml",
        "configs/training/cuda.yaml",
        "configs/training/mps.yaml",
        "configs/training/cpu.yaml",
        "configs/inference/cuda.yaml",
        "configs/inference/mps.yaml",
        "configs/inference/cpu.yaml",
    ]

    for rel in expected:
        path = REPO_ROOT / rel
        if path.exists():
            print(f"[OK] {rel}")
        else:
            print(f"[WARN] missing: {rel}")


def check_torch_runtime() -> None:
    try:
        import torch

        print(f"[OK] torch: {torch.__version__}")
        print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
        print(f"[INFO] MPS available: {torch.backends.mps.is_available()}")

        if torch.cuda.is_available():
            try:
                print(f"[INFO] CUDA device count: {torch.cuda.device_count()}")
                print(f"[INFO] CUDA device 0: {torch.cuda.get_device_name(0)}")
            except Exception as exc:
                print(f"[WARN] could not inspect CUDA device: {exc}")
    except Exception as exc:
        print(f"[WARN] torch runtime check failed: {exc}")


def check_hf_token() -> None:
    token = os.environ.get("HF_TOKEN")

    if not token:
        print("[WARN] HF_TOKEN is NOT set")
        print("[INFO] Set HF_TOKEN in your shell or .env if you need gated/private models")
        return

    print("[OK] HF_TOKEN is set")

    try:
        from huggingface_hub import whoami
    except Exception as exc:
        print(f"[WARN] huggingface_hub not available for token validation: {exc}")
        return

    try:
        user = whoami(token=token)
        name = user.get("name") or user.get("fullname") or "unknown"
        print(f"[OK] HF_TOKEN valid for user: {name}")
    except Exception as exc:
        print(f"[WARN] HF_TOKEN invalid or unusable: {exc}")


def main() -> int:
    print("========== ironsquishy.ai Doctor ==========")
    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {sys.version.split()[0]}")
    print(f"Platform:          {platform.platform()}")
    print(f"Repo root:         {REPO_ROOT}")

    print_header("system")
    check_command("python")
    check_command("pip")
    check_command("git")
    check_command("make")
    check_command("nvcc")

    print_header("python")
    check_package("torch")
    check_package("transformers")
    check_package("accelerate")
    check_package("peft")
    check_package("datasets")
    check_package("yaml", "yaml")
    check_package("fastapi")
    check_package("uvicorn")
    check_package("huggingface_hub")

    print_header("hardware")
    check_torch_runtime()

    print_header("env")
    check_hf_token()

    print_header("project")
    check_paths()

    print_header("configs")
    check_config_files()

    print_header("optional cuda check")
    if shutil.which("nvidia-smi"):
        run_command(["nvidia-smi"])
    else:
        print("[INFO] nvidia-smi not found")

    print_header("optional nvcc check")
    if shutil.which("nvcc"):
        run_command(["nvcc", "--version"])
    else:
        print("[INFO] nvcc not found")

    print("\n========== Doctor complete ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())