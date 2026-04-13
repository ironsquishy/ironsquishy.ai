from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
import argparse

import torch
from peft import PeftModel

from utils.load_config import load_config
from utils.resolve_runtime import resolve_runtime
from utils.load_tokenizer import load_tokenizer
from utils.load_base_model import load_base_model
from utils.validate_adapter import validate_adapter
from utils.build_prompt import build_prompt
from utils.get_generation_kwargs import get_generation_kwargs
from utils.get_system_prompt import get_system_prompt


def main() -> None:
    print("[main] Starting run_local_inference.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Run inference with the base model only, without loading the adapter",
    )
    args = parser.parse_args()

    print(f"[main] Loading config from: {args.config}")
    cfg = load_config(args.config)
    print("[main] Config loaded successfully")

    base_model = cfg["base_model"]
    adapter_dir = cfg["adapter_dir"]

    print("[runtime] Resolving runtime")
    runtime = resolve_runtime(args.device)
    print(f"[runtime] Resolved runtime: {runtime}")

    print("\n========== EFFECTIVE INFERENCE CONFIG ==========")
    print(f"Base model:                 {base_model}")
    print(f"Adapter dir:                {adapter_dir}")
    print(f"Resolved device:            {runtime['device']}")
    print(f"Resolved dtype:             {runtime['dtype']}")
    print(f"Use 4-bit:                  {runtime['use_4bit']}")
    print(f"Base-only mode:             {args.base_only}")
    print(f"User prompt:                {args.prompt}")
    print("================================================\n")

    print("[tokenizer] Loading tokenizer")
    tokenizer = load_tokenizer(base_model)
    print("[tokenizer] Tokenizer loaded successfully")

    print("[model] Loading base model")
    base = load_base_model(base_model, runtime)
    print("[model] Base model loaded successfully")

    print("[model] Resolving final inference model")
    if args.base_only:
        print("[model] Base-only mode enabled, skipping adapter load")
        model = base
    else:
        print("[adapter] Validating adapter")
        validate_adapter(adapter_dir)
        print("[adapter] Adapter validation successful")

        print("[adapter] Loading adapter onto base model")
        model = PeftModel.from_pretrained(base, adapter_dir)
        print("[adapter] Adapter loaded successfully")

    print("[model] Setting model to eval mode")
    model.eval()
    print("[model] Model ready for inference")

    print("[prompt] Loading system prompt")
    system_prompt = get_system_prompt()
    print("[prompt] System prompt loaded")

    print("[prompt] Building final prompt with chat template")
    prompt = build_prompt(tokenizer, system_prompt, args.prompt)
    print("[prompt] Prompt built successfully")

    print("[prompt] Prompt preview:")
    print("---------- PROMPT START ----------")
    print(prompt)
    print("----------- PROMPT END -----------")

    print("[tokenizer] Tokenizing prompt")
    inputs = tokenizer(prompt, return_tensors="pt")
    print("[tokenizer] Prompt tokenized successfully")

    if runtime["device"] == "cuda":
        print("[device] Moving inputs to CUDA model device")
        inputs = inputs.to(model.device)
    else:
        print(f"[device] Moving inputs to device: {runtime['device']}")
        inputs = {k: v.to(runtime["device"]) for k, v in inputs.items()}

    print("[generate] Resolving generation kwargs")
    generation_kwargs = get_generation_kwargs(cfg, runtime)
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    print("[generate] Final generation kwargs:")
    for key, value in generation_kwargs.items():
        print(f"  - {key}: {value}")

    print("[generate] Starting generation")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            **generation_kwargs,
        )
    print("[generate] Generation finished successfully")

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output[0][input_len:]

    print(f"[output] Input token length: {input_len}")
    print(f"[output] Number of generated tokens: {generated_ids.shape[0]}")
    print(f"[output] First 20 generated token ids: {generated_ids[:20].tolist()}")

    print("[output] Decoding response-only tokens")
    response = tokenizer.batch_decode(
        output[:, input_len:],
        skip_special_tokens=True,
    )[0].strip()

    print("\n========== MODEL RESPONSE ==========")
    print(response)
    print("====================================")


if __name__ == "__main__":
    main()