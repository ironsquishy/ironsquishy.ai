from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def load_base_model(base_model: str, runtime: dict):
    print("[model] Loading base model...")
    print(
        f"[model] Runtime settings -> device={runtime['device']}, "
        f"dtype={runtime['dtype']}, use_4bit={runtime['use_4bit']}"
    )

    if runtime["use_4bit"]:
        print("[model] Using 4-bit quantized loading path")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=runtime["dtype"],
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=runtime["dtype"],
        )
    else:
        print("[model] Using full precision / non-quantized loading path")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=runtime["dtype"],
        )

        if runtime["device"] != "cpu":
            print(f"[model] Moving model to device: {runtime['device']}")
            model = model.to(runtime["device"])

    print("[model] Base model loaded successfully")
    return model