from transformers import AutoTokenizer

def load_tokenizer(base_model: str):
    print(f"[tokenizer] Loading tokenizer for model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token is None:
        print("[tokenizer] No pad token found, setting pad token to eos token")
        tokenizer.pad_token = tokenizer.eos_token

    print("[tokenizer] Tokenizer loaded successfully")
    return tokenizer