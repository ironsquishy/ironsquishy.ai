#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-adapters/merged}"
LLAMACPP_DIR="${2:-$HOME/llama.cpp}"
OUTFILE="${3:-phi3-f16.gguf}"

python "$LLAMACPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" --outfile "$OUTFILE"

echo "GGUF export complete: $OUTFILE"
echo "Example quantization:"
echo "$LLAMACPP_DIR/build/bin/llama-quantize $OUTFILE phi3-q4_k_m.gguf Q4_K_M"