#!/usr/bin/env bash
set -euo pipefail
# Target vLLM serve for SHINRA weights. Requires a vLLM Shinra plugin
# or Transformers backend with trust_remote_code.

CKPT="${1:?path or Hub id of SHINRA checkpoint}"
PORT="${PORT:-8000}"

# Phase-0 path: use the in-repo OpenAI server until a vLLM model class exists.
if [[ "${SHINRA_VLLM_BACKEND:-transformers_http}" == "transformers_http" ]]; then
  python -m inference.server --model "$CKPT" --port "$PORT"
  exit 0
fi

vllm serve "$CKPT" \
  --trust-remote-code \
  --dtype bfloat16
