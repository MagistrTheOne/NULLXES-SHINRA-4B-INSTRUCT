#!/usr/bin/env bash
set -euo pipefail
# Target vLLM serve for SHINRA weights. Requires a vLLM Shinra plugin
# or Transformers backend with trust_remote_code. Not Qwen MRoPE.

CKPT="${1:?path or Hub id of SHINRA checkpoint}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"

# Phase-0 path: use the in-repo OpenAI server until a vLLM model class exists.
if [[ "${SHINRA_VLLM_BACKEND:-transformers_http}" == "transformers_http" ]]; then
  python -m inference.server --model "$CKPT" --port "$PORT"
  exit 0
fi

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve "$CKPT" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":8192}}'
