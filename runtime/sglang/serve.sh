#!/usr/bin/env bash
set -euo pipefail
# SGLang is the agent/tool-calling runtime target for SHINRA-Agent.
# Do not pass Qwen mrope_section. SHINRA v1 is text-only RoPE.

CKPT="${1:?path or Hub id of SHINRA checkpoint}"
PORT="${PORT:-30000}"

if [[ "${SHINRA_SGLANG_BACKEND:-transformers_http}" == "transformers_http" ]]; then
  python -m inference.server --model "$CKPT" --port "$PORT"
  exit 0
fi

SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
  --model-path "$CKPT" \
  --trust-remote-code \
  --context-length 32768 \
  --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":8192}}'
