#!/usr/bin/env bash
set -euo pipefail
# SGLang is the agent/tool-calling runtime target for SHINRA-Agent.
# SHINRA v1 is text-only RoPE.

CKPT="${1:?path or Hub id of SHINRA checkpoint}"
PORT="${PORT:-30000}"

if [[ "${SHINRA_SGLANG_BACKEND:-transformers_http}" == "transformers_http" ]]; then
  python -m inference.server --model "$CKPT" --port "$PORT"
  exit 0
fi

python -m sglang.launch_server \
  --model-path "$CKPT" \
  --trust-remote-code
