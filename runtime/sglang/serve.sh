#!/usr/bin/env bash
set -euo pipefail
# SGLang is the agent/tool-calling runtime target for SHINRA-Agent.
# SHINRA v1 is text-only RoPE.

CKPT="${1:?local path to a SHINRA checkpoint directory is required}"
PORT="${PORT:-30000}"
if [[ ! -d "$CKPT" ]]; then
  echo "UNAVAILABLE: runtime accepts local checkpoint directories only; Hub access is disabled" >&2
  exit 2
fi
CKPT="$(cd "$CKPT" && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

if [[ "${SHINRA_SGLANG_BACKEND:-transformers_http}" == "transformers_http" ]]; then
  python -m inference.server --model "$CKPT" --port "$PORT"
  exit 0
fi

python -m sglang.launch_server \
  --model-path "$CKPT" \
  --trust-remote-code
