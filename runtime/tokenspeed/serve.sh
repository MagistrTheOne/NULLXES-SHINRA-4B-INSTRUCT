#!/usr/bin/env bash
set -euo pipefail
# TokenSpeed: same deployment contract as vLLM/SGLang — inference only.

CKPT="${1:?path or Hub id of SHINRA checkpoint}"
PORT="${PORT:-8000}"

if [[ "${SHINRA_TOKENSPEED_BACKEND:-transformers_http}" == "transformers_http" ]]; then
  python -m inference.server --model "$CKPT" --port "$PORT"
  exit 0
fi

TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 tokenspeed serve "$CKPT" \
  --trust-remote-code \
  --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":8192}}'
