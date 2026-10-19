#!/usr/bin/env bash
set -euo pipefail
# Compatibility launcher; SGLang has no admitted SHINRA architecture adapter.

CKPT="${1:?local path to a SHINRA checkpoint directory is required}"
PORT="${PORT:-30000}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -d "$CKPT" ]]; then
  echo "UNAVAILABLE: runtime accepts local checkpoint directories only; Hub access is disabled" >&2
  exit 2
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
(cd "$REPO_ROOT" && python -m scripts.checkpoint_admission "$CKPT")
CKPT="$(cd "$CKPT" && pwd)"

BACKEND="${SHINRA_SGLANG_BACKEND:-transformers_http}"
if [[ "$BACKEND" != "transformers_http" ]]; then
  echo "UNAVAILABLE: SHINRA has no admitted SGLang architecture adapter; backend=$BACKEND" >&2
  exit 2
fi

cd "$REPO_ROOT"
python -m inference.server --model "$CKPT" --port "$PORT"
