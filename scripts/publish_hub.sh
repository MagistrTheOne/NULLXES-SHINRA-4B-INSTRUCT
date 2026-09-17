#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
CKPT="${1:?checkpoint dir}"
REPO="${2:?NULLXES/SHINRA-4B-INSTRUCT}"
python -m scripts.publish_hub --checkpoint "$CKPT" --repo-id "$REPO" --private
