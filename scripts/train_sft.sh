#!/usr/bin/env bash
set -euo pipefail
# Stage 2: instruction-tune NULLXES SHINRA-4B-INSTRUCT from BASE.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
BASE="${1:?path to SHINRA-4B-BASE checkpoint}"

accelerate launch \
  --config_file configs/accelerate_a100.yaml \
  -m training.sft \
  --config configs/shinra_4b.yaml \
  --train-config configs/sft_a100.yaml \
  --data-dir data/packed/sft \
  --tokenizer tokenizer/artifacts \
  --output-dir outputs/shinra-4b-instruct \
  --base-from "$BASE" \
  --attention-implementation sdpa \
  --wandb-project nullxes-shinra \
  --wandb-run-name shinra-4b-sft
