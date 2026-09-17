#!/usr/bin/env bash
set -euo pipefail
# Stage 3: DPO alignment on SHINRA-4B-INSTRUCT.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
SFT="${1:?path to SHINRA-4B-INSTRUCT SFT checkpoint}"

accelerate launch \
  --config_file configs/accelerate_a100.yaml \
  -m training.dpo \
  --config configs/shinra_4b.yaml \
  --train-config configs/dpo_a100.yaml \
  --data-dir data/packed/dpo \
  --tokenizer tokenizer/artifacts \
  --output-dir outputs/shinra-4b-instruct-dpo \
  --sft-from "$SFT" \
  --wandb-project nullxes-shinra \
  --wandb-run-name shinra-4b-dpo
