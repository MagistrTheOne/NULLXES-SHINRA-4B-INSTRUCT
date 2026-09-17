#!/usr/bin/env bash
set -euo pipefail
# Stage 1: pretrain NULLXES SHINRA-4B-BASE on 8× A100 80GB.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

accelerate launch \
  --config_file configs/accelerate_a100.yaml \
  -m training.pretrain \
  --config configs/shinra_4b.yaml \
  --train-config configs/pretrain_a100.yaml \
  --data-dir data/packed/pretrain \
  --tokenizer tokenizer/artifacts \
  --output-dir outputs/shinra-4b-base \
  --attention-implementation sdpa \
  --wandb-project nullxes-shinra \
  --wandb-run-name shinra-4b-pretrain
