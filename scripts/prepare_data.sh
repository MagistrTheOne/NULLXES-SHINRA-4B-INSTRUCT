#!/usr/bin/env bash
set -euo pipefail
# Clean + pack SHINRA pretrain / SFT / DPO corpora.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
TOK="${TOK:-tokenizer/artifacts}"

python -m data.build_pretrain \
  --work-dir data \
  --tokenizer "$TOK" \
  --sequence-length 8192 \
  ${MAX_DOCS:+--max-docs-per-source "$MAX_DOCS"}

python -m data.build_sft \
  --tokenizer "$TOK" \
  --output-dir data/packed/sft \
  --sequence-length 8192 \
  ${MAX_DOCS:+--max-docs-per-source "$MAX_DOCS"}

python -m data.build_dpo \
  --tokenizer "$TOK" \
  --output-dir data/packed/dpo \
  --max-length 8192 \
  ${MAX_DOCS:+--max-docs-per-source "$MAX_DOCS"}
