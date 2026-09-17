#!/usr/bin/env bash
set -euo pipefail
# Train the proprietary NULLXES SHINRA Unigram tokenizer (131072).
# Feed a representative mix: web, wikipedia, books, code, STEM, multilingual.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

python -m tokenizer.train_tokenizer \
  --input "${1:?pass corpus directories or files}" \
  --output-dir tokenizer/artifacts \
  --vocab-size 131072 \
  --character-coverage 0.99995 \
  --max-chars 10000000000 \
  --input-sentence-size 20000000
