"""Held-out perplexity on packed parquet shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.collator import PackedParquetDataset, collate_lm


@torch.no_grad()
def evaluate_perplexity(model_path: str, data_dir: str, batch_size: int, max_batches: int | None) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    loader = DataLoader(
        PackedParquetDataset(data_dir),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_lm,
    )
    nll_sum = 0.0
    token_count = 0
    for i, batch in enumerate(tqdm(loader, desc="ppl")):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        labels = batch["labels"]
        valid = (labels[:, 1:] != -100).sum().item()
        nll_sum += float(out.loss.item()) * valid
        token_count += valid
    mean_nll = nll_sum / max(token_count, 1)
    return {
        "nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 20)),
        "tokens": token_count,
        "model": model_path,
        "data": data_dir,
        "vocab_size": tokenizer.vocab_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SHINRA held-out perplexity")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", default="evaluation/results/perplexity.json")
    args = parser.parse_args()
    result = evaluate_perplexity(args.model, args.data_dir, args.batch_size, args.max_batches)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
