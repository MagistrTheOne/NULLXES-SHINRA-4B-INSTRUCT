"""Held-out perplexity on packed parquet shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs.loader import load_config
from inference.generate import load_generate_stack

from training.collator import PackedParquetDataset, collate_lm


@torch.no_grad()
def evaluate_perplexity(model_path: str, data_dir: str, batch_size: int, max_batches: int | None) -> dict:
    tokenizer, model = load_generate_stack(model_path)
    loader = DataLoader(
        PackedParquetDataset(data_dir, expected_split="validation"),
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
        ce = model.compute_loss_parts(out.logits, labels)["train_ce"]
        if not math.isfinite(ce):
            raise FloatingPointError("non-finite held-out cross-entropy")
        nll_sum += ce * valid
        token_count += valid
    mean_nll = nll_sum / max(token_count, 1)
    if token_count == 0:
        raise ValueError("held-out dataset produced zero scored next-token targets")
    return {
        "nll": mean_nll,
        "heldout_ce": mean_nll,
        "perplexity": math.exp(min(mean_nll, 20)),
        "tokens": token_count,
        "counter_version": "counter-v1",
        "model": model_path,
        "data": data_dir,
        "vocab_size": tokenizer.vocab_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SHINRA held-out perplexity")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=load_config()["evaluation"]["perplexity_batch_size"])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", default=f"{load_config()['evaluation']['report_dir']}/perplexity.json")
    args = parser.parse_args()
    result = evaluate_perplexity(args.model, args.data_dir, args.batch_size, args.max_batches)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
