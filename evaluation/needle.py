"""Needle-in-a-haystack context evaluation for SHINRA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NEEDLE = "The secret NULLXES project codeword is MIDGAR-SHINRA-42."
QUESTION = "What is the secret NULLXES project codeword?"
HAY = (
    "Engineering notes on rotary embeddings, grouped-query attention, and SwiGLU "
    "fill this page. Residual streams propagate representations across decoder layers. "
)


def build_prompt(context_tokens: int, depth: float, tokenizer) -> str:
    hay_ids = tokenizer.encode(HAY, add_special_tokens=False)
    needle_ids = tokenizer.encode(NEEDLE, add_special_tokens=False)
    question_ids = tokenizer.encode("\n" + QUESTION, add_special_tokens=False)
    budget = max(context_tokens - len(needle_ids) - len(question_ids) - 8, 64)
    reps = (budget // max(len(hay_ids), 1)) + 1
    haystack = (hay_ids * reps)[:budget]
    insert_at = int(len(haystack) * depth)
    ids = haystack[:insert_at] + needle_ids + haystack[insert_at:] + question_ids
    return tokenizer.decode(ids, skip_special_tokens=True)


@torch.no_grad()
def run_needle(model_path: str, contexts: list[int], depths: list[float]) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    model.eval()
    rows = []
    for ctx in contexts:
        for depth in depths:
            prompt = build_prompt(ctx, depth, tokenizer)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            hit = "MIDGAR-SHINRA-42" in decoded
            rows.append({"context": ctx, "depth": depth, "hit": hit, "output": decoded})
    score = sum(r["hit"] for r in rows) / max(len(rows), 1)
    return {"score": score, "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Needle-in-haystack for SHINRA")
    parser.add_argument("--model", required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=[2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--depths", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--output", default="evaluation/results/needle.json")
    args = parser.parse_args()
    random.seed(0)
    result = run_needle(args.model, args.contexts, args.depths)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"score": result["score"], "n": len(result["cases"])}, indent=2))


if __name__ == "__main__":
    main()
