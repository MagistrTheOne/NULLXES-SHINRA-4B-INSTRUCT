"""V2 UPGRADE single continuation-eval path (review-only, no auto-run side effects).

Contract: use_cache=false, document stop eos_token_id=18 (END_OF_TEXT),
wrap [BOS=1] + encode(prompt, add_special_tokens=False) + [END=18] is done by caller pack.
HF generate() KV-mask path is NOT used here (known-broken path exclusion).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONTINUATION_EOS_ID = 18
COUNTER_VERSION = "counter-v1"
EVALUATOR_VERSION = "continuation-v1"
EVAL_CONTRACT = {"use_cache": False, "eos_token_id": CONTINUATION_EOS_ID, "do_sample": False}


@torch.no_grad()
def continuation_eval(model_path: str, prompts_path: str, output_path: str, max_new_tokens: int = 128) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    model.eval()
    prompts = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
    rows = []
    per_lang: dict[str, dict[str, float]] = {}
    unk_id = tokenizer.unk_token_id
    for item in prompts:
        if not isinstance(item, dict) or "prompt" not in item or "lang" not in item:
            raise ValueError("each prompt must be {prompt, lang}; lang in {en, ru}")
        if item["lang"] not in ("en", "ru"):
            raise ValueError(f"bad lang {item['lang']!r}: expected en|ru")
        prompt = item["prompt"]
        # Single BOS, no chat template (pretrain-eval): encode adds no specials.
        body = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([[1, *body]], dtype=torch.long, device=model.device)
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=CONTINUATION_EOS_ID,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
        )
        # UNK counted ONLY among newly generated tokens (never the prompt).
        gen = out[0, input_ids.shape[1]:].tolist()
        text = tokenizer.decode(gen, skip_special_tokens=False)
        unk = sum(1 for t in gen if t == unk_id)
        rows.append({
            "prompt": prompt,
            "lang": item["lang"],
            "generated_ids": gen,
            "text": text,
            "unk_count": unk,
            "unk_rate": (unk / max(len(gen), 1)),
            "eos_id": CONTINUATION_EOS_ID,
            "use_cache": False,
            "do_sample": False,
            "counter_version": COUNTER_VERSION,
        })
    for lang in ("en", "ru"):
        sub = [r for r in rows if r["lang"] == lang]
        if sub:
            per_lang[lang] = {
                "n": float(len(sub)),
                "unk_rate": sum(r["unk_rate"] for r in sub) / len(sub),
            }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(model_path),
        "evaluator_version": EVALUATOR_VERSION,
        "contract": {**EVAL_CONTRACT, "counter_version": COUNTER_VERSION},
        "max_new_tokens": max_new_tokens,
        "rows": rows,
        "per_lang": per_lang,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"n": len(rows), "per_lang": per_lang, "output": output_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="SHINRA single continuation-eval path")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", default="evaluation/results/continuation.json")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(continuation_eval(args.model, args.prompts, args.output, args.max_new_tokens), indent=2))


if __name__ == "__main__":
    main()
