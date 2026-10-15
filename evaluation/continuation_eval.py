"""Deterministic continuation evaluation over a caller-supplied held-out prompt set.

Document termination uses the configured end-of-document token; generation turn-stop remains EOT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from configs.loader import load_config
from inference.generate import load_generate_stack
from tokenizer.contract import encode_text, plain_text, suppressed_controls

_APPROVED = load_config()
CONTINUATION_EOS_ID = int(_APPROVED["tokenizer"]["end_of_document_id"])
COUNTER_VERSION = "counter-v1"
EVALUATOR_VERSION = "continuation-v1"
EVAL_CONTRACT = {"use_cache": False, "eos_token_id": CONTINUATION_EOS_ID, "do_sample": False}


@torch.no_grad()
def continuation_eval(model_path: str, prompts_path: str, output_path: str,
                      max_new_tokens: int | None = None) -> dict:
    max_new_tokens = int(_APPROVED["evaluation"]["continuation_max_new_tokens"] if max_new_tokens is None
                          else max_new_tokens)
    tokenizer, model = load_generate_stack(model_path)
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
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not 0 < max_new_tokens < int(model.config.max_position_embeddings):
            raise ValueError("max_new_tokens must be positive and below the model context limit")
        # Single BOS, no chat template (pretrain-eval): encode adds no specials.
        prompt = plain_text(prompt, _APPROVED)
        body = encode_text(tokenizer, prompt, _APPROVED)
        if 1 + len(body) + max_new_tokens > int(model.config.max_position_embeddings):
            raise ValueError("continuation exceeds context budget; implicit truncation is forbidden")
        input_device = model.get_input_embeddings().weight.device
        input_ids = torch.tensor([[int(tokenizer.bos_token_id), *body]], dtype=torch.long, device=input_device)
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=CONTINUATION_EOS_ID,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False,
            suppress_tokens=suppressed_controls(_APPROVED, CONTINUATION_EOS_ID),
            forced_bos_token_id=None,
            forced_eos_token_id=None,
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
    parser.add_argument("--max-new-tokens", type=int, default=_APPROVED["evaluation"]["continuation_max_new_tokens"])
    args = parser.parse_args()
    print(json.dumps(continuation_eval(args.model, args.prompts, args.output, args.max_new_tokens), indent=2))


if __name__ == "__main__":
    main()
