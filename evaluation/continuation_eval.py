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


@torch.no_grad()
def continuation_eval(model_path: str, prompts_path: str, output_path: str, max_new_tokens: int = 128) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    )
    model.eval()
    prompts = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
    rows = []
    unk_id = tokenizer.unk_token_id
    for item in prompts:
        prompt = item.get("prompt", "") if isinstance(item, dict) else str(item)
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
        gen = out[0, input_ids.shape[1]:].tolist()
        text = tokenizer.decode(gen, skip_special_tokens=False)
        rows.append({
            "prompt": prompt,
            "generated_ids": gen,
            "text": text,
            "unk_count": sum(1 for t in gen if t == unk_id),
            "eos_id": CONTINUATION_EOS_ID,
            "use_cache": False,
            "counter_version": COUNTER_VERSION,
        })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"n": len(rows), "output": output_path}


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
