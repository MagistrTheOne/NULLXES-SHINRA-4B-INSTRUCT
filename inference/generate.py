"""SHINRA generation entrypoint (Hugging Face generate + chat template)."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_generate_stack(model_path: str, dtype: str = "bfloat16"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, dtype),
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate_chat(
    model_path: str,
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    tokenizer=None,
    model=None,
) -> str:
    if tokenizer is None or model is None:
        tokenizer, model = load_generate_stack(model_path)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-5) if do_sample else None,
        top_p=top_p if do_sample else None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with SHINRA")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default="You are SHINRA, the NULLXES language intelligence layer.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    print(generate_chat(args.model, messages, max_new_tokens=args.max_new_tokens, temperature=args.temperature))


if __name__ == "__main__":
    main()
