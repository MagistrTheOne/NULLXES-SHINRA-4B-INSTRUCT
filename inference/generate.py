"""SHINRA generation entrypoint (Hugging Face generate + chat template)."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    if not hasattr(torch, dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return getattr(torch, dtype)


def load_generate_stack(model_path: str, dtype: str = "bfloat16"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=_resolve_dtype(dtype),
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

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    context_limit = int(
        getattr(model.config, "max_position_embeddings", tokenizer.model_max_length)
    )
    max_input_tokens = max(1, context_limit - max_new_tokens)

    previous_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
    finally:
        tokenizer.truncation_side = previous_truncation_side

    input_device = model.get_input_embeddings().weight.device
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    out = model.generate(**inputs, **generation_kwargs)
    generated = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with SHINRA")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--system",
        default="You are SHINRA, a language model developed by NULLXES.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    tokenizer, model = load_generate_stack(args.model, dtype=args.dtype)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    print(
        generate_chat(
            args.model,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tokenizer=tokenizer,
            model=model,
        )
    )


if __name__ == "__main__":
    main()
