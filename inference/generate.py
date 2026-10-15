"""SHINRA generation entrypoint (Hugging Face generate + chat template)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from configs.loader import load_config
from model.modeling_shinra import ShinraForCausalLM
from tokenizer import load_shinra_tokenizer


def load_generate_stack(model_path: str):
    approved = load_config()
    checkpoint = Path(model_path)
    if not checkpoint.is_dir():
        raise FileNotFoundError("inference requires a local SHINRA checkpoint directory")
    for name in ("config.json", "tokenizer_config.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"checkpoint is missing {name}: {checkpoint}")
    tokenizer = load_shinra_tokenizer(checkpoint)
    expected = approved["tokenizer"]
    if len(tokenizer) != expected["vocab_size"]:
        raise ValueError("checkpoint tokenizer vocabulary differs from configs/shinra.yaml")
    for config_key, attr in (("unk_id", "unk_token_id"), ("bos_id", "bos_token_id"),
                             ("end_of_turn_id", "eos_token_id"), ("pad_id", "pad_token_id")):
        if getattr(tokenizer, attr) != expected[config_key]:
            raise ValueError(f"checkpoint tokenizer {attr} differs from configs/shinra.yaml")
    model = ShinraForCausalLM.from_pretrained(
        str(checkpoint),
        dtype=getattr(torch, approved["precision"]["preferred"]),
        local_files_only=True,
        device_map="auto",
    )
    if model.config.model_type != approved["model"]["model_type"]:
        raise ValueError("checkpoint model_type differs from configs/shinra.yaml")
    if model.config.vocab_size != approved["model"]["vocab_size"]:
        raise ValueError("checkpoint model vocabulary differs from configs/shinra.yaml")
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "max_position_embeddings", "bos_token_id",
                "eos_token_id", "pad_token_id", "document_end_token_id"):
        if getattr(model.config, key) != approved["model"][key]:
            raise ValueError(f"checkpoint model config {key} differs from configs/shinra.yaml")
    model.eval()
    return tokenizer, model


_INFERENCE_DEFAULTS = load_config()["inference"]


@torch.no_grad()
def generate_chat(
    model_path: str,
    messages: list[dict[str, str]],
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    tokenizer=None,
    model=None,
) -> str:
    max_new_tokens = _INFERENCE_DEFAULTS["default_max_new_tokens"] if max_new_tokens is None else max_new_tokens
    temperature = _INFERENCE_DEFAULTS["default_temperature"] if temperature is None else temperature
    top_p = _INFERENCE_DEFAULTS["default_top_p"] if top_p is None else top_p
    if tokenizer is None or model is None:
        tokenizer, model = load_generate_stack(model_path)

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0 and top_p != 1:
        raise ValueError("top_p is only applied when temperature is greater than zero")

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    context_limit = int(
        getattr(model.config, "max_position_embeddings", tokenizer.model_max_length)
    )
    if max_new_tokens >= context_limit:
        raise ValueError("max_new_tokens must be smaller than the configured context window")
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
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    out = model.generate(**inputs, **generation_kwargs)
    generated = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    inference_cfg = load_config()["inference"]
    parser = argparse.ArgumentParser(description="Generate with SHINRA")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=inference_cfg["default_max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=inference_cfg["default_temperature"])
    parser.add_argument("--top-p", type=float, default=inference_cfg["default_top_p"])
    args = parser.parse_args()

    tokenizer, model = load_generate_stack(args.model)
    messages = [{"role": "user", "content": args.prompt}]
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
