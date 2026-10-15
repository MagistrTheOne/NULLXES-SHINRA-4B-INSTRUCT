"""Shared text/control contract. No model or ML imports."""
from __future__ import annotations

import unicodedata

from configs.loader import load_config


def normalize_text(text: str, normalization: str = "NFC") -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if normalization != "NFC":
        raise ValueError("Unicode normalization must be NFC; SentencePiece identity is separate")
    return unicodedata.normalize(normalization, text)


def special_ids(config: dict) -> dict[str, int]:
    tok = config["tokenizer"]
    ids = dict(tok["special_token_ids"])
    ids.update({key: tok[f"{key}_id"] for key in ("unk", "bos", "end_of_turn", "pad", "end_of_document")})
    return {tok["special_tokens"][key]: int(value) for key, value in ids.items()}


def plain_text(text: str, config: dict | None = None) -> str:
    config = config or load_config()
    text = normalize_text(text, config["data"]["cleaning"]["unicode_normalization"])
    if any(token in text for token in special_ids(config)):
        raise ValueError("literal reserved control strings are forbidden in ordinary text")
    if "▁" in text:
        raise ValueError("literal U+2581 is reserved by SentencePiece Metaspace; lossless encoding is unavailable")
    return text


def encode_text(tokenizer, text: str, config: dict | None = None) -> list[int]:
    config = config or load_config()
    text = plain_text(text, config)
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    forbidden = set(special_ids(config).values())
    if any(i in forbidden or not 0 <= i < config["tokenizer"]["vocab_size"] for i in ids):
        raise ValueError("ordinary text encoded to UNK, a control token or an out-of-range ID")
    if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
        raise ValueError("tokenizer round-trip differs from NFC-normalized input")
    return ids


def encode_chat(tokenizer, messages: list[dict], max_tokens: int, config: dict | None = None) -> list[int]:
    config = config or load_config()
    if not messages:
        raise ValueError("chat requires at least one message")
    normalized = []
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"} or message.get("tool_calls") or message.get("reasoning"):
            raise ValueError("tool calls and reasoning channels are not implemented")
        content = plain_text(message.get("content"), config)
        encode_text(tokenizer, content, config)
        normalized.append({"role": message["role"], "content": content})
    prompt = tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
    bos = int(config["tokenizer"]["bos_id"])
    if not ids or ids[0] != bos or ids.count(bos) != 1:
        raise ValueError("chat must contain exactly one leading BOS")
    if len(ids) > max_tokens:
        raise ValueError("chat exceeds context budget; implicit truncation is forbidden")
    if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != prompt:
        raise ValueError("chat encoding changed message boundaries or content")
    return ids


def suppressed_controls(config: dict, stop_id: int) -> list[int]:
    return sorted(set(special_ids(config).values()) - {stop_id})
