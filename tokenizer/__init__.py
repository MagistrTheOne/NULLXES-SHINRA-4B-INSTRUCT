"""Load the SHINRA tokenizer from a trained artifact directory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

from configs.loader import load_config


def load_shinra_tokenizer(path: str | Path) -> PreTrainedTokenizerFast:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
    config = load_config()["tokenizer"]
    if len(tokenizer) != int(config["vocab_size"]):
        raise ValueError("tokenizer vocabulary size differs from configs/shinra.yaml")
    expected = {config["special_tokens"][key]: int(value)
                for key, value in config["special_token_ids"].items()}
    expected.update({config["special_tokens"]["unk"]: int(config["unk_id"]),
                     config["special_tokens"]["bos"]: int(config["bos_id"]),
                     config["special_tokens"]["end_of_turn"]: int(config["end_of_turn_id"]),
                     config["special_tokens"]["pad"]: int(config["pad_id"]),
                     config["special_tokens"]["end_of_document"]: int(config["end_of_document_id"])})
    actual = {token: tokenizer.convert_tokens_to_ids(token) for token in expected}
    if actual != expected:
        raise ValueError(f"tokenizer special token IDs mismatch: {actual} != {expected}")
    if tokenizer.eos_token_id != config["end_of_turn_id"]:
        raise ValueError("tokenizer eos_token_id must remain the configured EOT ID")
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer
