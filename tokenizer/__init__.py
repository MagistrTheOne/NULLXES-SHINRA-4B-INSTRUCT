"""Load the SHINRA tokenizer from a trained artifact directory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

from configs.loader import load_config


def load_shinra_tokenizer(path: str | Path, config: dict | None = None) -> PreTrainedTokenizerFast:
    from .bundle import verify_bundle

    approved = config or load_config()
    bundle_sha = verify_bundle(path, approved)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
    config = approved["tokenizer"]
    if bool(getattr(tokenizer, "add_bos_token", False)) != bool(config["add_bos_in_tokenizer"]):
        raise ValueError("tokenizer implicit BOS behavior differs from configs/shinra.yaml")
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
    if set(tokenizer.all_special_ids) != set(expected.values()):
        raise ValueError("all fixed control IDs must be registered as special tokens")
    from .special_tokens import render_chat_template
    if tokenizer.chat_template != render_chat_template(approved):
        raise ValueError("tokenizer chat template differs from the current contract")
    if tokenizer.model_max_length != approved["model"]["max_position_embeddings"]:
        raise ValueError("tokenizer context limit differs from configs/shinra.yaml")
    if tokenizer.backend_tokenizer.post_processor is not None:
        raise ValueError("tokenizer post-processing is forbidden: BOS is inserted explicitly")
    if tokenizer.backend_tokenizer.truncation is not None:
        raise ValueError("implicit tokenizer truncation is forbidden")
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.shinra_bundle_path = Path(path).resolve()
    tokenizer.shinra_bundle_sha256 = bundle_sha
    tokenizer.shinra_config = approved
    return tokenizer
