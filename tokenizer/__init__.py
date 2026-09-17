"""Load the SHINRA tokenizer from a trained artifact directory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


def load_shinra_tokenizer(path: str | Path) -> PreTrainedTokenizerFast:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer
