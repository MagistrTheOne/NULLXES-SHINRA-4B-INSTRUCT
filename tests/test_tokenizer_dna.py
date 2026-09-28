"""Frozen tokenizer DNA. No Hub download, no trained artifacts required."""

from __future__ import annotations

from tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    BOS,
    CHAT_TEMPLATE,
    END_OF_TEXT,
    EOT,
    PAD,
    UNK,
)


def test_control_plane_prefix_and_document_stop():
    assert ALL_SPECIAL_TOKENS[:4] == [UNK, BOS, EOT, PAD]
    assert END_OF_TEXT in ALL_SPECIAL_TOKENS
    assert "<|end|>" not in ALL_SPECIAL_TOKENS
    assert "<|tool|>" not in ALL_SPECIAL_TOKENS
    assert "<|im_end|>" not in ALL_SPECIAL_TOKENS


def test_chat_template_emits_bos_and_eot():
    assert CHAT_TEMPLATE.lstrip().startswith("{{- bos_token -}}")
    assert "<|eot|>" in CHAT_TEMPLATE
    assert "<|end_of_text|>" not in CHAT_TEMPLATE
