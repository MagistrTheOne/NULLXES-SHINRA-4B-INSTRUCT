"""Canonical control table and rendered template, derived from the single YAML."""
from __future__ import annotations

import json
from pathlib import Path

from configs.loader import load_config
from .contract import special_ids


def render_chat_template(config: dict) -> str:
    template = Path(__file__).with_name("chat_template.jinja").read_text(encoding="utf-8")
    for name, token in config["tokenizer"]["special_tokens"].items():
        template = template.replace(f"@@{name}@@", token)
    return template.replace("@@reserved_tokens@@", json.dumps(list(special_ids(config))))


_CONFIG = load_config()
_ORDERED = sorted(special_ids(_CONFIG).items(), key=lambda item: item[1])
CORE_SPECIALS = tuple(token for token, token_id in _ORDERED if token_id < _CONFIG["tokenizer"]["end_of_document_id"])
USER_DEFINED_SYMBOLS = tuple(token for token, token_id in _ORDERED if token_id >= _CONFIG["tokenizer"]["end_of_document_id"])
ALL_SPECIAL_TOKENS = CORE_SPECIALS + USER_DEFINED_SYMBOLS
CHAT_TEMPLATE = render_chat_template(_CONFIG)
