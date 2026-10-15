from __future__ import annotations

import json

from configs.loader import load_config
from data.pack import PackContractError, resolve_pretrain_special_ids, wrap_pretrain_document
from tokenizer.special_tokens import CHAT_TEMPLATE


class TokenizerFixture:
    def __init__(self, ids):
        self.ids = ids
        self.unk_token_id = 0
        self.eos_token_id = 2

    def convert_tokens_to_ids(self, token):
        return self.ids[token]


def test_special_ids_match_yaml_contract():
    cfg = load_config()
    tok = cfg["tokenizer"]
    ids = {tok["special_tokens"][k]: v for k, v in tok["special_token_ids"].items()}
    ids.update({tok["special_tokens"]["unk"]: tok["unk_id"], tok["special_tokens"]["bos"]: tok["bos_id"],
                tok["special_tokens"]["end_of_turn"]: tok["end_of_turn_id"], tok["special_tokens"]["pad"]: tok["pad_id"],
                tok["special_tokens"]["end_of_document"]: tok["end_of_document_id"]})
    assert resolve_pretrain_special_ids(TokenizerFixture(ids), cfg) == (1, 4, 3)


def test_eot_is_not_a_document_terminator():
    cfg = load_config()
    try:
        wrap_pretrain_document([42], bos_id=1, end_id=2, config=cfg)
    except PackContractError:
        pass
    else:
        raise AssertionError("EOT must not terminate a pretraining document")


def test_config_file_is_valid_yaml_and_excludes_old_corpora():
    cfg = load_config()
    excluded = " ".join(cfg["data"]["excluded_sources"]).casefold()
    assert all(name in excluded for name in ("fineweb", "smollm"))


def test_chat_template_uses_configured_tokens_without_unresolved_markers():
    cfg = load_config()
    assert "@@" not in CHAT_TEMPLATE
    for name in ("end_of_turn", "system", "user", "assistant", "reasoning", "code", "language",
                 "tool_call", "tool_response"):
        assert cfg["tokenizer"]["special_tokens"][name] in CHAT_TEMPLATE
    assert "bos_token" in CHAT_TEMPLATE
