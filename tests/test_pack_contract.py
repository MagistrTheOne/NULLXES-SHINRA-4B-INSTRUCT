"""SHINRA v2 pack wrap. No Hub tokenizer required."""

from __future__ import annotations

import pytest

from data.pack import (
    PACKER_VERSION,
    PackContractError,
    assert_packed_ids_clean,
    assert_pretrain_body_clean,
    pack_token_stream,
    resolve_pretrain_special_ids,
    wrap_pretrain_document,
)
from tokenizer.special_tokens import PRETRAIN_FORBIDDEN_IDS


class _Tok:
    unk_token_id = 0
    eos_token_id = 2

    def __init__(self, mapping: dict[str, int]):
        self.mapping = mapping

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.mapping[token]


def test_packer_version():
    assert PACKER_VERSION == "shinra-v2-pack.v1"


def test_wrap_is_bos_body_end_of_text():
    wrapped = wrap_pretrain_document([19, 20, 21])
    assert wrapped[0] == 1
    assert wrapped[-1] == 18
    assert wrapped[1:-1] == [19, 20, 21]
    assert 17 not in wrapped


def test_wrap_rejects_eot_as_document_end():
    with pytest.raises(PackContractError):
        wrap_pretrain_document([19, 20, 21], end_id=2)
    with pytest.raises(PackContractError):
        wrap_pretrain_document([19, 20, 21], end_id=1)


def test_wrap_rejects_eot_chat_document_and_wrap_ids_in_body():
    with pytest.raises(PackContractError):
        wrap_pretrain_document([19, 2, 20])
    with pytest.raises(PackContractError):
        wrap_pretrain_document([5, 19])
    with pytest.raises(PackContractError):
        wrap_pretrain_document([17, 19])
    with pytest.raises(PackContractError):
        wrap_pretrain_document([1, 19])
    with pytest.raises(PackContractError):
        wrap_pretrain_document([18, 19])
    with pytest.raises(PackContractError):
        wrap_pretrain_document([3, 19])
    with pytest.raises(PackContractError):
        wrap_pretrain_document([])


def test_resolve_rejects_eot_fallback():
    tok = _Tok({"<|bos|>": 1, "<|pad|>": 3, "<|end_of_text|>": 2})
    tok.eos_token_id = 2
    with pytest.raises(PackContractError):
        resolve_pretrain_special_ids(tok)


def test_resolve_accepts_frozen_ids():
    tok = _Tok({"<|bos|>": 1, "<|pad|>": 3, "<|end_of_text|>": 18})
    assert resolve_pretrain_special_ids(tok) == (1, 18, 3)


def test_forbidden_set_excludes_end_of_text_and_bos():
    assert 1 not in PRETRAIN_FORBIDDEN_IDS
    assert 3 not in PRETRAIN_FORBIDDEN_IDS
    assert 18 not in PRETRAIN_FORBIDDEN_IDS
    assert 2 in PRETRAIN_FORBIDDEN_IDS
    assert 5 in PRETRAIN_FORBIDDEN_IDS
    assert 17 in PRETRAIN_FORBIDDEN_IDS
    assert_pretrain_body_clean([19, 40])
    assert_packed_ids_clean([1, 19, 18, 3])
    with pytest.raises(PackContractError):
        assert_packed_ids_clean([1, 2, 19, 18])


def test_concat_pack_pads_labels_and_stays_uncontaminated():
    d1 = wrap_pretrain_document([19, 20, 21])
    d2 = wrap_pretrain_document([22, 23])
    rows = pack_token_stream([d1, d2], sequence_length=8, pad_id=3)
    assert len(rows) == 2
    first, last = rows
    assert first["input_ids"] == [1, 19, 20, 21, 18, 1, 22, 23]
    assert first["labels"] == first["input_ids"]
    assert first["attention_mask"] == [1] * 8
    assert last["input_ids"][0] == 18
    assert last["input_ids"][1:] == [3] * 7
    assert last["labels"][0] == 18
    assert last["labels"][1:] == [-100] * 7
    assert last["attention_mask"] == [1] + [0] * 7
    for row in rows:
        assert 2 not in row["input_ids"]
        assert not PRETRAIN_FORBIDDEN_IDS.intersection(row["input_ids"])
        assert 17 not in row["input_ids"]
