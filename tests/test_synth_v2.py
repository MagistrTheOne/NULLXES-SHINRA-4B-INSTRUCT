"""Deterministic SHINRA v2 synth fixtures. No 50M emit. CPU only."""

from __future__ import annotations

import hashlib
import json

import pytest

from data.synth.contamination import ContaminationError, assert_clean_pretrain_text
from data.synth.generate import generate_record, iter_records, records_to_jsonl, write_corpus
from data.synth.schema import CORPUS_ID, LAYERS, LICENSE, MANIFEST_FIELDS, MAX_RECORDS, validate_manifest


def test_same_seed_same_sha():
    a = records_to_jsonl(iter_records(7, 4))
    b = records_to_jsonl(iter_records(7, 4))
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_different_seed_different_sha():
    a = records_to_jsonl(iter_records(7, 4))
    b = records_to_jsonl(iter_records(8, 4))
    assert hashlib.sha256(a).hexdigest() != hashlib.sha256(b).hexdigest()


def test_layers_languages_and_hidden_semantics(tmp_path):
    records = iter_records(0, 6)
    assert len(records) == 30
    layers = {r["layer"] for r in records}
    langs = {r["language"] for r in records}
    assert layers == set(LAYERS)
    assert langs == {"en", "ru"}
    for rec in records:
        assert rec["semantics"]
        assert json.dumps(rec["semantics"], ensure_ascii=False, sort_keys=True) not in rec["text"]
        assert_clean_pretrain_text(rec["text"])
        assert "<|" not in rec["text"]
        assert "I am SHINRA" not in rec["text"]
        assert "я шинра" not in rec["text"].lower()
    en = [r for r in records if r["language"] == "en"]
    ru = [r for r in records if r["language"] == "ru"]
    assert len(en) == len(ru)
    index = write_corpus(tmp_path, seed=0, per_layer=6)
    assert index["corpus_id"] == CORPUS_ID
    assert index["license"] == LICENSE
    assert len(index["shards"]) == 10
    for shard in index["shards"]:
        blob = (tmp_path / shard["path"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == shard["sha256"]
        man = json.loads((tmp_path / shard["family"] / f"{shard['language']}.manifest.json").read_text(encoding="utf-8"))
        validate_manifest(man)
        assert list(man) == sorted(man)
        assert set(man) == set(MANIFEST_FIELDS)


def test_contamination_rejects_chat_and_identity():
    with pytest.raises(ContaminationError):
        assert_clean_pretrain_text("Hello <|user|> there")
    with pytest.raises(ContaminationError):
        assert_clean_pretrain_text("I am SHINRA and I help.")
    with pytest.raises(ContaminationError):
        assert_clean_pretrain_text("User: do a thing")


def test_cap_refuses_50m_scale():
    with pytest.raises(ValueError, match="cap"):
        iter_records(0, MAX_RECORDS + 2)


def test_knowledge_mentions_invented_entities():
    texts = [generate_record(1, "knowledge_shaped", "en", i)["text"] for i in range(40)]
    blob = " ".join(texts)
    assert "Lerna-7" in blob or "Varek" in blob or "K-12" in blob
    assert "Wikipedia" not in blob
