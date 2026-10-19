from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data.language_engine import ENGINE_VERSION, iter_records, write_corpus


def test_tokenizer_language_records_are_reproducible_bilingual_and_unique(tmp_path):
    first = list(iter_records(20261019, 4))
    second = list(iter_records(20261019, 4))
    assert first == second
    assert {row["language"] for row in first} == {"en", "ru"}
    assert all(row["generator_version"] == ENGINE_VERSION for row in first)
    assert all(row["purpose"] == "tokenizer_training_only" for row in first)
    assert len({row["text_sha256"] for row in first}) == len(first)
    assert all(row["text_sha256"] == hashlib.sha256(row["text"].encode()).hexdigest() for row in first)
    assert any("controller" in row["text"] for row in first if row["language"] == "en")
    assert any("Регулятор" in row["text"] for row in first if row["language"] == "ru")


def test_tokenizer_language_artifact_is_stream_written_and_reports_measured_stats(tmp_path):
    artifact = tmp_path / "tokenizer_train.jsonl"
    first = write_corpus(artifact, root_seed=71, records_per_concept=3, max_sentence_bytes=65536, workers=1)
    first_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
    second_artifact = tmp_path / "second.jsonl"
    second = write_corpus(second_artifact, root_seed=71, records_per_concept=3,
                          max_sentence_bytes=65536, workers=1)
    assert first == second
    assert first_sha == hashlib.sha256(second_artifact.read_bytes()).hexdigest()
    assert len(rows) == first["record_count"]
    assert first["exact_duplicate_text_records"] == 0
    assert set(first["records_by_language"]) == {"en", "ru"}
    assert first["records_by_language"]["en"] == first["records_by_language"]["ru"]
    assert not (tmp_path / "train.jsonl").exists()


def test_sentencepiece_reads_tokenizer_only_file_not_model_training_splits(tmp_path, monkeypatch):
    import tokenizer.train_tokenizer as train_tokenizer

    tokenizer_path = tmp_path / "tokenizer_train.jsonl"
    tokenizer_path.write_text(json.dumps({"text": "NULLXES-authored language text"}) + "\n", encoding="utf-8")
    (tmp_path / "train.jsonl").write_text(json.dumps({"text": "model-only task record"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(train_tokenizer, "plain_text", lambda text, _config: text)
    manifest = {"tokenizer_language_corpus": {"file": tokenizer_path.name, "file_sha256": "a" * 64}}
    stats = {}
    sentences = list(train_tokenizer.training_sentences(tmp_path, manifest, {}, 1000, stats))
    assert sentences == ["NULLXES-authored language text"]
    assert stats["training_source"] == "tokenizer-only-language-corpus"


def test_sentencepiece_configuration_keeps_hard_limit_and_propagates_vocab_failure(monkeypatch):
    from types import SimpleNamespace
    import sys

    import tokenizer.train_tokenizer as train_tokenizer

    observed = {}

    def reject(**kwargs):
        observed.update(kwargs)
        raise RuntimeError("Vocabulary size too high; maximum is 19075")

    monkeypatch.setitem(sys.modules, "sentencepiece", SimpleNamespace(
        SentencePieceTrainer=SimpleNamespace(train=reject)
    ))
    monkeypatch.setattr(train_tokenizer, "training_sentences", lambda *args: iter(("authored text",)))
    monkeypatch.setattr(train_tokenizer, "special_ids", lambda _config: {})
    config = {"tokenizer": {
        "special_tokens": {}, "vocab_size": 65536, "algorithm": "unigram",
        "character_coverage": 0.99995, "byte_fallback": True, "split_digits": False,
        "normalization": "identity", "unk_id": 0, "bos_id": 1,
        "end_of_turn_id": 2, "pad_id": 3, "build": {"max_sentence_bytes": 65536},
    }}
    args = SimpleNamespace(max_chars=1000, num_threads=1, input_sentence_size=101,
                           max_sentence_bytes=65536)
    with pytest.raises(RuntimeError, match="maximum is 19075"):
        train_tokenizer.train_sentencepiece(None, {}, Path("unused"), config, args, {})
    assert observed["vocab_size"] == 65536
    assert observed["model_type"] == "unigram"
    assert observed["hard_vocab_limit"] is True
    assert observed["byte_fallback"] is True
