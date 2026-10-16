from __future__ import annotations

import inspect
import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizer

from configs.loader import load_config
from data.pack import PackContractError, iter_clean_records, pack_token_stream, resolve_pretrain_special_ids, wrap_pretrain_document
from tokenizer import load_shinra_tokenizer
from tokenizer.bundle import verify_bundle
from tokenizer.contract import encode_chat, encode_text, plain_text
from training.collator import PackedParquetDataset, collate_lm


@pytest.fixture(scope="module")
def native_runtime():
    bundle_value = os.environ.get("SHINRA_NATIVE_BUNDLE")
    corpus_value = os.environ.get("SHINRA_TOKENIZER_CORPUS")
    packed_value = os.environ.get("SHINRA_PACKED_SMOKE_DIR")
    if not bundle_value or not corpus_value or not packed_value:
        pytest.skip("Colab integration requires bundle, real clean corpus and packed smoke output environment paths")
    bundle = Path(bundle_value)
    corpus = Path(corpus_value)
    packed = Path(packed_value)
    config = load_config()
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    claim = manifest.pop("manifest_sha256", None)
    from tokenizer.bundle import _digest
    if claim != _digest(manifest):
        raise AssertionError("real clean corpus manifest self-hash mismatch")
    manifest["manifest_sha256"] = claim
    config["tokenizer"]["artifact_sha256"] = json.loads(
        (bundle / "tokenizer_manifest.json").read_text(encoding="utf-8")
    )["manifest_sha256"]
    config["data"]["manifest_sha256"] = claim
    tokenizer = load_shinra_tokenizer(bundle, config)
    rows = []
    per_language = {"en": None, "ru": None}
    for row in iter_clean_records(corpus, "validation", manifest):
        lang = row.get("language")
        if lang in per_language and per_language[lang] is None:
            ids = encode_text(tokenizer, row["text"], config)
            if len(ids) + 2 <= config["model"]["train_sequence_length"]:
                per_language[lang] = (row, ids)
        if all(per_language.values()) and len(rows) >= 2:
            break
        rows.append(row)
    if not all(per_language.values()):
        raise AssertionError("real held-out EN and RU documents fitting the 4096-token contract are required")
    return {"bundle": bundle, "corpus": corpus, "packed": packed, "config": config,
            "manifest": manifest, "tokenizer": tokenizer, "per_language": per_language, "rows": rows}


def test_native_loader_keeps_existing_vocabulary_and_special_ids(native_runtime):
    tokenizer = native_runtime["tokenizer"]
    cfg = native_runtime["config"]["tokenizer"]
    assert tokenizer.is_fast is False
    assert len(tokenizer) == cfg["vocab_size"] == 65536
    assert native_runtime["config"]["model"]["max_position_embeddings"] == 32768
    assert native_runtime["config"]["model"]["train_sequence_length"] == 4096
    assert resolve_pretrain_special_ids(tokenizer, native_runtime["config"]) == (1, 4, 3)
    assert tokenizer.unk_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 3
    assert set(tokenizer.all_special_ids) == set(range(19))


def test_native_special_token_registration_handles_transformers_api_aliases(native_runtime):
    tokenizer = native_runtime["tokenizer"]
    token_config = native_runtime["config"]["tokenizer"]
    base_parameters = inspect.signature(PreTrainedTokenizer.add_special_tokens).parameters
    assert "replace_extra_special_tokens" in base_parameters or "replace_additional_special_tokens" in base_parameters

    expected_ids = {
        token_config["special_tokens"]["unk"]: token_config["unk_id"],
        token_config["special_tokens"]["bos"]: token_config["bos_id"],
        token_config["special_tokens"]["end_of_turn"]: token_config["end_of_turn_id"],
        token_config["special_tokens"]["pad"]: token_config["pad_id"],
        token_config["special_tokens"]["end_of_document"]: token_config["end_of_document_id"],
    }
    expected_ids.update({
        token_config["special_tokens"][name]: token_id
        for name, token_id in token_config["special_token_ids"].items()
    })
    controls = [token for token, token_id in sorted(expected_ids.items(), key=lambda item: item[1]) if token_id >= 4]

    original_size = len(tokenizer)
    assert tokenizer.add_special_tokens(
        {"additional_special_tokens": controls}, replace_additional_special_tokens=True
    ) == 0
    assert tokenizer.add_special_tokens(
        {"extra_special_tokens": controls[:1]}, replace_extra_special_tokens=False
    ) == 0
    assert set(tokenizer.all_special_ids) == set(range(19))
    assert tokenizer.add_special_tokens(
        {"extra_special_tokens": controls}, replace_extra_special_tokens=True
    ) == 0

    assert len(tokenizer) == original_size == token_config["vocab_size"] == 65536
    assert {token: tokenizer.convert_tokens_to_ids(token) for token in expected_ids} == expected_ids
    assert set(tokenizer.all_special_ids) == set(range(19))


def test_real_english_and_russian_native_roundtrip(native_runtime):
    tokenizer = native_runtime["tokenizer"]
    for language in ("en", "ru"):
        row, expected_native_ids = native_runtime["per_language"][language]
        text = plain_text(row["text"], native_runtime["config"])
        actual = encode_text(tokenizer, text, native_runtime["config"])
        assert actual == expected_native_ids
        assert tokenizer.decode(actual, skip_special_tokens=False) == text


def test_native_chat_template_has_one_bos_and_reserved_controls_fail(native_runtime):
    tokenizer = native_runtime["tokenizer"]
    cfg = native_runtime["config"]
    row, _ = native_runtime["per_language"]["en"]
    prompt = encode_chat(tokenizer, [{"role": "user", "content": row["text"]}], 4096, cfg)
    assert prompt[0] == cfg["tokenizer"]["bos_id"]
    assert prompt.count(cfg["tokenizer"]["bos_id"]) == 1
    for special in cfg["tokenizer"]["special_tokens"].values():
        with pytest.raises(ValueError, match="reserved control"):
            encode_text(tokenizer, f"quoted {special} content", cfg)


def test_native_tokenizer_refuses_implicit_truncation(native_runtime):
    tokenizer = native_runtime["tokenizer"]
    with pytest.raises(ValueError, match="never truncates"):
        tokenizer.encode("ordinary text", truncation=True)
    with pytest.raises(ValueError, match="never truncates"):
        tokenizer("ordinary text", truncation=True)


def test_native_bundle_is_bound_to_every_file_and_self_hash(native_runtime, tmp_path):
    bundle = native_runtime["bundle"]
    config = native_runtime["config"]
    digest = verify_bundle(bundle, config, backend="sentencepiece_native")
    manifest = json.loads((bundle / "tokenizer_manifest.json").read_text(encoding="utf-8"))
    assert digest == config["tokenizer"]["artifact_sha256"]
    assert set(manifest["files"]) == {
        "tokenizer.model", "tokenizer.vocab", "tokenizer_config.json", "special_tokens_map.json",
        "chat_template.jinja", "tokenizer_stats.json", "corpus_stats.json",
    }
    tampered = tmp_path / "tampered-native-bundle"
    shutil.copytree(bundle, tampered)
    with (tampered / "tokenizer.vocab").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_bundle(tampered, config, backend="sentencepiece_native")
    corrupt_manifest = tmp_path / "corrupt-manifest-native-bundle"
    shutil.copytree(bundle, corrupt_manifest)
    bad_manifest_path = corrupt_manifest / "tokenizer_manifest.json"
    bad_manifest = json.loads(bad_manifest_path.read_text(encoding="utf-8"))
    bad_manifest["manifest_sha256"] = "0" * 64
    bad_manifest_path.write_text(json.dumps(bad_manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest identity"):
        verify_bundle(corrupt_manifest, config, backend="sentencepiece_native")


def test_real_documents_pack_with_bos_eod_pad_loss_and_isolation(native_runtime):
    cfg = native_runtime["config"]
    body_a = native_runtime["per_language"]["en"][1]
    body_b = native_runtime["per_language"]["ru"][1]
    rows = pack_token_stream([
        wrap_pretrain_document(body_a, config=cfg),
        wrap_pretrain_document(body_b, config=cfg),
    ], 4096, config=cfg)
    assert len(rows) == 2
    assert rows[0]["input_ids"][0] == rows[1]["input_ids"][0] == 1
    assert rows[0]["labels"][0] == rows[1]["labels"][0] == -100
    for packed in rows:
        active = sum(packed["attention_mask"])
        assert packed["input_ids"][active - 1] == 4
        assert packed["labels"][active - 1] == 4
        assert all(label == -100 for label in packed["labels"][active:])
        assert all(token == 3 for token in packed["input_ids"][active:])


def test_real_overlength_document_is_rejected_without_truncation(native_runtime):
    tokenizer = native_runtime["tokenizer"]
    cfg = native_runtime["config"]
    manifest = native_runtime["manifest"]
    for row in iter_clean_records(native_runtime["corpus"], "validation", manifest):
        body = encode_text(tokenizer, row["text"], cfg)
        if len(body) + 2 > 4096:
            wrapped = wrap_pretrain_document(body, config=cfg)
            with pytest.raises(PackContractError, match="exceeds sequence_length"):
                pack_token_stream([wrapped], 4096, config=cfg)
            return
    pytest.skip("clean validation split contains no overlength document to exercise this rejection")


def test_packed_parquet_smoke_loads_real_shards_and_collates(native_runtime):
    dataset = PackedParquetDataset(native_runtime["packed"], native_runtime["config"], "validation",
                                   allow_smoke_test=True)
    assert len(dataset) > 0
    batch = collate_lm([dataset[0]])
    assert batch["input_ids"].shape == (1, 4096)
    assert batch["labels"].shape == (1, 4096)
    assert torch.all(batch["labels"][0, 0] == -100)


def test_packed_loader_rejects_corpus_or_bundle_identity_mismatch(native_runtime):
    cfg = native_runtime["config"]
    wrong_corpus = dict(cfg)
    wrong_corpus["data"] = dict(cfg["data"], manifest_sha256="f" * 64)
    with pytest.raises(ValueError, match="frozen YAML identity"):
        PackedParquetDataset(native_runtime["packed"], wrong_corpus, "validation", allow_smoke_test=True)
    wrong_bundle = dict(cfg)
    wrong_bundle["tokenizer"] = dict(cfg["tokenizer"], artifact_sha256="0" * 64)
    with pytest.raises(ValueError, match="tokenizer bundle mismatch"):
        PackedParquetDataset(native_runtime["packed"], wrong_bundle, "validation", allow_smoke_test=True)


def test_smoke_shards_are_not_admitted_to_production_loader(native_runtime):
    with pytest.raises(ValueError, match="cannot be used for training or evaluation"):
        PackedParquetDataset(native_runtime["packed"], native_runtime["config"], "validation")
