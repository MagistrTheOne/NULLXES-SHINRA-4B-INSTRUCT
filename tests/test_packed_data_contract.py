from __future__ import annotations

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import pytest

from configs.loader import load_config
from data.pack import _row, pack_token_stream
from data.packed_data import count_target_tokens_post_shift
from training.collator import PackedParquetDataset
from training.sampler import StatefulRandomSampler


def test_optimizer_token_counter_counts_post_shift_targets():
    labels = torch.tensor([[1, 8, 9, -100]])
    assert count_target_tokens_post_shift(labels) == 2


def test_stateful_sampler_restores_exact_next_index():
    dataset = list(range(20))
    sampler = StatefulRandomSampler(dataset, seed=5)
    iterator = iter(sampler)
    first = [next(iterator) for _ in range(7)]
    state = sampler.state_dict()
    expected_next = next(iterator)
    resumed = StatefulRandomSampler(dataset, seed=999)
    resumed.load_state_dict(state)
    assert list(iter(resumed))[0] == expected_next
    assert len(set(first)) == len(first)


def test_isolated_document_rows_mask_pad_and_cross_pad_targets():
    config = load_config()
    rows = pack_token_stream([[1, 42, 4], [1, 43, 4]], 6, config=config)
    assert len(rows) == 2
    assert rows[0]["input_ids"] == [1, 42, 4, 3, 3, 3]
    assert rows[0]["labels"] == [-100, 42, 4, -100, -100, -100]
    assert rows[0]["attention_mask"] == [1, 1, 1, 0, 0, 0]


def test_single_token_pretrain_sequence_has_no_shifted_target():
    row = _row([4, 3, 3], 3, load_config())
    assert count_target_tokens_post_shift(torch.tensor([row["labels"]])) == 0


def _write_packed_fixture(path):
    path.mkdir()
    rows = [{"input_ids": [1, 42, 4, 3], "attention_mask": [1, 1, 1, 0],
             "labels": [1, 42, 4, -100], "source_manifest_sha256": "a" * 64}]
    shard = path / "packed-00000.parquet"
    pq.write_table(pa.Table.from_pylist(rows), shard)
    manifest = {"schema": "nullxes.shinra.packed.v1", "source_manifest_sha256": "a" * 64,
                "split": "train", "shard_files": [shard.name],
                "shard_sha256": {shard.name: hashlib.sha256(shard.read_bytes()).hexdigest()}}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (path / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_parquet_loader_checks_shard_hash_and_reads_sequence(tmp_path):
    packed = tmp_path / "train"
    _write_packed_fixture(packed)
    dataset = PackedParquetDataset(packed)
    assert len(dataset) == 1
    assert dataset[0]["input_ids"].tolist() == [1, 42, 4, 3]
    assert dataset.source_manifest_sha256 == "a" * 64


def test_parquet_loader_rejects_a_shard_missing_from_manifest(tmp_path):
    packed = tmp_path / "train"
    _write_packed_fixture(packed)
    manifest_path = packed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    body = dict(manifest)
    body.pop("manifest_sha256")
    body["shard_files"].append("missing.parquet")
    body["shard_sha256"]["missing.parquet"] = "b" * 64
    body["manifest_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in body.items() if k != "manifest_sha256"},
                                                         sort_keys=True).encode()).hexdigest()
    manifest_path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="shard set differs"):
        PackedParquetDataset(packed)
