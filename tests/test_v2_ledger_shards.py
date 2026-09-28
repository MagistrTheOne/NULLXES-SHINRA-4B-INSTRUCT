"""Rolling shards, ledger SHA, disk ceiling, streaming synth."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from data.ledger import RunLedger
from data.shard_lifecycle import DiskCeilingError, assert_under_ceiling, build_packed_shard
from data.shards import PackedBinDataset, count_label_tokens, verify_shard
from data.synth.stream import iter_stream_records
from tokenizer.special_tokens import PRETRAIN_FORBIDDEN_IDS


def _encode(text: str) -> list[int]:
    ids = []
    for word in text.split():
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        ids.append(19 + int.from_bytes(digest[:2], "big") % 40)
    return ids or [19]


def test_same_seed_same_shard_sha(tmp_path: Path):
    kwargs = dict(
        stage="s0",
        shard_index=0,
        seed=3,
        record_count=24,
        encode=_encode,
        sequence_length=32,
        start_index=0,
    )
    a = tmp_path / "a"
    b = tmp_path / "b"
    ma = build_packed_shard(output_dir=a, **kwargs)
    mb = build_packed_shard(output_dir=b, **kwargs)
    assert ma["sha256"] == mb["sha256"]
    verify_shard(a / "shard-00000.bin")


def test_stream_not_capped_at_20k():
    recs = list(iter_stream_records("s0", seed=1, count=80, start_index=0))
    assert len(recs) == 80
    langs = {r["language"] for r in recs}
    assert langs == {"en", "ru"}


def test_packed_bin_forbids_eot_and_counts_labels(tmp_path: Path):
    meta = build_packed_shard(
        stage="s0",
        shard_index=0,
        seed=0,
        record_count=20,
        encode=_encode,
        output_dir=tmp_path,
        sequence_length=32,
    )
    ds = PackedBinDataset(tmp_path)
    assert len(ds) == meta["n_sequences"]
    item = ds[0]
    ids = item["input_ids"].tolist()
    assert not PRETRAIN_FORBIDDEN_IDS.intersection(ids)
    n = count_label_tokens(item["labels"])
    assert n == int((item["labels"] != -100).sum())
    assert item["input_ids"][0] == 1 or 1 in ids


def test_disk_ceiling(tmp_path: Path):
    (tmp_path / "blob.bin").write_bytes(b"x" * 2048)
    with pytest.raises(DiskCeilingError):
        assert_under_ceiling(tmp_path, ceiling_gb=1e-9)


def test_ledger_status(tmp_path: Path):
    led = RunLedger(tmp_path)
    led.append({"event": "shard_ready", "produced_tokens": 10})
    led.log_metrics({"loss": 1.2, "lr": 3e-4, "consumed_tokens": 4, "tokens_per_sec": 2.0})
    led.write_status({"consumed_tokens": 4, "produced_tokens": 10, "remaining_tokens": 6})
    assert led.consumed_tokens() == 4
    assert "loss" in led.metrics_path.read_text(encoding="utf-8")


def test_stage_yaml_mix_is_synth_only():
    for name in ("s0_bringup.yaml", "s1_language.yaml", "s2_semantic.yaml"):
        payload = yaml.safe_load((Path("configs/stages") / name).read_text(encoding="utf-8"))
        assert payload["mix"]["synth"] == 1.0
        assert payload["mix"]["natural"] == 0.0
        assert payload["training"]["attention_implementation"] == "sdpa" or name.startswith("tiny")
        assert payload["hardware"]["flash_attention"] is False


def test_pretrain_colab_100m_disables_flash():
    payload = yaml.safe_load(Path("configs/pretrain_colab_100m.yaml").read_text(encoding="utf-8"))
    assert payload["hardware"]["flash_attention"] is False
    assert payload["hardware"]["attention_implementation"] == "sdpa"
    payload = yaml.safe_load(Path("configs/colab.yaml").read_text(encoding="utf-8"))
    assert payload["training"]["max_tokens"] == 0
    assert payload["storage"]["disk_ceiling_gb"] == 400
    assert payload["storage"]["working_cap_gb"] == 150
    assert payload["model"]["attention_implementation"] == "sdpa"
