"""Rolling drum: one shard at a time, SHA cursor, kill/resume without replay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data.rolling_drum import DrumError, fake_encode, read_cursor, run_rolling_stage, write_cursor
from data.shard_lifecycle import build_packed_shard
from data.shards import ShardIdentityError, list_shard_bins
from training.trainer import load_trainer_state


def _args(tmp: Path, *, output: Path, **overrides) -> SimpleNamespace:
    ns = SimpleNamespace(
        config="tests/fixtures/tiny_shinra.yaml",
        stage_config="tests/fixtures/tiny_s0.yaml",
        storage_config="configs/storage_g4.yaml",
        corpus_dir=tmp / "corpus",
        tokenizer=str(tmp / "missing-tok"),
        output_dir=str(output),
        resume_from=None,
        records_per_shard=16,
        max_shards=8,
        fake_tokenizer=True,
        heldout_records=0,
        max_tokens=120,
        max_steps=1000,
        sequence_length=32,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        attention_implementation="eager",
        wandb_project=None,
        wandb_run_name=None,
        seed=0,
        stop_after_steps=None,
        record_consumed_trace=True,
        delete_consumed=True,
        force_cpu=True,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _events(run_dir: Path) -> list[str]:
    path = run_dir / "ledger.jsonl"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line)["event"])
    return rows


def test_drum_trains_one_shard_before_producing_next(tmp_path: Path):
    probe = tmp_path / "probe"
    meta = build_packed_shard(
        stage="s0",
        shard_index=0,
        seed=0,
        record_count=8,
        encode=fake_encode,
        output_dir=probe,
        sequence_length=32,
    )
    target = int(meta["produced_tokens"]) + 24
    out = tmp_path / "out"
    result = run_rolling_stage(
        _args(tmp_path, output=out, max_tokens=target, records_per_shard=8, seed=0)
    )
    assert result["done"] is True
    assert result["consumed_tokens"] >= target
    assert result["overshoot_tokens"] == result["consumed_tokens"] - target
    train_dir = tmp_path / "corpus" / "s0"
    assert list_shard_bins(train_dir) == []
    metas = sorted(train_dir.glob("shard-*.meta.json"))
    assert len(metas) >= 2
    events = _events(out / "run")
    ready = [i for i, name in enumerate(events) if name == "shard_ready"]
    trained = [i for i, name in enumerate(events) if name == "train_done"]
    assert len(ready) >= 2
    assert trained, events
    assert trained[0] < ready[1], events
    cursor = read_cursor(out / "run")
    assert cursor["status"] == "done"
    ckpt = Path(result["checkpoint_dir"])
    blob = load_trainer_state(ckpt / "trainer_state.pt")
    assert blob["shard_id"]
    assert blob["shard_sha256"]
    assert len(blob["shard_sha256"]) == 64


def test_max_shards_aborts_before_silent_short_corpus(tmp_path: Path):
    with pytest.raises(DrumError, match="max_shards"):
        run_rolling_stage(
            _args(tmp_path, output=tmp_path / "out", max_tokens=5000, records_per_shard=8, max_shards=1)
        )


def test_kill_mid_shard_resume_matches_uninterrupted(tmp_path: Path):
    full_dir = tmp_path / "full"
    killed_dir = tmp_path / "killed"
    common = dict(max_tokens=192, records_per_shard=20, seed=1)
    full = run_rolling_stage(_args(tmp_path / "a", output=full_dir, **common))
    assert full["consumed_trace"]
    assert len(full["consumed_trace"]) > 4

    first = run_rolling_stage(
        _args(tmp_path / "b", output=killed_dir, stop_after_steps=2, **common)
    )
    assert first["killed"] is True
    cursor = read_cursor(killed_dir / "run")
    assert cursor["status"] == "active"
    assert cursor["dataset_index"] == 2
    assert cursor["n_sequences"] > 2
    assert list_shard_bins(tmp_path / "b" / "corpus" / "s0")
    ckpt = Path(first["checkpoint_dir"])
    blob = load_trainer_state(ckpt / "trainer_state.pt")
    assert blob["dataset_index"] == 2
    assert blob["shard_sha256"] == cursor["shard_sha256"]
    assert blob["consumed_tokens"] == first["consumed_tokens"]

    resumed = run_rolling_stage(_args(tmp_path / "b", output=killed_dir, **common))
    assert resumed["killed"] is False
    assert resumed["done"] is True
    assert resumed["consumed_trace"] == full["consumed_trace"]
    assert resumed["consumed_trace"][: len(first["consumed_trace"])] == first["consumed_trace"]
    assert resumed["consumed_tokens"] == full["consumed_tokens"]
    assert resumed["step"] == full["step"]
    assert resumed["overshoot_tokens"] == full["overshoot_tokens"]
    assert resumed["shard_index"] == full["shard_index"]
    final_cursor = read_cursor(killed_dir / "run")
    full_cursor = read_cursor(full_dir / "run")
    assert final_cursor["consumed_tokens"] == full_cursor["consumed_tokens"]
    assert final_cursor["step"] == full_cursor["step"]


def test_resume_rejects_shard_sha_mismatch(tmp_path: Path):
    out = tmp_path / "out"
    run_rolling_stage(
        _args(tmp_path, output=out, stop_after_steps=2, max_tokens=192, records_per_shard=20)
    )
    cursor = read_cursor(out / "run")
    cursor["shard_sha256"] = "0" * 64
    write_cursor(out / "run", cursor)
    with pytest.raises(ShardIdentityError):
        run_rolling_stage(_args(tmp_path, output=out, max_tokens=192, records_per_shard=20))


def test_stage_c_refuses_synth_produce(tmp_path: Path):
    with pytest.raises(DrumError, match="refuses synth"):
        run_rolling_stage(
            _args(
                tmp_path,
                output=tmp_path / "out",
                stage_config="tests/fixtures/tiny_c.yaml",
                max_tokens=64,
                records_per_shard=8,
            )
        )


def test_stage_c_trains_prepacked_jsonl(tmp_path: Path):
    from data.data_v1.phase_c import pack_corpus

    jsonl = tmp_path / "docs.jsonl"
    lines = []
    for i in range(20):
        text = " ".join(f"tok{i}-{j}" for j in range(12))
        lines.append(json.dumps({"text": text, "source_id": "fineweb-edu-en"}))
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pack_corpus(
        input_path=jsonl,
        output_dir=tmp_path / "corpus" / "c",
        encode=fake_encode,
        sequence_length=32,
        target_honest_tokens=64,
        ceiling_honest_tokens=256,
    )
    result = run_rolling_stage(
        _args(
            tmp_path,
            output=tmp_path / "out",
            stage_config="tests/fixtures/tiny_c.yaml",
            max_tokens=64,
            records_per_shard=8,
            max_shards=1,
        )
    )
    assert result["done"] is True
    assert result["consumed_tokens"] >= 64
    assert result["stage"] == "c"
