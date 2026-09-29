"""Honest pretrain loop: no silent replay, stop at token budget."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from data.shard_lifecycle import build_packed_shard
from training.arguments import build_train_config
from training.trainer import run_lm_training


def _encode(text: str) -> list[int]:
    ids = []
    for word in text.split():
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        ids.append(19 + int.from_bytes(digest[:2], "big") % 40)
    return ids or [19]


def test_no_replay_stops_at_budget(tmp_path: Path):
    data = tmp_path / "data"
    meta = build_packed_shard(
        stage="s0",
        shard_index=0,
        seed=0,
        record_count=40,
        encode=_encode,
        output_dir=data,
        sequence_length=32,
    )
    out = tmp_path / "out"
    ns = SimpleNamespace(
        config="tests/fixtures/tiny_shinra.yaml",
        train_config="tests/fixtures/tiny_s0.yaml",
        data_dir=str(data),
        tokenizer=str(tmp_path / "missing-tok"),
        output_dir=str(out),
        resume_from=None,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        max_steps=1000,
        max_tokens=64,
        sequence_length=32,
        attention_implementation="eager",
        wandb_project=None,
        wandb_run_name=None,
        seed=0,
        run_dir=str(out / "run"),
        heldout_dir=None,
        allow_replay=False,
        force_cpu=True,
    )
    cfg = build_train_config("s0", ns)
    cfg.allow_replay = False
    cfg.dataloader_num_workers = 0
    cfg.eval_steps = 0
    cfg.save_steps = 0
    run_lm_training(cfg)
    status = (out / "run" / "status.json").read_text(encoding="utf-8")
    assert "consumed_tokens" in status
    consumed = int(__import__("json").loads(status)["consumed_tokens"])
    assert consumed >= 64
    assert consumed <= meta["produced_tokens"]
    assert __import__("json").loads(status).get("done") is True
    assert __import__("json").loads(status).get("overshoot_tokens", 0) >= 0


def test_token_budget_does_not_cut_mid_accumulation(tmp_path: Path):
    data = tmp_path / "data"
    build_packed_shard(
        stage="s0",
        shard_index=0,
        seed=0,
        record_count=40,
        encode=_encode,
        output_dir=data,
        sequence_length=32,
    )
    out = tmp_path / "out"
    ns = SimpleNamespace(
        config="tests/fixtures/tiny_shinra.yaml",
        train_config="tests/fixtures/tiny_s0.yaml",
        data_dir=str(data),
        tokenizer=str(tmp_path / "missing-tok"),
        output_dir=str(out),
        resume_from=None,
        micro_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        max_steps=1000,
        max_tokens=20,
        sequence_length=32,
        attention_implementation="eager",
        wandb_project=None,
        wandb_run_name=None,
        seed=0,
        run_dir=str(out / "run"),
        heldout_dir=None,
        allow_replay=False,
        record_consumed_trace=True,
        force_cpu=True,
    )
    cfg = build_train_config("s0", ns)
    cfg.allow_replay = False
    cfg.dataloader_num_workers = 0
    cfg.eval_steps = 0
    cfg.save_steps = 0
    cfg.record_consumed_trace = True
    result = run_lm_training(cfg)
    assert result["step"] == 1
    assert result["dataset_index"] == 2
    assert result["consumed_tokens"] >= 20
    assert result["overshoot_tokens"] == result["consumed_tokens"] - 20
    assert len(result["consumed_trace"]) == 2


def test_fresh_stage_ledger_resets_consumed_from_prior_ckpt(tmp_path: Path):
    data = tmp_path / "data"
    build_packed_shard(
        stage="s0",
        shard_index=0,
        seed=0,
        record_count=40,
        encode=_encode,
        output_dir=data,
        sequence_length=32,
    )

    def _ns(out: Path, **over):
        ns = SimpleNamespace(
            config="tests/fixtures/tiny_shinra.yaml",
            train_config="tests/fixtures/tiny_s0.yaml",
            data_dir=str(data),
            tokenizer=str(tmp_path / "missing-tok"),
            output_dir=str(out),
            resume_from=None,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=3e-4,
            max_steps=1000,
            max_tokens=64,
            sequence_length=32,
            attention_implementation="eager",
            wandb_project=None,
            wandb_run_name=None,
            seed=0,
            run_dir=str(out / "run"),
            heldout_dir=None,
            allow_replay=False,
            force_cpu=True,
        )
        for key, value in over.items():
            setattr(ns, key, value)
        return ns

    first_out = tmp_path / "first"
    cfg = build_train_config("s0", _ns(first_out))
    cfg.allow_replay = False
    cfg.dataloader_num_workers = 0
    cfg.eval_steps = 0
    cfg.save_steps = 0
    first = run_lm_training(cfg)
    assert first["consumed_tokens"] >= 64

    second_out = tmp_path / "second"
    ns = _ns(second_out, resume_from=first["checkpoint_dir"], max_tokens=32)
    cfg2 = build_train_config("c", ns)
    cfg2.allow_replay = False
    cfg2.dataloader_num_workers = 0
    cfg2.eval_steps = 0
    cfg2.save_steps = 0
    cfg2.fresh_stage_ledger = True
    cfg2.force_cpu = True
    second = run_lm_training(cfg2)
    assert second["consumed_tokens"] >= 32
    assert second["consumed_tokens"] < first["consumed_tokens"]
    assert second["step"] < first["step"] or second["consumed_tokens"] <= 64
