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
    assert consumed <= meta["produced_tokens"]
    assert consumed >= 32
    assert __import__("json").loads(status).get("done") is True
