"""Rolling S0–S2 drum: produce shard N → SHA → train once → consume/delete → N+1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import yaml
from transformers import AutoTokenizer

from data.ledger import RunLedger
from data.pack import resolve_pretrain_special_ids
from data.shard_lifecycle import build_packed_shard, encode_with_tokenizer, update_shard_status
from data.shards import (
    ShardIdentityError,
    assert_single_shard,
    list_shard_bins,
    verify_shard,
    verify_shard_identity,
)
from training.arguments import build_train_config
from training.trainer import run_lm_training


class DrumError(RuntimeError):
    pass


EncodeFn = Callable[[str], list[int]]


def fake_encode(text: str) -> list[int]:
    ids = []
    for word in text.split():
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        ids.append(19 + int.from_bytes(digest[:2], "big") % 40)
    return ids or [19]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def cursor_path(run_dir: Path) -> Path:
    return Path(run_dir) / "cursor.json"


def write_cursor(run_dir: Path, cursor: dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(cursor)
    cursor_path(run_dir).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_cursor(run_dir: Path) -> dict[str, Any]:
    path = cursor_path(run_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def local_checkpoint(output_dir: Path) -> Path | None:
    cursor = read_cursor(Path(output_dir) / "run")
    listed = cursor.get("checkpoint_dir")
    if listed and (Path(listed) / "trainer_state.pt").exists():
        return Path(listed)
    final_root = Path(output_dir) / "final"
    if final_root.is_dir():
        steps = sorted(p for p in final_root.glob("step-*") if p.is_dir() and (p / "trainer_state.pt").exists())
        if steps:
            return steps[-1]
    steps = sorted(p for p in Path(output_dir).glob("step-*") if p.is_dir() and (p / "trainer_state.pt").exists())
    if steps:
        return steps[-1]
    return None


def resolve_checkpoint(output_dir: Path, resume_from: str | Path | None) -> Path | None:
    local = local_checkpoint(output_dir)
    if local:
        return local
    if resume_from:
        path = Path(resume_from)
        if (path / "trainer_state.pt").exists():
            return path
    return None


def _train_ns(
    *,
    model_config: Path,
    stage_config: Path,
    train_dir: Path,
    tokenizer: str,
    output_dir: Path,
    run_dir: Path,
    heldout_dir: Path,
    resume_from: Path | None,
    stage_cfg: dict,
    seq_len: int,
    target: int,
    seed: int,
    args: Any,
    shard_id: str,
    shard_sha: str,
    stop_after_steps: int | None,
    record_consumed_trace: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        config=str(model_config),
        train_config=str(stage_config),
        data_dir=str(train_dir),
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        resume_from=str(resume_from) if resume_from else None,
        micro_batch_size=getattr(args, "micro_batch_size", None),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", None),
        learning_rate=getattr(args, "learning_rate", None),
        max_steps=getattr(args, "max_steps", None),
        max_tokens=target,
        sequence_length=seq_len,
        attention_implementation=getattr(args, "attention_implementation", None),
        wandb_project=getattr(args, "wandb_project", None),
        wandb_run_name=getattr(args, "wandb_run_name", None),
        seed=seed,
        run_dir=str(run_dir),
        heldout_dir=str(heldout_dir),
        allow_replay=False,
        active_shard_id=shard_id,
        active_shard_sha256=shard_sha,
        stop_after_steps=stop_after_steps,
        record_consumed_trace=record_consumed_trace,
        force_cpu=bool(getattr(args, "force_cpu", False)),
        fresh_stage_ledger=bool(getattr(args, "fresh_stage_ledger", False)),
    )


def _produce_shard(
    *,
    stage: str,
    shard_index: int,
    seed: int,
    record_count: int,
    encode: EncodeFn,
    train_dir: Path,
    seq_len: int,
    start_index: int,
    bos_id: int,
    end_id: int,
    pad_id: int,
    corpus: Path,
    ceiling_gb: float,
    ledger: RunLedger,
    expected_sha: str | None,
) -> dict:
    leftover = list_shard_bins(train_dir)
    if leftover:
        raise DrumError(f"cannot produce shard-{shard_index:05d}: active bins still present {[p.name for p in leftover]}")
    meta = build_packed_shard(
        stage=stage,
        shard_index=shard_index,
        seed=seed,
        record_count=record_count,
        encode=encode,
        output_dir=train_dir,
        sequence_length=seq_len,
        start_index=start_index,
        bos_id=bos_id,
        end_id=end_id,
        pad_id=pad_id,
        ceiling_root=corpus,
        ceiling_gb=ceiling_gb,
        ledger=ledger,
    )
    bin_path = train_dir / f"{meta['shard_id']}.bin"
    verify_shard(bin_path)
    if expected_sha is not None and meta["sha256"] != expected_sha:
        raise ShardIdentityError(
            f"rebuilt {meta['shard_id']} sha {meta['sha256']} != committed {expected_sha}"
        )
    return meta


def run_rolling_stage(args: Any) -> dict[str, Any]:
    stage_cfg = load_yaml(Path(args.stage_config))
    storage = load_yaml(Path(args.storage_config)).get("storage", {})
    stage = stage_cfg["stage"]
    seq_len = args.sequence_length or int(stage_cfg.get("batch", {}).get("sequence_length", 2048))
    target = args.max_tokens if args.max_tokens is not None else int(
        stage_cfg.get("new_tokens", stage_cfg.get("max_tokens", 0))
    )
    seed = args.seed or int(stage_cfg.get("training", {}).get("seed", 42))
    corpus = Path(args.corpus_dir)
    train_dir = corpus / stage
    heldout_dir = corpus / "heldout"
    output_dir = Path(args.output_dir)
    run_dir = output_dir / "run"
    train_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(run_dir)
    ceiling_gb = float(storage.get("disk_ceiling_gb", 400))
    delete_consumed = bool(getattr(args, "delete_consumed", True))
    stop_after_steps = getattr(args, "stop_after_steps", None)
    record_trace = bool(getattr(args, "record_consumed_trace", False))
    records_per_shard = int(args.records_per_shard)
    max_shards = int(args.max_shards)

    if args.fake_tokenizer:
        encode: EncodeFn = fake_encode
        bos_id, end_id, pad_id = 1, 18, 3
    else:
        tok = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
        encode = lambda text: encode_with_tokenizer(tok, text)
        bos_id, end_id, pad_id = resolve_pretrain_special_ids(tok)

    if stage != "c" and not list_shard_bins(heldout_dir) and int(getattr(args, "heldout_records", 0) or 0) > 0:
        build_packed_shard(
            stage=stage,
            shard_index=0,
            seed=seed + 10_000,
            record_count=int(args.heldout_records),
            encode=encode,
            output_dir=heldout_dir,
            sequence_length=seq_len,
            start_index=0,
            bos_id=bos_id,
            end_id=end_id,
            pad_id=pad_id,
            ledger=ledger,
        )

    checkpoint = resolve_checkpoint(output_dir, getattr(args, "resume_from", None))
    fresh_stage_ledger = bool(checkpoint) and not _checkpoint_under(Path(checkpoint), Path(output_dir))
    args.fresh_stage_ledger = fresh_stage_ledger
    cursor = read_cursor(run_dir)
    shard_index = int(cursor.get("shard_index", 0))
    start_index = int(cursor.get("next_record_start", 0))
    produced = int(cursor.get("produced_tokens", 0))
    consumed = int(cursor.get("consumed_tokens", 0))
    step = int(cursor.get("step", 0))
    active_id = cursor.get("shard_id")
    active_sha = cursor.get("shard_sha256")
    trace: list[int] = list(cursor.get("consumed_trace") or [])
    resume_from = checkpoint

    ledger.append(
        {
            "event": "drum_start",
            "stage": stage,
            "target_tokens": target,
            "resume_from": str(resume_from) if resume_from else None,
            "shard_index": shard_index,
        }
    )

    while True:
        if target > 0 and consumed >= target:
            break
        bins = list_shard_bins(train_dir)
        need_new = not bins
        if bins:
            live = assert_single_shard(train_dir, active_id if active_id else None)
            meta = verify_shard(live)
            if active_sha:
                verify_shard_identity(live, meta["shard_id"], active_sha)
            else:
                active_id = meta["shard_id"]
                active_sha = meta["sha256"]
                shard_index = int(meta["shard_id"].split("-")[1])
        if need_new:
            if stage == "c":
                raise DrumError("stage c refuses synth produce; pack FineWeb JSONL first")
            if shard_index >= max_shards:
                raise DrumError(
                    f"max_shards={max_shards} reached with consumed_tokens={consumed} < target={target}"
                )
            meta = _produce_shard(
                stage=stage,
                shard_index=shard_index,
                seed=seed,
                record_count=records_per_shard,
                encode=encode,
                train_dir=train_dir,
                seq_len=seq_len,
                start_index=start_index,
                bos_id=bos_id,
                end_id=end_id,
                pad_id=pad_id,
                corpus=corpus,
                ceiling_gb=ceiling_gb,
                ledger=ledger,
                expected_sha=active_sha if active_id == f"shard-{shard_index:05d}" else None,
            )
            produced += int(meta["produced_tokens"])
            active_id = meta["shard_id"]
            active_sha = meta["sha256"]
        else:
            meta = verify_shard(assert_single_shard(train_dir))
            active_id = meta["shard_id"]
            active_sha = meta["sha256"]
            if produced == 0:
                produced += int(meta.get("produced_tokens") or 0)

        bin_path = train_dir / f"{active_id}.bin"
        update_shard_status(bin_path, "active", ledger=ledger)
        cursor_state = {
            "stage": stage,
            "status": "active",
            "shard_index": shard_index,
            "shard_id": active_id,
            "shard_sha256": active_sha,
            "n_sequences": meta["n_sequences"],
            "produced_tokens": produced,
            "consumed_tokens": consumed,
            "step": step,
            "next_record_start": start_index,
            "records_per_shard": records_per_shard,
            "seed": seed,
            "sequence_length": seq_len,
            "target_tokens": target,
            "checkpoint_dir": str(resume_from) if resume_from else None,
            "consumed_trace": trace if record_trace else None,
        }
        write_cursor(run_dir, cursor_state)
        ledger.write_status({**cursor_state, "event": "shard_active"})

        ns = _train_ns(
            model_config=Path(args.config),
            stage_config=Path(args.stage_config),
            train_dir=train_dir,
            tokenizer=str(args.tokenizer),
            output_dir=output_dir,
            run_dir=run_dir,
            heldout_dir=heldout_dir,
            resume_from=resume_from,
            stage_cfg=stage_cfg,
            seq_len=seq_len,
            target=target,
            seed=seed,
            args=args,
            shard_id=active_id,
            shard_sha=active_sha,
            stop_after_steps=stop_after_steps,
            record_consumed_trace=record_trace,
        )
        cfg = build_train_config(stage, ns)
        cfg.allow_replay = False
        cfg.dataloader_num_workers = int(stage_cfg.get("training", {}).get("dataloader_num_workers", cfg.dataloader_num_workers))
        result = run_lm_training(cfg)
        consumed = int(result["consumed_tokens"])
        step = int(result["step"])
        resume_from = Path(result["checkpoint_dir"]) if result.get("checkpoint_dir") else resume_from
        if record_trace:
            trace.extend(result.get("consumed_trace") or [])
        overshoot = max(0, consumed - target) if target > 0 else 0
        cursor_state.update(
            {
                "consumed_tokens": consumed,
                "step": step,
                "dataset_index": result["dataset_index"],
                "checkpoint_dir": str(resume_from) if resume_from else None,
                "overshoot_tokens": overshoot,
                "exhausted": result["exhausted"],
                "killed": result["killed"],
                "consumed_trace": trace if record_trace else None,
            }
        )
        write_cursor(run_dir, cursor_state)

        if result["killed"]:
            ledger.write_status({**cursor_state, "status": "killed", "done": False})
            return {**cursor_state, "killed": True, "consumed_trace": trace}

        if not result["exhausted"] and not (target > 0 and consumed >= target):
            raise DrumError(
                f"trainer stopped before shard exhaustion or token target "
                f"(step={step} consumed={consumed} target={target} max_steps={cfg.max_steps})"
            )

        mark = update_shard_status(
            bin_path,
            "consumed",
            ledger=ledger,
            extra={"consumed_tokens": consumed, "dataset_index": result["dataset_index"]},
            delete_bin=delete_consumed,
        )
        if list_shard_bins(train_dir):
            raise DrumError(f"shard {active_id} still on disk after consume")
        shard_index += 1
        start_index += records_per_shard
        active_id = None
        active_sha = None
        cursor_state.update(
            {
                "status": "need_shard",
                "shard_index": shard_index,
                "shard_id": None,
                "shard_sha256": None,
                "next_record_start": start_index,
                "last_consumed_shard": mark["shard_id"],
            }
        )
        write_cursor(run_dir, cursor_state)
        if target > 0 and consumed >= target:
            break

    overshoot = max(0, consumed - target) if target > 0 else 0
    final = {
        "stage": stage,
        "status": "done",
        "done": True,
        "killed": False,
        "consumed_tokens": consumed,
        "produced_tokens": produced,
        "target_tokens": target,
        "overshoot_tokens": overshoot,
        "step": step,
        "shard_index": shard_index,
        "checkpoint_dir": str(resume_from) if resume_from else None,
        "consumed_trace": trace if record_trace else None,
    }
    write_cursor(run_dir, final)
    ledger.write_status(final)
    ledger.append({"event": "drum_done", **{k: v for k, v in final.items() if k != "consumed_trace"}})
    return final
