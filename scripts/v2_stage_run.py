"""S0–S2 stage controller: synth shard → pack bin → train → ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import yaml
from transformers import AutoTokenizer

from data.ledger import RunLedger
from data.pack import resolve_pretrain_special_ids
from data.shard_lifecycle import build_packed_shard, encode_with_tokenizer
from data.shards import verify_shard
from training.arguments import build_train_config
from training.trainer import run_lm_training


def fake_encode(text: str) -> list[int]:
    ids = []
    for word in text.split():
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        ids.append(19 + int.from_bytes(digest[:2], "big") % 1000)
    return ids or [19]


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="SHINRA v2 S0–S2 rolling shard train")
    parser.add_argument("--config", default="configs/shinra_4b.yaml")
    parser.add_argument("--stage-config", required=True)
    parser.add_argument("--storage-config", default="configs/storage_g4.yaml")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="tokenizer/artifacts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--records-per-shard", type=int, default=4096)
    parser.add_argument("--max-shards", type=int, default=8)
    parser.add_argument("--fake-tokenizer", action="store_true")
    parser.add_argument("--heldout-records", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--attention-implementation", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    stage_cfg = load_yaml(Path(args.stage_config))
    storage = load_yaml(Path(args.storage_config)).get("storage", {})
    stage = stage_cfg["stage"]
    seq_len = args.sequence_length or int(stage_cfg.get("batch", {}).get("sequence_length", 2048))
    target = args.max_tokens or int(stage_cfg.get("new_tokens", stage_cfg.get("max_tokens", 0)))
    seed = args.seed or int(stage_cfg.get("training", {}).get("seed", 42))
    corpus = Path(args.corpus_dir)
    train_dir = corpus / stage
    heldout_dir = corpus / "heldout"
    output_dir = Path(args.output_dir)
    run_dir = output_dir / "run"
    ledger = RunLedger(run_dir)

    if args.fake_tokenizer:
        encode = fake_encode
        bos_id, end_id, pad_id = 1, 18, 3
    else:
        tok = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
        encode = lambda text: encode_with_tokenizer(tok, text)
        bos_id, end_id, pad_id = resolve_pretrain_special_ids(tok)

    produced = 0
    start = 0
    shard_i = 0
    metas = []
    while produced < target and shard_i < args.max_shards:
        meta = build_packed_shard(
            stage=stage,
            shard_index=shard_i,
            seed=seed,
            record_count=args.records_per_shard,
            encode=encode,
            output_dir=train_dir,
            sequence_length=seq_len,
            start_index=start,
            bos_id=bos_id,
            end_id=end_id,
            pad_id=pad_id,
            ceiling_root=corpus,
            ceiling_gb=float(storage.get("disk_ceiling_gb", 400)),
            ledger=ledger,
        )
        verify_shard(train_dir / f"{meta['shard_id']}.bin")
        produced += int(meta["produced_tokens"])
        start += args.records_per_shard
        shard_i += 1
        metas.append(meta)
        ledger.write_status({"stage": stage, "produced_tokens": produced, "target_tokens": target, "shards": shard_i})

    build_packed_shard(
        stage=stage,
        shard_index=0,
        seed=seed + 10_000,
        record_count=args.heldout_records,
        encode=encode,
        output_dir=heldout_dir,
        sequence_length=seq_len,
        start_index=0,
        bos_id=bos_id,
        end_id=end_id,
        pad_id=pad_id,
        ledger=ledger,
    )

    ns = SimpleNamespace(
        config=args.config,
        train_config=args.stage_config,
        data_dir=str(train_dir),
        tokenizer=args.tokenizer,
        output_dir=str(output_dir),
        resume_from=args.resume_from,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        max_tokens=target,
        sequence_length=seq_len,
        attention_implementation=args.attention_implementation,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=seed,
        run_dir=str(run_dir),
        heldout_dir=str(heldout_dir),
        allow_replay=False,
    )
    cfg = build_train_config(stage, ns)
    cfg.allow_replay = False
    run_lm_training(cfg)
    print(json.dumps({"stage": stage, "produced_tokens": produced, "shards": len(metas)}, indent=2))


if __name__ == "__main__":
    main()
