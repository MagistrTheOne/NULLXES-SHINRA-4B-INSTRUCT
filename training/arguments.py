"""CLI adapter for the one production training configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from configs.loader import CONFIG_PATH, load_config


@dataclass
class TrainConfig:
    stage: str
    model_config: Path
    data_dir: Path
    tokenizer_path: Path
    output_dir: Path
    resume_from: Path | None
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    min_lr_ratio: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    warmup_steps: int
    max_steps: int
    max_tokens: int
    fused_optimizer: bool | None
    scheduler: str
    stable_ratio: float
    grad_clip: float
    seed: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    gradient_checkpointing: bool
    attention_implementation: str
    z_loss_coefficient: float
    dataloader_num_workers: int
    wandb_project: str | None
    wandb_run_name: str | None
    dpo_beta: float
    sft_from: Path | None
    base_from: Path | None
    run_dir: Path
    heldout_dir: Path | None
    allow_replay: bool
    disk_ceiling_gb: float
    active_shard_id: str | None
    active_shard_sha256: str | None
    stop_after_steps: int | None
    record_consumed_trace: bool
    force_cpu: bool
    fresh_stage_ledger: bool = False


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/artifacts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--heldout-dir", default=None)
    parser.add_argument("--stop-after-steps", type=int, default=None)
    parser.add_argument("--force-cpu", action="store_true", help=argparse.SUPPRESS)


def build_train_config(stage: str, args: argparse.Namespace) -> TrainConfig:
    config_path = Path(args.config)
    raw = load_config(config_path, require_ready=True)
    train, model = raw["training"], raw["model"]
    opt, sched = train["optimizer"], train["scheduler"]
    max_tokens = int(train["diagnostic_tokens"])
    max_steps = train.get("max_steps")
    if max_steps is None:
        raise ValueError("training.max_steps must be explicitly frozen before training")
    output = Path(args.output_dir)
    run_dir = Path(args.run_dir) if args.run_dir else output
    heldout = args.heldout_dir
    return TrainConfig(
        stage=stage, model_config=config_path, data_dir=Path(args.data_dir),
        tokenizer_path=Path(args.tokenizer), output_dir=output,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        sequence_length=int(train["sequence_length"]),
        micro_batch_size=int(train["micro_batch_size"]),
        gradient_accumulation_steps=int(train["gradient_accumulation_steps"]),
        learning_rate=float(opt["learning_rate"]), min_lr_ratio=float(sched["min_lr_ratio"]),
        weight_decay=float(opt["weight_decay"]), betas=tuple(map(float, opt["betas"])),
        eps=float(opt["eps"]), warmup_steps=int(max_steps * float(sched["warmup_ratio"])),
        max_steps=int(max_steps), max_tokens=max_tokens, fused_optimizer=opt.get("fused"), scheduler=str(sched["name"]),
        stable_ratio=1.0, grad_clip=float(opt["max_grad_norm"]), seed=int(train["seed"]),
        logging_steps=int(train["logging_steps"]), eval_steps=int(train["eval_steps"]),
        save_steps=int(train["save_steps"]),
        gradient_checkpointing=bool(train["gradient_checkpointing"]),
        attention_implementation=str(model["attention_implementation"]),
        z_loss_coefficient=float(model["z_loss_coefficient"]),
        dataloader_num_workers=int(train["dataloader_num_workers"]),
        wandb_project=args.wandb_project or None, wandb_run_name=args.wandb_run_name,
        dpo_beta=0.1, sft_from=None, base_from=None, run_dir=run_dir,
        heldout_dir=Path(heldout) if heldout else None, allow_replay=False,
        disk_ceiling_gb=float(raw["runtime"].get("minimum_free_disk_gb", 80)),
        active_shard_id=None, active_shard_sha256=None,
        stop_after_steps=args.stop_after_steps, record_consumed_trace=False,
        force_cpu=bool(args.force_cpu),
    )
