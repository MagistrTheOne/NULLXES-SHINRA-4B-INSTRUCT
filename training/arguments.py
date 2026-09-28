"""Shared argument parsing for SHINRA training stages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml


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


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/shinra_4b.yaml")
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/artifacts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--attention-implementation", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--heldout-dir", default=None)
    parser.add_argument("--allow-replay", action="store_true")


def build_train_config(stage: str, args: argparse.Namespace) -> TrainConfig:
    root = load_yaml(Path(args.config))
    extra = load_yaml(Path(args.train_config)) if args.train_config else {}
    batch = extra.get("batch", root.get("batch", {}))
    opt = extra.get("optimizer", root.get("optimizer", {}))
    sch = extra.get("scheduler", root.get("scheduler", {}))
    train = extra.get("training", root.get("training", {}))
    model_cfg = root.get("model", {})
    tokens_per_step = batch.get("tokens_per_step", 2_097_152)
    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    else:
        max_tokens = train.get("max_tokens", extra.get("max_tokens", 200_000_000_000))
    max_steps = args.max_steps or extra.get("max_steps") or int(int(max_tokens) / max(tokens_per_step, 1))
    if int(max_tokens) <= 0:
        max_steps = args.max_steps or extra.get("max_steps") or 0
    betas = tuple(opt.get("betas", [0.9, 0.95]))
    wandb = getattr(args, "wandb_project", None) or None
    if wandb == "":
        wandb = None
    allow_replay = bool(train.get("allow_replay", stage in {"sft", "dpo"}))
    if getattr(args, "allow_replay", False):
        allow_replay = True
    run_dir = Path(args.run_dir) if getattr(args, "run_dir", None) else Path(args.output_dir)
    heldout = getattr(args, "heldout_dir", None) or extra.get("heldout_dir") or train.get("heldout_dir")
    storage = extra.get("storage", root.get("storage", {}))
    attn = (
        args.attention_implementation
        or train.get("attention_implementation")
        or model_cfg.get("attention_implementation", "sdpa")
    )
    return TrainConfig(
        stage=stage,
        model_config=Path(args.config),
        data_dir=Path(args.data_dir),
        tokenizer_path=Path(args.tokenizer),
        output_dir=Path(args.output_dir),
        resume_from=Path(args.resume_from) if args.resume_from else None,
        sequence_length=args.sequence_length or batch.get("sequence_length", 8192),
        micro_batch_size=args.micro_batch_size or batch.get("micro_batch_size", 2),
        gradient_accumulation_steps=args.gradient_accumulation_steps
        or batch.get("gradient_accumulation_steps", 16),
        learning_rate=args.learning_rate or opt.get("learning_rate", 3e-4),
        min_lr_ratio=sch.get("min_lr_ratio", 0.1),
        weight_decay=opt.get("weight_decay", 0.1),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(opt.get("eps", 1e-8)),
        warmup_steps=int(sch.get("warmup_steps", 2000)),
        max_steps=int(max_steps),
        max_tokens=int(max_tokens),
        scheduler=sch.get("name", "wsd"),
        stable_ratio=float(sch.get("stable_ratio", 0.80)),
        grad_clip=float(opt.get("grad_clip", train.get("max_grad_norm", 1.0))),
        seed=args.seed or int(train.get("seed", 42)),
        logging_steps=int(train.get("logging_steps", 10)),
        eval_steps=int(train.get("eval_steps", 1000)),
        save_steps=int(train.get("save_steps", 1000)),
        gradient_checkpointing=bool(train.get("gradient_checkpointing", True)),
        attention_implementation=attn,
        z_loss_coefficient=float(model_cfg.get("z_loss_coefficient", 1e-5)),
        dataloader_num_workers=int(train.get("dataloader_num_workers", 8)),
        wandb_project=wandb,
        wandb_run_name=args.wandb_run_name,
        dpo_beta=float(extra.get("dpo", {}).get("beta", 0.1)),
        sft_from=Path(args.sft_from) if getattr(args, "sft_from", None) else None,
        base_from=Path(args.base_from) if getattr(args, "base_from", None) else None,
        run_dir=run_dir,
        heldout_dir=Path(heldout) if heldout else None,
        allow_replay=allow_replay,
        disk_ceiling_gb=float(storage.get("disk_ceiling_gb", storage.get("max_disk_gb", 400))),
    )
