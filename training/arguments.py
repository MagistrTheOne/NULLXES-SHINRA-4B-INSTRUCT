"""Path-only CLI adapter; model and training parameters come from one YAML."""

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
    stable_ratio: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    warmup_steps: int
    max_steps: int
    max_tokens: int
    fused_optimizer: bool | None
    scheduler: str
    grad_clip: float
    seed: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    gradient_checkpointing: bool
    attention_implementation: str
    dataloader_num_workers: int
    run_dir: Path
    heldout_dir: Path | None


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--heldout-dir", default=None)


def build_train_config(stage: str, args: argparse.Namespace) -> TrainConfig:
    config_path = Path(args.config)
    raw = load_config(config_path, require_ready=True)
    train, model = raw["training"], raw["model"]
    opt, sched = train["optimizer"], train["scheduler"]
    max_steps = train.get("max_steps")
    if max_steps is None:
        raise ValueError("training.max_steps must be explicitly frozen before training")
    output = Path(args.output_dir or raw["checkpoint"]["output_dir"])
    return TrainConfig(
        stage=stage,
        model_config=config_path,
        data_dir=Path(args.data_dir),
        tokenizer_path=Path(args.tokenizer or raw["tokenizer"]["artifact_dir"]),
        output_dir=output,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        sequence_length=int(train["sequence_length"]),
        micro_batch_size=int(train["micro_batch_size"]),
        gradient_accumulation_steps=int(train["gradient_accumulation_steps"]),
        learning_rate=float(opt["learning_rate"]),
        min_lr_ratio=float(sched["min_lr_ratio"]),
        stable_ratio=float(sched.get("stable_ratio", 0.0)),
        weight_decay=float(opt["weight_decay"]),
        betas=tuple(map(float, opt["betas"])),
        eps=float(opt["eps"]),
        warmup_steps=int(max_steps * float(sched["warmup_ratio"])),
        max_steps=int(max_steps),
        max_tokens=int(train["diagnostic_tokens"]),
        fused_optimizer=opt.get("fused"),
        scheduler=str(sched["name"]),
        grad_clip=float(opt["max_grad_norm"]),
        seed=int(train["seed"]),
        logging_steps=int(train["logging_steps"]),
        eval_steps=int(train["eval_steps"]),
        save_steps=int(train["save_steps"]),
        gradient_checkpointing=bool(train["gradient_checkpointing"]),
        attention_implementation=str(model["attention_implementation"]),
        dataloader_num_workers=int(train["dataloader_num_workers"]),
        run_dir=Path(args.run_dir) if args.run_dir else output,
        heldout_dir=Path(args.heldout_dir) if args.heldout_dir else None,
    )
