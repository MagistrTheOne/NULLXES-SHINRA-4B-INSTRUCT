"""Path-only CLI adapter; model and training parameters come from one YAML."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from configs.loader import CONFIG_PATH, load_config


@dataclass
class TrainConfig:
    stage: str
    admission: str
    budget_kind: str
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
    schedule_steps: int
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
    parser.add_argument("--budget", choices=("diagnostic", "foundation"), default="diagnostic",
                        help="selects the separate diagnostic or foundation token target in shinra.yaml")
    parser.add_argument("--diagnostic-audit", action="store_true",
                        help="request diagnostic training after P0 identities and runtime gates are admitted")
    parser.add_argument("--diagnostic-max-steps", type=int, default=None,
                        help="optimizer-step cap for an explicit diagnostic run; defaults to one real update")


def build_train_config(stage: str, args: argparse.Namespace) -> TrainConfig:
    config_path = Path(args.config)
    diagnostic = bool(getattr(args, "diagnostic_audit", False))
    raw = load_config(config_path, require_ready=not diagnostic)
    if diagnostic:
        if raw["status"] != "ready_for_training" or raw["training"].get("enabled") is not True \
                or raw["data"].get("training_enabled") is not True \
                or raw["runtime"].get("colab_launch_allowed") is not True:
            raise ValueError("diagnostic training is unavailable until every P0 gate is explicitly admitted")
        if not raw["data"].get("manifest_sha256") \
                or not raw["tokenizer"].get("artifact_sha256") \
                or not raw["tokenizer"].get("native_model_sha256") \
                or raw["data"].get("acquisition") != "nullxes_synthetic_engine" \
                or raw["data"].get("synthetic", {}).get("external_llm_generation_allowed") is not False \
                or raw["data"].get("synthetic", {}).get("external_text_sources_allowed") is not False:
            raise ValueError("diagnostic training requires frozen Native tokenizer and synthetic corpus identities")
        if raw["model"].get("loss_memory_status") != "runtime_verified":
            raise ValueError("diagnostic training is blocked until chunked loss memory and gradients are verified remotely")
        if stage != "pretrain":
            raise ValueError("diagnostic audit mode is available only for causal pretraining")
    elif getattr(args, "diagnostic_max_steps", None) is not None:
        raise ValueError("--diagnostic-max-steps requires the explicit --diagnostic-audit mode")
    train, model = raw["training"], raw["model"]
    opt, sched = train["optimizer"], train["scheduler"]
    budget_kind = getattr(args, "budget", "diagnostic")
    if diagnostic and budget_kind != "diagnostic":
        raise ValueError("audit-only runs can use the diagnostic budget only")
    diagnostic_steps = getattr(args, "diagnostic_max_steps", None)
    max_steps = ((1 if diagnostic_steps is None else diagnostic_steps) if diagnostic else train.get("max_steps"))
    if max_steps is None:
        raise ValueError("training.max_steps must be explicitly frozen before production training")
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("diagnostic optimizer-step cap must be a positive integer")
    budget_tokens = train.get(f"{budget_kind}_tokens")
    if not isinstance(budget_tokens, int) or budget_tokens <= 0:
        raise ValueError(f"training.{budget_kind}_tokens must be approved and positive before this budget can run")
    schedule_steps = (math.ceil(budget_tokens / int(train["global_tokens_per_update"]))
                      if diagnostic else int(max_steps))
    admission = "diagnostic_audit" if diagnostic else "production"
    default_output = Path(raw["checkpoint"]["output_dir"]) / ("diagnostic-audit" if diagnostic else budget_kind)
    output = Path(args.output_dir or default_output)
    if diagnostic and "diagnostic" not in str(output).casefold():
        raise ValueError("diagnostic checkpoint output path must include 'diagnostic' to prevent production confusion")
    return TrainConfig(
        stage=stage,
        admission=admission,
        budget_kind=budget_kind,
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
        warmup_steps=int(schedule_steps * float(sched["warmup_ratio"])),
        max_steps=int(max_steps),
        schedule_steps=int(schedule_steps),
        max_tokens=int(budget_tokens),
        fused_optimizer=opt.get("fused"),
        scheduler=str(sched["name"]),
        grad_clip=float(opt["max_grad_norm"]),
        seed=int(train["seed"]),
        logging_steps=int(train["logging_steps"]),
        eval_steps=int(train["eval_steps"]),
        save_steps=1 if diagnostic else int(train["save_steps"]),
        gradient_checkpointing=bool(train["gradient_checkpointing"]),
        attention_implementation=str(model["attention_implementation"]),
        dataloader_num_workers=int(train["dataloader_num_workers"]),
        run_dir=Path(args.run_dir) if args.run_dir else output,
        heldout_dir=Path(args.heldout_dir) if args.heldout_dir else None,
    )
