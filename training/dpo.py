"""Stage 3 — DPO / preference optimization on SHINRA-4B-INSTRUCT."""

from __future__ import annotations

from pathlib import Path

import torch
from accelerate.utils import set_seed
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

from model import ShinraForCausalLM
from .arguments import add_common_args, build_train_config
from .parallel import enable_tf32


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DPO alignment for NULLXES SHINRA-4B-INSTRUCT")
    add_common_args(parser)
    parser.add_argument("--sft-from", required=True, help="Path to SHINRA-4B-INSTRUCT SFT checkpoint")
    parser.set_defaults(train_config="configs/dpo_a100.yaml")
    args = parser.parse_args()
    cfg = build_train_config("dpo", args)
    enable_tf32()
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.tokenizer_path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = ShinraForCausalLM.from_pretrained(cfg.sft_from, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    files = sorted(str(p) for p in Path(cfg.data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No DPO parquet files in {cfg.data_dir}")
    dataset = load_dataset("parquet", data_files=files, split="train")

    dpo_args = DPOConfig(
        output_dir=str(cfg.output_dir),
        bf16=True,
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=1,
        max_steps=cfg.max_steps if cfg.max_steps > 0 else -1,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        beta=cfg.dpo_beta,
        max_length=cfg.sequence_length,
        max_prompt_length=cfg.sequence_length // 2,
        remove_unused_columns=False,
        report_to=["wandb"] if cfg.wandb_project else [],
        run_name=cfg.wandb_run_name or "shinra-dpo",
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=cfg.dataloader_num_workers,
        lr_scheduler_type="cosine",
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.grad_clip,
        seed=cfg.seed,
        fsdp="full_shard auto_wrap",
        fsdp_config={"transformer_layer_cls_to_wrap": ["ShinraDecoderLayer"]},
    )
    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
    )
    try:
        trainer = DPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = DPOTrainer(tokenizer=tokenizer, **trainer_kwargs)
    trainer.train()
    trainer.save_model(str(cfg.output_dir / "final"))
    tokenizer.save_pretrained(str(cfg.output_dir / "final"))


if __name__ == "__main__":
    main()
