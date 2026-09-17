"""Accelerate + FSDP training loop for SHINRA pretrain and SFT."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from model.configuration_shinra import ShinraConfig
from model.modeling_shinra import ShinraForCausalLM
from .arguments import TrainConfig
from .collator import PackedParquetDataset, collate_lm
from .optim import build_optimizer
from .parallel import enable_tf32
from .schedule import build_scheduler


def config_from_yaml(path: Path, attention_implementation: str) -> ShinraConfig:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)["model"]
    cfg = ShinraConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw.get("head_dim", 128),
        hidden_act=raw.get("hidden_act", "silu"),
        max_position_embeddings=raw["max_position_embeddings"],
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
        rope_theta=float(raw.get("rope_theta", 1_000_000.0)),
        rope_scaling=raw.get("rope_scaling"),
        attention_dropout=float(raw.get("attention_dropout", 0.0)),
        residual_dropout=float(raw.get("residual_dropout", 0.0)),
        embedding_dropout=float(raw.get("embedding_dropout", 0.0)),
        attention_bias=bool(raw.get("attention_bias", False)),
        mlp_bias=bool(raw.get("mlp_bias", False)),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", True)),
        qk_norm=bool(raw.get("qk_norm", True)),
        pad_token_id=int(raw.get("pad_token_id", 3)),
        bos_token_id=int(raw.get("bos_token_id", 1)),
        eos_token_id=int(raw.get("eos_token_id", 2)),
        z_loss_coefficient=float(raw.get("z_loss_coefficient", 1e-5)),
        attention_implementation=attention_implementation,
        use_cache=False,
    )
    cfg._attn_implementation = attention_implementation
    cfg.auto_map = {
        "AutoConfig": "configuration_shinra.ShinraConfig",
        "AutoModel": "modeling_shinra.ShinraModel",
        "AutoModelForCausalLM": "modeling_shinra.ShinraForCausalLM",
    }
    return cfg


def load_model(cfg: TrainConfig) -> ShinraForCausalLM:
    if cfg.resume_from and (Path(cfg.resume_from) / "config.json").exists():
        model = ShinraForCausalLM.from_pretrained(cfg.resume_from, torch_dtype=torch.bfloat16)
    elif cfg.base_from and cfg.stage in {"sft", "dpo"}:
        model = ShinraForCausalLM.from_pretrained(cfg.base_from, torch_dtype=torch.bfloat16)
    elif cfg.sft_from and cfg.stage == "dpo":
        model = ShinraForCausalLM.from_pretrained(cfg.sft_from, torch_dtype=torch.bfloat16)
    else:
        shinra_cfg = config_from_yaml(cfg.model_config, cfg.attention_implementation)
        model = ShinraForCausalLM(shinra_cfg)
        model = model.to(dtype=torch.bfloat16)
    model.config.use_cache = False
    model.config._attn_implementation = cfg.attention_implementation
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def save_checkpoint(accelerator: Accelerator, model: ShinraForCausalLM, tokenizer, output_dir: Path, step: int) -> None:
    ckpt_dir = output_dir / f"step-{step:08d}"
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.config.auto_map = {
        "AutoConfig": "configuration_shinra.ShinraConfig",
        "AutoModel": "modeling_shinra.ShinraModel",
        "AutoModelForCausalLM": "modeling_shinra.ShinraForCausalLM",
    }
    accelerator.save_model(unwrapped, str(ckpt_dir))
    if accelerator.is_main_process:
        tokenizer.save_pretrained(str(ckpt_dir))
        unwrapped.config.save_pretrained(str(ckpt_dir))
        from scripts.export_hf_bundle import export_code

        export_code(ckpt_dir)
        meta = {"step": step, "stage": getattr(model, "stage", None)}
        (ckpt_dir / "train_state.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def run_lm_training(cfg: TrainConfig) -> None:
    enable_tf32()
    set_seed(cfg.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb" if cfg.wandb_project else None,
        kwargs_handlers=[ddp_kwargs],
    )
    if cfg.wandb_project and accelerator.is_main_process:
        accelerator.init_trackers(
            cfg.wandb_project,
            config=cfg.__dict__ | {"stage": cfg.stage},
            init_kwargs={"wandb": {"name": cfg.wandb_run_name or f"shinra-{cfg.stage}"}},
        )

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.tokenizer_path), use_fast=True)
    dataset = PackedParquetDataset(cfg.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=cfg.dataloader_num_workers > 0,
        prefetch_factor=4 if cfg.dataloader_num_workers > 0 else None,
        collate_fn=collate_lm,
        drop_last=True,
    )
    model = load_model(cfg)
    optimizer = build_optimizer(model, cfg.learning_rate, cfg.weight_decay, cfg.betas, cfg.eps)
    scheduler = build_scheduler(
        optimizer, cfg.scheduler, cfg.warmup_steps, cfg.max_steps, cfg.stable_ratio, cfg.min_lr_ratio
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    tokens = 0
    running = 0.0
    t0 = time.time()
    model.train()
    iterator = iter(loader)
    progress = tqdm(total=cfg.max_steps, disable=not accelerator.is_main_process, desc=cfg.stage)
    while step < cfg.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        with accelerator.accumulate(model):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = outputs.loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if accelerator.sync_gradients:
            step += 1
            gathered = accelerator.gather(loss.detach())
            loss_val = float(gathered.mean().item())
            running += loss_val
            tokens += cfg.micro_batch_size * cfg.sequence_length * accelerator.num_processes * cfg.gradient_accumulation_steps
            progress.update(1)
            if step % cfg.logging_steps == 0 and accelerator.is_main_process:
                elapsed = max(time.time() - t0, 1e-6)
                tps = tokens / elapsed
                avg = running / cfg.logging_steps
                ppl = math.exp(min(avg, 20))
                lr = scheduler.get_last_lr()[0]
                metrics = {
                    "loss": avg,
                    "ppl": ppl,
                    "lr": lr,
                    "tokens": tokens,
                    "tokens_per_sec": tps,
                    "step": step,
                }
                accelerator.log(metrics, step=step)
                progress.set_postfix(loss=f"{avg:.4f}", ppl=f"{ppl:.2f}", tps=f"{tps:.0f}")
                running = 0.0
            if step % cfg.save_steps == 0:
                save_checkpoint(accelerator, model, tokenizer, cfg.output_dir, step)
    save_checkpoint(accelerator, model, tokenizer, cfg.output_dir / "final", cfg.max_steps)
    accelerator.wait_for_everyone()
    accelerator.end_training()
