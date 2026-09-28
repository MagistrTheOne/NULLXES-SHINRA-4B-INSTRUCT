"""Accelerate + FSDP training loop for SHINRA pretrain and SFT."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer

from data.ledger import RunLedger
from data.shard_lifecycle import assert_under_ceiling
from data.shards import count_label_tokens
from model.configuration_shinra import ShinraConfig
from model.modeling_shinra import ShinraForCausalLM
from .arguments import TrainConfig
from .collator import collate_lm, load_packed_dataset
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
        document_end_token_id=int(raw.get("document_end_token_id", 18)),
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
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = model.to(dtype=dtype)
    model.config.use_cache = False
    model.config._attn_implementation = cfg.attention_implementation
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def load_trainer_state(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def last_token_entropy(logits: torch.Tensor) -> float:
    last = logits[:, -1].float()
    log_probs = F.log_softmax(last, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
    return float(entropy.detach().cpu())


def prune_checkpoints(output_dir: Path, keep_full: int = 1, keep_weights_only: int = 1) -> None:
    steps = sorted(p for p in output_dir.glob("step-*") if p.is_dir())
    if len(steps) <= keep_full + keep_weights_only:
        return
    newest = list(reversed(steps))
    keep = set(newest[: keep_full + keep_weights_only])
    for path in newest[keep_full : keep_full + keep_weights_only]:
        state = path / "trainer_state.pt"
        if state.exists():
            state.unlink()
    for path in steps:
        if path not in keep:
            shutil.rmtree(path, ignore_errors=True)


def save_checkpoint(
    accelerator: Accelerator,
    model: ShinraForCausalLM,
    tokenizer,
    output_dir: Path,
    step: int,
    *,
    optimizer=None,
    scheduler=None,
    consumed_tokens: int = 0,
    dataset_index: int = 0,
    extra: dict | None = None,
) -> Path:
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
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(ckpt_dir))
        unwrapped.config.save_pretrained(str(ckpt_dir))
        try:
            from scripts.export_hf_bundle import export_code

            export_code(ckpt_dir)
        except Exception:
            pass
        meta = {"step": step, "stage": getattr(model, "stage", None), "consumed_tokens": consumed_tokens}
        if extra:
            meta.update(extra)
        (ckpt_dir / "train_state.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        blob = {
            "step": step,
            "consumed_tokens": consumed_tokens,
            "dataset_index": dataset_index,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(blob, ckpt_dir / "trainer_state.pt")
        prune_checkpoints(output_dir)
    accelerator.wait_for_everyone()
    return ckpt_dir


@torch.no_grad()
def eval_heldout_ce(model, loader, max_batches: int = 32) -> dict:
    model.eval()
    losses: list[float] = []
    tokens = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        losses.append(float(outputs.loss.detach().cpu()))
        tokens += count_label_tokens(batch["labels"])
    model.train()
    if not losses:
        return {"eval_loss": None, "eval_tokens": 0}
    avg = sum(losses) / len(losses)
    return {"eval_loss": avg, "eval_ppl": math.exp(min(avg, 20)), "eval_tokens": tokens}


def run_lm_training(cfg: TrainConfig) -> None:
    enable_tf32()
    set_seed(cfg.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    mixed = "bf16" if torch.cuda.is_available() else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=mixed,
        log_with="wandb" if cfg.wandb_project else None,
        kwargs_handlers=[ddp_kwargs],
    )
    if cfg.wandb_project and accelerator.is_main_process:
        commit = "unknown"
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
            ).strip()
        except Exception:
            pass
        accelerator.init_trackers(
            cfg.wandb_project,
            config={
                **{k: str(v) for k, v in cfg.__dict__.items()},
                "stage": cfg.stage,
                "git_commit": commit,
                "config_path": str(cfg.model_config),
                "train_config": str(cfg.model_config),
            },
            init_kwargs={"wandb": {"name": cfg.wandb_run_name or f"shinra-{cfg.stage}"}},
        )

    tok_path = Path(cfg.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=True) if tok_path.exists() else None
    dataset = load_packed_dataset(cfg.data_dir)
    dataset_index = 0
    consumed = 0
    step = 0
    if cfg.resume_from:
        state_file = Path(cfg.resume_from) / "trainer_state.pt"
        if state_file.exists():
            blob = load_trainer_state(state_file)
            dataset_index = int(blob.get("dataset_index", 0))
            consumed = int(blob.get("consumed_tokens", 0))
            step = int(blob.get("step", 0))
    if dataset_index > 0:
        dataset = Subset(dataset, list(range(dataset_index, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.dataloader_num_workers > 0,
        prefetch_factor=4 if cfg.dataloader_num_workers > 0 else None,
        collate_fn=collate_lm,
        drop_last=False,
    )
    eval_loader = None
    if cfg.heldout_dir and Path(cfg.heldout_dir).exists() and any(Path(cfg.heldout_dir).glob("*.bin")):
        eval_loader = DataLoader(
            load_packed_dataset(cfg.heldout_dir),
            batch_size=cfg.micro_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_lm,
        )
    model = load_model(cfg)
    optimizer = build_optimizer(model, cfg.learning_rate, cfg.weight_decay, cfg.betas, cfg.eps)
    scheduler = build_scheduler(
        optimizer, cfg.scheduler, cfg.warmup_steps, cfg.max_steps, cfg.stable_ratio, cfg.min_lr_ratio
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    if eval_loader is not None:
        eval_loader = accelerator.prepare(eval_loader)
    if cfg.resume_from:
        state_file = Path(cfg.resume_from) / "trainer_state.pt"
        if state_file.exists():
            blob = load_trainer_state(state_file)
            if blob.get("optimizer") is not None:
                optimizer.load_state_dict(blob["optimizer"])
            if blob.get("scheduler") is not None:
                scheduler.load_state_dict(blob["scheduler"])
            if blob.get("rng") is not None:
                torch.set_rng_state(blob["rng"])

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(cfg.run_dir)
    running = 0.0
    running_grad = 0.0
    t0 = time.time()
    model.train()
    iterator = iter(loader)
    remaining_steps = max(cfg.max_steps - step, 0)
    progress = tqdm(total=cfg.max_steps, initial=step, disable=not accelerator.is_main_process, desc=cfg.stage)
    exhausted = False
    while step < cfg.max_steps and (cfg.max_tokens <= 0 or consumed < cfg.max_tokens):
        try:
            batch = next(iterator)
            dataset_index += 1
        except StopIteration:
            if cfg.allow_replay:
                iterator = iter(loader)
                try:
                    batch = next(iterator)
                    dataset_index += 1
                except StopIteration:
                    exhausted = True
                    break
            else:
                exhausted = True
                break
        grad_norm = None
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
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        batch_tokens = count_label_tokens(batch["labels"])
        consumed += batch_tokens
        if accelerator.sync_gradients:
            step += 1
            gathered = accelerator.gather(loss.detach())
            loss_val = float(gathered.mean().item())
            running += loss_val
            if grad_norm is not None:
                running_grad += float(grad_norm.detach() if hasattr(grad_norm, "detach") else grad_norm)
            progress.update(1)
            if step % cfg.logging_steps == 0 and accelerator.is_main_process:
                elapsed = max(time.time() - t0, 1e-6)
                tps = consumed / elapsed
                avg = running / cfg.logging_steps
                ppl = math.exp(min(avg, 20))
                lr = scheduler.get_last_lr()[0]
                nan_flag = not math.isfinite(avg)
                entropy = last_token_entropy(outputs.logits)
                metrics = {
                    "loss": avg,
                    "ppl": ppl,
                    "lr": lr,
                    "consumed_tokens": consumed,
                    "tokens_per_sec": tps,
                    "step": step,
                    "nan": int(nan_flag),
                    "grad_norm": running_grad / cfg.logging_steps,
                    "entropy": entropy,
                    "stage": cfg.stage,
                }
                ledger.log_metrics(metrics)
                ledger.write_status(
                    {
                        "stage": cfg.stage,
                        "step": step,
                        "consumed_tokens": consumed,
                        "remaining_tokens": max(cfg.max_tokens - consumed, 0) if cfg.max_tokens > 0 else None,
                        "tokens_per_sec": tps,
                        "loss": avg,
                        "lr": lr,
                        "ppl": ppl,
                        "nan": nan_flag,
                        "exhausted": exhausted,
                    }
                )
                accelerator.log(metrics, step=step)
                progress.set_postfix(loss=f"{avg:.4f}", ppl=f"{ppl:.2f}", tps=f"{tps:.0f}")
                running = 0.0
                running_grad = 0.0
                try:
                    assert_under_ceiling(cfg.output_dir, cfg.disk_ceiling_gb)
                except Exception as exc:
                    ledger.append({"event": "disk_ceiling", "error": str(exc)})
                    raise
            if cfg.eval_steps > 0 and step % cfg.eval_steps == 0 and eval_loader is not None and accelerator.is_main_process:
                ev = eval_heldout_ce(model, eval_loader)
                ledger.log_metrics({"step": step, **ev})
                ledger.append({"event": "eval", "step": step, **ev})
            if cfg.save_steps > 0 and step % cfg.save_steps == 0:
                save_checkpoint(
                    accelerator,
                    model,
                    tokenizer,
                    cfg.output_dir,
                    step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    consumed_tokens=consumed,
                    dataset_index=dataset_index,
                )
    save_checkpoint(
        accelerator,
        model,
        tokenizer,
        cfg.output_dir / "final",
        step,
        optimizer=optimizer,
        scheduler=scheduler,
        consumed_tokens=consumed,
        dataset_index=dataset_index,
        extra={"exhausted": exhausted},
    )
    if accelerator.is_main_process:
        ledger.write_status(
            {
                "stage": cfg.stage,
                "step": step,
                "consumed_tokens": consumed,
                "remaining_tokens": max(cfg.max_tokens - consumed, 0) if cfg.max_tokens > 0 else None,
                "exhausted": exhausted,
                "done": True,
            }
        )
        ledger.append({"event": "train_done", "step": step, "consumed_tokens": consumed, "exhausted": exhausted})
    accelerator.wait_for_everyone()
    accelerator.end_training()
