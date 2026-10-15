"""Single random-init causal pretraining engine for SHINRA."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.loader import load_config
from data.packed_data import ShardIdentityError, count_label_tokens, count_target_tokens_post_shift
from data.run_ledger import RunLedger
from model.configuration_shinra import ShinraConfig
from model.modeling_shinra import ShinraForCausalLM
from .arguments import TrainConfig
from .collator import collate_lm, load_packed_dataset
from .optim import build_optimizer
from .parallel import enable_tf32
from .schedule import build_scheduler


def config_from_yaml(path: Path, attention_implementation: str | None = None) -> ShinraConfig:
    approved = load_config(path)
    raw = approved["model"]
    kwargs = {key: value for key, value in raw.items() if key in {
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "hidden_act", "max_position_embeddings", "initializer_range",
        "rms_norm_eps", "pad_token_id", "bos_token_id", "eos_token_id", "tie_word_embeddings", "rope_theta",
        "rope_scaling", "attention_bias", "attention_dropout", "residual_dropout", "embedding_dropout",
        "mlp_bias", "qk_norm", "sliding_window", "layer_types", "z_loss_coefficient", "document_end_token_id",
    }}
    kwargs["attention_implementation"] = attention_implementation or raw["attention_implementation"]
    kwargs["use_cache"] = False
    config = ShinraConfig(**kwargs)
    config._attn_implementation = kwargs["attention_implementation"]
    return config


def load_model(cfg: TrainConfig) -> ShinraForCausalLM:
    if cfg.resume_from:
        raise SystemExit("resume is gated until a real immutable data manifest and tokenizer SHA are frozen")
    model = ShinraForCausalLM(config_from_yaml(cfg.model_config, cfg.attention_implementation))
    dtype = torch.bfloat16 if torch.cuda.is_available() and not cfg.force_cpu else torch.float32
    model.to(device="cuda", dtype=dtype)
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def save_checkpoint(accelerator: Accelerator, model, tokenizer, output_dir: Path, *, step: int, optimizer,
                    scheduler, consumed_tokens: int, dataset_index: int, manifest: dict, stage: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / f"step-{step:08d}"
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    temporary = Path(tempfile.mkdtemp(prefix=f".step-{step:08d}-", dir=output_dir))
    try:
        unwrapped = accelerator.unwrap_model(model)
        accelerator.save_model(unwrapped, str(temporary))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(temporary)
            unwrapped.config.save_pretrained(temporary)
            from scripts.export_hf_bundle import export_code

            export_code(temporary)
            state = {
                "step": step, "consumed_tokens": consumed_tokens, "dataset_index": dataset_index,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "stage": stage,
                "architecture_sha256": _sha(manifest["architecture"]),
                "tokenizer_sha256": manifest["tokenizer_sha256"],
                "data_manifest_sha256": manifest["data_manifest_sha256"],
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "dataloader_generator_state": manifest["dataloader_generator_state"],
            }
            torch.save(state, temporary / "trainer_state.pt")
            details = {k: v for k, v in state.items() if k not in {"optimizer", "scheduler", "rng", "cuda_rng", "python_rng", "numpy_rng", "dataloader_generator_state"}}
            (temporary / "train_state.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            # Rename within the same filesystem; the visible checkpoint is complete or absent.
            os.replace(temporary, final)
        accelerator.wait_for_everyone()
    except Exception:
        if accelerator.is_main_process and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


@torch.no_grad()
def evaluate(model, loader, max_batches: int = 32) -> dict:
    model.eval()
    total_loss = 0.0
    tokens = 0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        n_tokens = count_target_tokens_post_shift(batch["labels"])
        if not n_tokens:
            continue
        result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], use_cache=False)
        if not torch.isfinite(result.loss):
            raise FloatingPointError("non-finite validation loss")
        total_loss += float(result.loss) * n_tokens
        tokens += n_tokens
    model.train()
    if tokens == 0:
        raise ValueError("held-out split produced zero target tokens")
    ce = total_loss / tokens
    return {"heldout_loss": ce, "heldout_perplexity": math.exp(min(ce, 20)), "heldout_tokens": tokens}


def run_lm_training(cfg: TrainConfig) -> dict:
    approved = load_config(cfg.model_config, require_ready=True)
    if cfg.stage != "pretrain":
        raise SystemExit("only causal pretraining is active; instruction tuning is a future phase")
    if cfg.resume_from:
        raise SystemExit("resume is blocked until immutable corpus and tokenizer hashes are present")
    if cfg.force_cpu or not torch.cuda.is_available():
        raise RuntimeError("training requires an explicitly approved accelerator; CPU training is disabled")
    device_name = torch.cuda.get_device_name(0)
    if not any(gpu in device_name for gpu in approved["runtime"]["allowed_gpu_families"]):
        raise RuntimeError(f"unsupported GPU: {device_name}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required by configs/shinra.yaml but unsupported on this GPU")
    disk_root = cfg.output_dir.parent if cfg.output_dir.parent.exists() else Path.cwd()
    free_bytes, _ = shutil.disk_usage(disk_root)
    if free_bytes < int(approved["runtime"]["minimum_free_disk_gb"] * 1024**3):
        raise RuntimeError("insufficient free disk before model allocation")
    corpus_root = cfg.data_dir
    data_manifest_path = corpus_root / "manifest.json"
    if not data_manifest_path.is_file():
        raise FileNotFoundError("training data manifest.json is required")
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    root_manifest_sha = hashlib.sha256(data_manifest_path.read_bytes()).hexdigest()
    manifest_without_self_hash = dict(data_manifest)
    manifest_sha = manifest_without_self_hash.pop("manifest_sha256", None)
    expected_manifest_sha = hashlib.sha256(json.dumps(manifest_without_self_hash, sort_keys=True).encode()).hexdigest()
    if manifest_sha != expected_manifest_sha:
        raise ShardIdentityError("corpus manifest self-hash is invalid")
    if approved["data"].get("manifest_sha256") != manifest_sha:
        raise ShardIdentityError("dataset manifest SHA256 does not match configs/shinra.yaml")
    if not approved["tokenizer"].get("artifact_sha256"):
        raise RuntimeError("tokenizer artifact SHA256 is not frozen")
    heldout_dir = cfg.heldout_dir or cfg.data_dir / "validation"
    if not heldout_dir.is_dir():
        raise FileNotFoundError("held-out validation data is required")
    train_dir = cfg.data_dir / "train"
    train_data = load_packed_dataset(train_dir)
    validation_data = load_packed_dataset(heldout_dir)
    if not data_manifest.get("train_hash") or not data_manifest.get("validation_hash") or not data_manifest.get("blind_hash"):
        raise ShardIdentityError("manifest missing frozen train/validation/blind split hashes")
    if len({data_manifest["train_hash"], data_manifest["validation_hash"], data_manifest["blind_hash"]}) != 3:
        raise ShardIdentityError("train/evaluation split hashes overlap")
    if getattr(train_data, "source_manifest_sha256", None) != manifest_sha:
        raise ShardIdentityError("packed train source manifest differs from frozen manifest")
    if getattr(validation_data, "source_manifest_sha256", None) != manifest_sha:
        raise ShardIdentityError("packed validation source manifest differs from frozen manifest")
    dataset_generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(train_data, batch_size=cfg.micro_batch_size, shuffle=True,
                        num_workers=cfg.dataloader_num_workers, pin_memory=True, generator=dataset_generator,
                        collate_fn=collate_lm, drop_last=False)
    validation_loader = DataLoader(validation_data, batch_size=cfg.micro_batch_size, shuffle=False,
                                   num_workers=0, collate_fn=collate_lm)
    enable_tf32()
    set_seed(cfg.seed)
    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps, mixed_precision="bf16")
    model = load_model(cfg)
    optimizer = build_optimizer(model, cfg.learning_rate, cfg.weight_decay, cfg.betas, cfg.eps,
                                fused=cfg.fused_optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, cfg.warmup_steps, cfg.max_steps, cfg.stable_ratio, cfg.min_lr_ratio)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    validation_loader = accelerator.prepare(validation_loader)
    tokenizer = __import__("transformers", fromlist=["AutoTokenizer"]).AutoTokenizer.from_pretrained(cfg.tokenizer_path, use_fast=True)
    manifest = {"architecture": approved["model"], "tokenizer_sha256": approved["tokenizer"]["artifact_sha256"],
                "data_manifest_sha256": manifest_sha,
                "dataloader_generator_state": dataset_generator.get_state()}
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(cfg.run_dir)
    model.train()
    step = 0
    consumed = 0
    iterator = iter(loader)
    started = time.monotonic()
    window_losses: list[float] = []
    progress = tqdm(total=cfg.max_steps, disable=not accelerator.is_local_main_process, desc="SHINRA pretrain")
    while step < cfg.max_steps and consumed < cfg.max_tokens:
        try:
            batch = next(iterator)
        except StopIteration:
            raise RuntimeError("training shard exhausted before configured diagnostic token budget")
        with accelerator.accumulate(model):
            result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], use_cache=False)
            if not torch.isfinite(result.loss):
                raise FloatingPointError("non-finite training loss")
            accelerator.backward(result.loss)
            if accelerator.sync_gradients:
                norm = accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite gradient norm")
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("non-finite gradient")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                progress.update(1)
        batch_tokens = count_label_tokens(batch["labels"])
        consumed += batch_tokens
        window_losses.append(float(result.loss.detach()))
        if step and step % cfg.logging_steps == 0 and accelerator.sync_gradients:
            elapsed = max(time.monotonic() - started, 1e-6)
            metrics = {"step": step, "processed_tokens": consumed, "optimizer_steps": step,
                       "loss": sum(window_losses[-cfg.logging_steps:]) / min(len(window_losses), cfg.logging_steps),
                       "tokens_per_second": consumed / elapsed, "learning_rate": scheduler.get_last_lr()[0],
                       "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}
            ledger.log_metrics(metrics)
            ledger.write_status(metrics)
        if cfg.eval_steps and step and step % cfg.eval_steps == 0 and accelerator.sync_gradients:
            metrics = evaluate(accelerator.unwrap_model(model), validation_loader)
            ledger.log_metrics({"step": step, **metrics})
        if cfg.save_steps and step and step % cfg.save_steps == 0 and accelerator.sync_gradients:
            manifest["dataloader_generator_state"] = dataset_generator.get_state()
            save_checkpoint(accelerator, model, tokenizer, cfg.output_dir, step=step, optimizer=optimizer,
                            scheduler=scheduler, consumed_tokens=consumed, dataset_index=consumed,
                            manifest=manifest, stage=cfg.stage)
    progress.close()
    final_eval = evaluate(accelerator.unwrap_model(model), validation_loader)
    manifest["dataloader_generator_state"] = dataset_generator.get_state()
    checkpoint = save_checkpoint(accelerator, model, tokenizer, cfg.output_dir, step=step, optimizer=optimizer,
                                 scheduler=scheduler, consumed_tokens=consumed, dataset_index=consumed,
                                 manifest=manifest, stage=cfg.stage)
    result = {"step": step, "processed_tokens": consumed, "checkpoint": str(checkpoint), **final_eval}
    ledger.write_status({"status": "complete", **result})
    ledger.append({"event": "training_complete", **result})
    accelerator.end_training()
    return result
