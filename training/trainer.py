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
from data.packed_data import ShardIdentityError, count_target_tokens_post_shift
from data.run_ledger import RunLedger
from model.configuration_shinra import ShinraConfig
from model.modeling_shinra import ShinraForCausalLM
from .arguments import TrainConfig
from .collator import collate_lm, load_packed_dataset
from .optim import build_optimizer
from .parallel import enable_tf32
from .schedule import build_scheduler
from .sampler import StatefulRandomSampler
from tokenizer import load_shinra_tokenizer


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
    model_config = config_from_yaml(cfg.model_config, cfg.attention_implementation)
    dtype = torch.bfloat16
    if cfg.resume_from:
        model = ShinraForCausalLM.from_pretrained(
            str(cfg.resume_from), config=model_config, dtype=dtype, local_files_only=True
        )
    else:
        model = ShinraForCausalLM(model_config)
        model.to(device="cuda", dtype=dtype)
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def estimate_training_memory(model: dict, training: dict) -> dict[str, int]:
    """Conservative single-GPU estimate made before model parameter allocation."""
    hidden = int(model["hidden_size"])
    q_width = int(model["num_attention_heads"]) * int(model["head_dim"])
    kv_width = int(model["num_key_value_heads"]) * int(model["head_dim"])
    intermediate = int(model["intermediate_size"])
    layers = int(model["num_hidden_layers"])
    attention = hidden * (q_width + 2 * kv_width) + q_width * hidden
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden + 2 * int(model["head_dim"])
    if model.get("attention_bias"):
        attention += q_width + 2 * kv_width + hidden
    if model.get("mlp_bias"):
        mlp += 2 * intermediate + hidden
    parameters = int(model["vocab_size"]) * hidden + layers * (attention + mlp + norms) + hidden
    parameter_and_optimizer_bytes = parameters * (2 + 2 + 4 + 4)
    activation_bytes = (2 * layers * int(training["micro_batch_size"])
                        * int(training["sequence_length"]) * hidden * 2)
    workspace_bytes = 2 * 1024**3
    return {
        "parameter_count_estimate": parameters,
        "weights_gradients_adam_states_bytes": parameter_and_optimizer_bytes,
        "activation_bytes_estimate": activation_bytes,
        "workspace_reserve_bytes": workspace_bytes,
        "peak_bytes_estimate": parameter_and_optimizer_bytes + activation_bytes + workspace_bytes,
    }


def save_checkpoint(accelerator: Accelerator, model, tokenizer, output_dir: Path, *, step: int, optimizer,
                    scheduler, consumed_tokens: int, sampler_state: dict, manifest: dict, stage: str,
                    objective: str, training_config_sha256: str) -> Path:
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
                "step": step, "consumed_tokens": consumed_tokens, "sampler": sampler_state,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "stage": stage,
                "objective": objective, "training_config_sha256": training_config_sha256,
                "architecture_sha256": _sha(manifest["architecture"]),
                "tokenizer_sha256": manifest["tokenizer_sha256"],
                "data_manifest_sha256": manifest["data_manifest_sha256"],
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            }
            torch.save(state, temporary / "trainer_state.pt")
            details = {k: v for k, v in state.items() if k not in {"optimizer", "scheduler", "rng", "cuda_rng", "python_rng", "numpy_rng", "sampler"}}
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
def evaluate(model, loader, max_batches: int | None = None) -> dict:
    model.eval()
    total_loss = 0.0
    tokens = 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
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
    if not Path("/content").exists():
        raise RuntimeError("training is restricted to the approved Google Colab runtime")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires an explicitly approved accelerator; CPU training is disabled")
    device_name = torch.cuda.get_device_name(0)
    if not any(gpu.casefold() in device_name.casefold() for gpu in approved["runtime"]["allowed_gpu_families"]):
        raise RuntimeError(f"unsupported GPU: {device_name}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required by configs/shinra.yaml but unsupported on this GPU")
    disk_root = cfg.output_dir.parent if cfg.output_dir.parent.exists() else Path.cwd()
    free_bytes, _ = shutil.disk_usage(disk_root)
    if free_bytes < int(approved["runtime"]["minimum_free_disk_gb"] * 1024**3):
        raise RuntimeError("insufficient free disk before model allocation")
    device_properties = torch.cuda.get_device_properties(0)
    total_gpu_bytes = int(device_properties.total_memory)
    free_gpu_bytes, _ = torch.cuda.mem_get_info(0)
    if "A100" in device_name.upper():
        memory_gb = round(total_gpu_bytes / 1_000_000_000)
        if memory_gb not in approved["runtime"]["allowed_a100_memory_gb"]:
            raise RuntimeError(f"A100 memory size is not approved: {memory_gb} GB")
    memory_estimate = estimate_training_memory(approved["model"], approved["training"])
    headroom = float(approved["precision"]["memory"]["allocator_headroom_ratio"])
    required_free = int(memory_estimate["peak_bytes_estimate"] / (1 - headroom))
    preflight = {
        "gpu_name": device_name,
        "gpu_total_bytes": total_gpu_bytes,
        "gpu_free_bytes_before_allocation": int(free_gpu_bytes),
        "disk_free_bytes": int(free_bytes),
        **memory_estimate,
        "required_free_bytes_with_headroom": required_free,
    }
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if free_gpu_bytes < required_free:
        raise RuntimeError(
            f"GPU memory preflight failed: need {required_free / 1e9:.1f} GB free, "
            f"have {free_gpu_bytes / 1e9:.1f} GB; see {cfg.run_dir / 'preflight.json'}"
        )
    corpus_root = cfg.data_dir
    data_manifest_path = corpus_root / "manifest.json"
    if not data_manifest_path.is_file():
        raise FileNotFoundError("training data manifest.json is required")
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
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
    accelerator = Accelerator(gradient_accumulation_steps=cfg.gradient_accumulation_steps, mixed_precision="bf16")
    if accelerator.num_processes != 1:
        raise RuntimeError("multi-process training is unavailable; use one approved GPU per run")
    tokenizer_file = cfg.tokenizer_path / "tokenizer.json"
    if hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() != approved["tokenizer"]["artifact_sha256"]:
        raise ShardIdentityError("tokenizer artifact hash does not match configs/shinra.yaml")
    training_config_sha = _sha(approved["training"])
    resume_state = None
    if cfg.resume_from:
        state_file = cfg.resume_from / "trainer_state.pt"
        if not state_file.is_file():
            raise FileNotFoundError("resume checkpoint is missing trainer_state.pt")
        resume_state = torch.load(state_file, map_location="cpu", weights_only=False)
        expected_state = {
            "architecture_sha256": _sha(approved["model"]),
            "tokenizer_sha256": approved["tokenizer"]["artifact_sha256"],
            "data_manifest_sha256": manifest_sha,
            "objective": approved["training"]["objective"],
            "stage": cfg.stage,
            "training_config_sha256": training_config_sha,
        }
        mismatched = [key for key, value in expected_state.items() if resume_state.get(key) != value]
        if mismatched:
            raise ShardIdentityError(f"resume checkpoint identity mismatch: {mismatched}")
        if int(resume_state["consumed_tokens"]) >= cfg.max_tokens or int(resume_state["step"]) >= cfg.max_steps:
            raise RuntimeError("resume checkpoint has already reached the configured diagnostic limit")
    sampler = StatefulRandomSampler(train_data, cfg.seed)
    if resume_state:
        sampler.load_state_dict(resume_state["sampler"])
    loader = DataLoader(train_data, batch_size=cfg.micro_batch_size, sampler=sampler,
                        num_workers=cfg.dataloader_num_workers, pin_memory=True,
                        collate_fn=collate_lm, drop_last=False)
    validation_loader = DataLoader(validation_data, batch_size=cfg.micro_batch_size, shuffle=False,
                                   num_workers=0, collate_fn=collate_lm)
    enable_tf32()
    set_seed(cfg.seed)
    model = load_model(cfg)
    optimizer = build_optimizer(model, cfg.learning_rate, cfg.weight_decay, cfg.betas, cfg.eps,
                                fused=cfg.fused_optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, cfg.warmup_steps, cfg.max_steps, cfg.stable_ratio, cfg.min_lr_ratio)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    validation_loader = accelerator.prepare(validation_loader)
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        torch.set_rng_state(resume_state["rng"])
        if torch.cuda.is_available() and resume_state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        random.setstate(resume_state["python_rng"])
        np.random.set_state(resume_state["numpy_rng"])
    tokenizer = load_shinra_tokenizer(cfg.tokenizer_path)
    manifest = {"architecture": approved["model"], "tokenizer_sha256": approved["tokenizer"]["artifact_sha256"],
                "data_manifest_sha256": manifest_sha}
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(cfg.run_dir)
    model.train()
    step = int(resume_state["step"]) if resume_state else 0
    consumed = int(resume_state["consumed_tokens"]) if resume_state else 0
    ledger.write_status({"status": "running", "step": step, "processed_tokens": consumed,
                         "optimizer_steps": step, "checkpoint": str(cfg.resume_from) if cfg.resume_from else None})
    iterator = iter(loader)
    started = time.monotonic()
    window_losses: list[float] = []
    accumulation_loss_sum = 0.0
    accumulation_tokens = 0
    progress = tqdm(total=cfg.max_steps, initial=step, disable=not accelerator.is_local_main_process, desc="SHINRA pretrain")
    last_checkpoint = None
    initial_consumed_tokens = consumed
    while step < cfg.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            raise RuntimeError("training shard exhausted before configured diagnostic token budget")
        with accelerator.accumulate(model):
            result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], use_cache=False)
            if not torch.isfinite(result.loss):
                raise FloatingPointError("non-finite training loss")
            target_tokens_this_batch = count_target_tokens_post_shift(batch["labels"])
            if target_tokens_this_batch <= 0:
                raise RuntimeError("training batch has no valid next-token targets")
            if torch.any(batch["labels"][:, 0] != -100):
                raise RuntimeError("the first token of each isolated sequence must be masked from shifted loss")
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
        batch_tokens = count_target_tokens_post_shift(batch["labels"])
        consumed += batch_tokens
        accumulation_loss_sum += float(result.loss.detach()) * batch_tokens
        accumulation_tokens += batch_tokens
        if accelerator.sync_gradients:
            window_losses.append(accumulation_loss_sum / max(accumulation_tokens, 1))
            accumulation_loss_sum = 0.0
            accumulation_tokens = 0
        if step and step % cfg.logging_steps == 0 and accelerator.sync_gradients:
            elapsed = max(time.monotonic() - started, 1e-6)
            metrics = {"step": step, "processed_tokens": consumed, "optimizer_steps": step,
                       "loss": sum(window_losses[-cfg.logging_steps:]) / min(len(window_losses), cfg.logging_steps),
                        "tokens_per_second": (consumed - initial_consumed_tokens) / elapsed,
                        "learning_rate": scheduler.get_last_lr()[0],
                       "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}
            ledger.log_metrics(metrics)
            ledger.write_status(metrics)
        if cfg.eval_steps and step and step % cfg.eval_steps == 0 and accelerator.sync_gradients:
            metrics = evaluate(accelerator.unwrap_model(model), validation_loader)
            ledger.log_metrics({"step": step, **metrics})
        if cfg.save_steps and step and step % cfg.save_steps == 0 and accelerator.sync_gradients:
            last_checkpoint = save_checkpoint(accelerator, model, tokenizer, cfg.output_dir, step=step, optimizer=optimizer,
                            scheduler=scheduler, consumed_tokens=consumed, sampler_state=sampler.state_dict(),
                            manifest=manifest, stage=cfg.stage, objective=approved["training"]["objective"],
                            training_config_sha256=training_config_sha)
        if accelerator.sync_gradients and consumed >= cfg.max_tokens:
            break
    progress.close()
    final_eval = evaluate(accelerator.unwrap_model(model), validation_loader)
    if last_checkpoint is not None and last_checkpoint.name == f"step-{step:08d}":
        checkpoint = last_checkpoint
    else:
        checkpoint = save_checkpoint(
            accelerator, model, tokenizer, cfg.output_dir, step=step, optimizer=optimizer,
            scheduler=scheduler, consumed_tokens=consumed, sampler_state=sampler.state_dict(),
            manifest=manifest, stage=cfg.stage, objective=approved["training"]["objective"],
            training_config_sha256=training_config_sha,
        )
    result = {"step": step, "processed_tokens": consumed, "checkpoint": str(checkpoint), **final_eval}
    ledger.write_status({"status": "complete", **result})
    ledger.append({"event": "training_complete", **result})
    accelerator.end_training()
    return result
