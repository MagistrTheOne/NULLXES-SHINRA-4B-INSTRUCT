"""Single random-init causal pretraining engine for SHINRA."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.loader import load_config
from data.hashing import sha256_file
from data.packed_data import ShardIdentityError, count_target_tokens_post_shift
from data.run_ledger import RunLedger
from model.configuration_shinra import ShinraConfig
from model.modeling_shinra import ShinraForCausalLM
from .arguments import TrainConfig
from .collator import collate_lm, load_packed_dataset
from .optim import build_optimizer
from .parallel import enable_tf32
from .schedule import build_scheduler
from .sampler import StatefulTokenMixSampler
from scripts.checkpoint_admission import checkpoint_admission
from tokenizer import load_shinra_tokenizer
from tokenizer.bundle import copy_bundle, verify_bundle


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


def verify_training_checkpoint(path: Path) -> dict:
    manifest_path = path / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise ShardIdentityError(f"{path}: checkpoint integrity manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShardIdentityError(f"{path}: checkpoint integrity manifest is unreadable: {exc}") from exc
    declared = manifest.pop("manifest_sha256", None)
    if manifest.get("schema") != "nullxes.shinra.training-checkpoint.v1" or declared != _sha(manifest):
        raise ShardIdentityError(f"{path}: checkpoint manifest schema or self-hash is invalid")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ShardIdentityError(f"{path}: checkpoint file checksum map is missing")
    actual_files = {}
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ShardIdentityError(f"{path}: symlinks are forbidden in a training checkpoint")
        if item.is_file() and item != manifest_path:
            actual_files[item.relative_to(path).as_posix()] = sha256_file(item)
    if actual_files != expected_files:
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        mismatched = sorted(name for name in set(actual_files) & set(expected_files)
                            if actual_files[name] != expected_files[name])
        raise ShardIdentityError(
            f"{path}: checkpoint files differ from manifest; missing={missing}, extra={extra}, sha_mismatch={mismatched}"
        )
    try:
        manifest["admission"] = checkpoint_admission(path)
    except ValueError as exc:
        raise ShardIdentityError(str(exc)) from exc
    return manifest


def estimate_training_memory(model: dict, training: dict) -> dict[str, int]:
    """Pre-allocation estimate including full-precision optimizer and LM-loss working storage."""
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
    # BF16 weights/gradients, FP32 Adam moments, plus a conservative FP32 update-temporary allowance.
    parameter_and_optimizer_bytes = parameters * (2 + 2 + 4 + 4 + 4)
    activation_bytes = (2 * layers * int(training["micro_batch_size"])
                        * int(training["sequence_length"]) * hidden * 2)
    # The loss path materializes FP32 logits and CE intermediates for every sequence position.
    loss_logits_bytes = (int(training["micro_batch_size"]) * int(training["sequence_length"])
                         * int(model["vocab_size"]) * 12)
    workspace_bytes = 4 * 1024**3
    return {
        "parameter_count_estimate": parameters,
        "parameter_optimizer_and_update_peak_bytes": parameter_and_optimizer_bytes,
        "activation_bytes_estimate": activation_bytes,
        "loss_logits_working_bytes_estimate": loss_logits_bytes,
        "workspace_reserve_bytes": workspace_bytes,
        "peak_bytes_estimate": parameter_and_optimizer_bytes + activation_bytes + loss_logits_bytes + workspace_bytes,
    }


def save_checkpoint(accelerator: Accelerator, model, tokenizer, output_dir: Path, *, step: int, optimizer,
                    scheduler, consumed_tokens: int, sampler_state: dict, manifest: dict, stage: str,
                    budget_kind: str, target_tokens: int, objective: str, training_config_sha256: str,
                    admission: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / f"step-{step:08d}"
    if final.exists():
        raise FileExistsError(f"checkpoint already exists: {final}")
    temporary = Path(tempfile.mkdtemp(prefix=f".step-{step:08d}-", dir=output_dir))
    try:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.config.shinra_training_admission = admission
        accelerator.save_model(unwrapped, str(temporary))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            copy_bundle(tokenizer.shinra_bundle_path, temporary, tokenizer.shinra_config,
                        require_frozen=(admission == "production"))
            unwrapped.config.save_pretrained(temporary)
            from scripts.export_hf_bundle import export_code

            export_code(temporary)
            state = {
                "step": step, "consumed_tokens": consumed_tokens, "sampler": sampler_state,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "stage": stage,
                "admission": admission,
                "objective": objective, "budget_kind": budget_kind, "target_tokens": target_tokens,
                "training_config_sha256": training_config_sha256,
                "architecture_sha256": _sha(manifest["architecture"]),
                "tokenizer_bundle_sha256": manifest["tokenizer_bundle_sha256"],
                "data_manifest_sha256": manifest["data_manifest_sha256"],
                "packed_data_manifests_sha256": manifest["packed_data_manifests_sha256"],
                "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            }
            torch.save(state, temporary / "trainer_state.pt")
            admission_record = {"schema": "nullxes.shinra.training-admission.v1", "admission": admission,
                                "source_corpus_admission": manifest["source_corpus_admission"],
                                "production_usable": admission == "production"}
            (temporary / "training_admission.json").write_text(
                json.dumps(admission_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            details = {k: v for k, v in state.items() if k not in {"optimizer", "scheduler", "rng", "cuda_rng", "python_rng", "numpy_rng", "sampler"}}
            (temporary / "train_state.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            checkpoint_files = {
                item.relative_to(temporary).as_posix(): sha256_file(item)
                for item in sorted(temporary.rglob("*")) if item.is_file()
            }
            checkpoint_manifest = {"schema": "nullxes.shinra.training-checkpoint.v1", "step": step,
                                   "files": checkpoint_files}
            checkpoint_manifest["manifest_sha256"] = _sha(checkpoint_manifest)
            checkpoint_manifest_tmp = temporary / ".checkpoint_manifest.json.tmp"
            checkpoint_manifest_tmp.write_text(
                json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(checkpoint_manifest_tmp, temporary / "checkpoint_manifest.json")
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
        # Evaluation CE excludes the training-only z-loss regularizer.
        core_model = model.module if hasattr(model, "module") else model
        total_loss += core_model.compute_loss_parts(result.logits, batch["labels"])["train_ce"] * n_tokens
        tokens += n_tokens
    model.train()
    if tokens == 0:
        raise ValueError("held-out split produced zero target tokens")
    ce = total_loss / tokens
    return {"heldout_loss": ce, "heldout_perplexity": math.exp(min(ce, 20)), "heldout_tokens": tokens}


def _run_lm_training(cfg: TrainConfig) -> dict:
    diagnostic = cfg.admission == "diagnostic_audit"
    if cfg.admission not in {"production", "diagnostic_audit"}:
        raise RuntimeError(f"unsupported training admission: {cfg.admission}")
    approved = load_config(cfg.model_config, require_ready=not diagnostic)
    if diagnostic:
        if approved["status"] == "ready_for_training" or approved["training"].get("enabled") is not False \
                or approved["data"].get("training_enabled") is not False \
                or approved["runtime"].get("colab_launch_allowed") is not False:
            raise RuntimeError("diagnostic audit mode requires every production training gate to remain false")
        if any(source.get("status") != "audit_only" for source in approved["data"].get("sources", [])):
            raise RuntimeError("diagnostic audit mode requires audit-only configured sources")
        if cfg.budget_kind != "diagnostic" or cfg.max_tokens > 20_000_000:
            raise RuntimeError("diagnostic audit token budget must be positive and no greater than 20M")
        if "diagnostic" not in str(cfg.output_dir).casefold():
            raise RuntimeError("diagnostic checkpoint output path must include 'diagnostic'")
    bundle_sha = verify_bundle(cfg.tokenizer_path, approved, require_frozen=not diagnostic)
    if cfg.resume_from:
        resumed_bundle_sha = verify_bundle(cfg.resume_from, approved, require_frozen=not diagnostic)
        checkpoint_manifest = verify_training_checkpoint(cfg.resume_from)
        if checkpoint_manifest.get("admission") != cfg.admission:
            raise ShardIdentityError("resume checkpoint admission differs from the explicitly selected run mode")
        if resumed_bundle_sha != bundle_sha:
            raise ShardIdentityError("resume checkpoint Native tokenizer bundle differs from the selected bundle")
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
    free_bytes = shutil.disk_usage(disk_root).free
    if free_bytes < int(approved["runtime"]["minimum_free_disk_gb"] * 1024**3):
        raise RuntimeError("insufficient free disk before model allocation")
    device_properties = torch.cuda.get_device_properties(0)
    total_gpu_bytes = int(device_properties.total_memory)
    free_gpu_bytes, _ = torch.cuda.mem_get_info(0)
    if "A100" in device_name.upper():
        memory_gb = round(total_gpu_bytes / 1_000_000_000)
        if memory_gb not in approved["runtime"]["allowed_a100_memory_gb"]:
            raise RuntimeError(f"A100 memory size is not approved: {memory_gb} GB")
    memory_estimate = estimate_training_memory(approved["model"], {
        **approved["training"], "micro_batch_size": cfg.micro_batch_size, "sequence_length": cfg.sequence_length,
    })
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
    expected_corpus_admission = "audit_only" if diagnostic else "production"
    if data_manifest.get("purpose") != "pretraining" or data_manifest.get("admission") != expected_corpus_admission:
        raise ShardIdentityError(f"{cfg.admission} mode requires corpus admission {expected_corpus_admission!r}")
    manifest_without_self_hash = dict(data_manifest)
    manifest_sha = manifest_without_self_hash.pop("manifest_sha256", None)
    expected_manifest_sha = hashlib.sha256(json.dumps(manifest_without_self_hash, sort_keys=True).encode()).hexdigest()
    if manifest_sha != expected_manifest_sha:
        raise ShardIdentityError("corpus manifest self-hash is invalid")
    if approved["data"].get("manifest_sha256") != manifest_sha:
        raise ShardIdentityError("dataset manifest SHA256 does not match configs/shinra.yaml")
    if not diagnostic and not approved["tokenizer"].get("artifact_sha256"):
        raise RuntimeError("tokenizer artifact SHA256 is not frozen")
    heldout_dir = cfg.heldout_dir or cfg.data_dir / "validation"
    if not heldout_dir.is_dir():
        raise FileNotFoundError("held-out validation data is required")
    train_dir = cfg.data_dir / "train"
    train_data = load_packed_dataset(train_dir, config=approved, expected_split="train",
                                     allow_audit_data=diagnostic)
    validation_data = load_packed_dataset(heldout_dir, config=approved, expected_split="validation",
                                         allow_audit_data=diagnostic)
    if train_data.tokenizer_bundle_sha256 != bundle_sha or validation_data.tokenizer_bundle_sha256 != bundle_sha:
        raise ShardIdentityError("packed train/validation tokenizer bundle differs from the verified selected bundle")
    expected_packed_admission = "audit_only" if diagnostic else "production"
    if train_data.admission != expected_packed_admission or validation_data.admission != expected_packed_admission:
        raise ShardIdentityError(f"{cfg.admission} mode requires {expected_packed_admission!r} packed artifacts")
    packed_manifest_shas = {"train": train_data.packed_manifest_sha256,
                            "validation": validation_data.packed_manifest_sha256}
    packed_data_manifests_sha = _sha(packed_manifest_shas)
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
    if cfg.budget_kind not in {"diagnostic", "foundation"} or cfg.max_tokens <= 0:
        raise RuntimeError("a positive diagnostic or foundation token budget must be selected")
    if approved["data"]["mix"].get("en") != 0.5 or approved["data"]["mix"].get("ru") != 0.5:
        raise RuntimeError("the current production trainer implements the approved EN/RU token mix at 50/50 only")
    training_config_sha = _sha({"training": approved["training"], "budget_kind": cfg.budget_kind,
                                "target_tokens": cfg.max_tokens, "language_mix": approved["data"]["mix"],
                                "packed_data_manifests_sha256": packed_data_manifests_sha,
                                "admission": cfg.admission, "schedule_steps": cfg.schedule_steps})
    resume_state = None
    if cfg.resume_from:
        state_file = cfg.resume_from / "trainer_state.pt"
        if not state_file.is_file():
            raise FileNotFoundError("resume checkpoint is missing trainer_state.pt")
        resume_state = torch.load(state_file, map_location="cpu", weights_only=False)
        expected_state = {
            "architecture_sha256": _sha(approved["model"]),
            "tokenizer_bundle_sha256": bundle_sha,
            "data_manifest_sha256": manifest_sha,
            "packed_data_manifests_sha256": packed_data_manifests_sha,
            "admission": cfg.admission,
            "objective": approved["training"]["objective"],
            "stage": cfg.stage,
            "training_config_sha256": training_config_sha,
        }
        mismatched = [key for key, value in expected_state.items() if resume_state.get(key) != value]
        if mismatched:
            raise ShardIdentityError(f"resume checkpoint identity mismatch: {mismatched}")
        if resume_state.get("budget_kind") != cfg.budget_kind or int(resume_state.get("target_tokens", -1)) != cfg.max_tokens:
            raise ShardIdentityError("resume checkpoint uses a different diagnostic/foundation budget")
        if int(resume_state["consumed_tokens"]) >= cfg.max_tokens or int(resume_state["step"]) >= cfg.max_steps:
            raise RuntimeError("resume checkpoint has already reached the configured token or step limit")
    sampler = StatefulTokenMixSampler(train_data, cfg.seed, approved["data"]["mix"])
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
    actual_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameter_count != memory_estimate["parameter_count_estimate"]:
        raise RuntimeError(
            "instantiated SHINRA parameter count differs from the approved architecture estimate: "
            f"{actual_parameter_count} != {memory_estimate['parameter_count_estimate']}"
        )
    preflight["instantiated_parameter_count"] = actual_parameter_count
    (cfg.run_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    optimizer = build_optimizer(model, cfg.learning_rate, cfg.weight_decay, cfg.betas, cfg.eps,
                                fused=cfg.fused_optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, cfg.warmup_steps, cfg.schedule_steps,
                                cfg.stable_ratio, cfg.min_lr_ratio)
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
    tokenizer = load_shinra_tokenizer(cfg.tokenizer_path, approved, require_frozen_bundle=not diagnostic)
    manifest = {"architecture": approved["model"], "tokenizer_bundle_sha256": bundle_sha,
                "data_manifest_sha256": manifest_sha,
                "packed_data_manifests_sha256": packed_data_manifests_sha,
                "source_corpus_admission": data_manifest["admission"]}
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(cfg.run_dir)
    model.train()
    step = int(resume_state["step"]) if resume_state else 0
    consumed = int(resume_state["consumed_tokens"]) if resume_state else 0
    ledger.write_status({"status": "running", "admission": cfg.admission, "budget_kind": cfg.budget_kind,
                         "target_tokens": cfg.max_tokens,
                         "step": step, "processed_tokens": consumed,
                         "optimizer_steps": step, "checkpoint": str(cfg.resume_from) if cfg.resume_from else None})
    iterator = iter(loader)
    started = time.monotonic()
    window_losses: list[float] = []
    accumulation_loss_sum = 0.0
    accumulation_tokens = 0
    progress = tqdm(total=cfg.max_steps, initial=step, disable=not accelerator.is_local_main_process, desc="SHINRA pretrain")
    last_checkpoint = cfg.resume_from
    initial_consumed_tokens = consumed
    max_row_targets = int(train_data.target_tokens.max())
    while step < cfg.max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("training split is empty; configured exhaustion policy cannot cycle it") from exc
        with accelerator.accumulate(model):
            result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], use_cache=False)
            if not torch.isfinite(result.loss):
                raise FloatingPointError("non-finite training loss")
            target_tokens_this_batch = count_target_tokens_post_shift(batch["labels"])
            if target_tokens_this_batch <= 0:
                raise RuntimeError("training batch has no valid next-token targets")
            if torch.any(batch["labels"][:, 0] != -100):
                raise RuntimeError("the first token of each isolated sequence must be masked from shifted loss")
            # Accumulate summed token loss; after Accelerator's fixed accumulation scaling,
            # normalize by the actual number of supervised targets in this optimizer update.
            accumulation_tokens += target_tokens_this_batch
            accumulation_loss_sum += float(result.loss.detach()) * target_tokens_this_batch
            accelerator.backward(result.loss * target_tokens_this_batch)
            if accelerator.sync_gradients:
                if accumulation_tokens <= 0:
                    raise RuntimeError("gradient accumulation window contains no supervised target tokens")
                gradient_scale = cfg.gradient_accumulation_steps / accumulation_tokens
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(gradient_scale)
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
        if accelerator.sync_gradients:
            window_losses.append(accumulation_loss_sum / max(accumulation_tokens, 1))
            accumulation_loss_sum = 0.0
            accumulation_tokens = 0
            if sampler.mix_report()["absolute_token_imbalance"] > max_row_targets:
                raise RuntimeError("actual EN/RU supervised-token mix exceeded one packed-row tolerance")
        if step and step % cfg.logging_steps == 0 and accelerator.sync_gradients:
            elapsed = max(time.monotonic() - started, 1e-6)
            metrics = {"step": step, "admission": cfg.admission,
                       "processed_tokens": consumed, "optimizer_steps": step,
                       "loss": sum(window_losses[-cfg.logging_steps:]) / min(len(window_losses), cfg.logging_steps),
                        "tokens_per_second": (consumed - initial_consumed_tokens) / elapsed,
                        "learning_rate": scheduler.get_last_lr()[0],
                       "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                       **sampler.mix_report()}
            ledger.log_metrics(metrics)
            ledger.write_status(metrics)
        if cfg.eval_steps and step and step % cfg.eval_steps == 0 and accelerator.sync_gradients:
            metrics = evaluate(accelerator.unwrap_model(model), validation_loader)
            ledger.log_metrics({"step": step, "admission": cfg.admission, **metrics})
        if cfg.save_steps and step and step % cfg.save_steps == 0 and accelerator.sync_gradients:
            last_checkpoint = save_checkpoint(accelerator, model, tokenizer, cfg.output_dir, step=step, optimizer=optimizer,
                            scheduler=scheduler, consumed_tokens=consumed, sampler_state=sampler.state_dict(),
                            manifest=manifest, stage=cfg.stage, objective=approved["training"]["objective"],
                            budget_kind=cfg.budget_kind, target_tokens=cfg.max_tokens,
                            training_config_sha256=training_config_sha, admission=cfg.admission)
            ledger.write_status({"status": "running", "admission": cfg.admission,
                                 "budget_kind": cfg.budget_kind, "target_tokens": cfg.max_tokens,
                                 "step": step, "processed_tokens": consumed, "optimizer_steps": step,
                                 "checkpoint": str(last_checkpoint)})
        if accelerator.sync_gradients and consumed >= cfg.max_tokens:
            break
    progress.close()
    training_elapsed = max(time.monotonic() - started, 1e-6)
    final_eval = evaluate(accelerator.unwrap_model(model), validation_loader)
    if last_checkpoint is not None and last_checkpoint.name == f"step-{step:08d}":
        checkpoint = last_checkpoint
    else:
        checkpoint = save_checkpoint(
            accelerator, model, tokenizer, cfg.output_dir, step=step, optimizer=optimizer,
            scheduler=scheduler, consumed_tokens=consumed, sampler_state=sampler.state_dict(),
            manifest=manifest, stage=cfg.stage, objective=approved["training"]["objective"],
            budget_kind=cfg.budget_kind, target_tokens=cfg.max_tokens,
            training_config_sha256=training_config_sha, admission=cfg.admission,
        )
        ledger.write_status({"status": "running", "admission": cfg.admission,
                             "budget_kind": cfg.budget_kind, "target_tokens": cfg.max_tokens,
                             "step": step, "processed_tokens": consumed, "optimizer_steps": step,
                             "checkpoint": str(checkpoint)})
    mix_report = sampler.mix_report()
    if mix_report["sampled_target_tokens"] != consumed:
        raise RuntimeError("token-mix sampler accounting differs from tokens consumed by the training loop")
    if mix_report["absolute_token_imbalance"] > max_row_targets:
        raise RuntimeError("actual EN/RU supervised-token mix exceeded one packed-row tolerance")
    completed = consumed >= cfg.max_tokens
    status = "complete" if completed else "budget_incomplete"
    training_metrics = {
        "admission": cfg.admission,
        "step": step,
        "processed_tokens": consumed,
        "train_loss": sum(window_losses) / len(window_losses) if window_losses else None,
        "training_tokens_per_second": (consumed - initial_consumed_tokens) / training_elapsed,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        **mix_report,
        **final_eval,
    }
    ledger.log_metrics(training_metrics)
    result = {"status": status, "admission": cfg.admission, "budget_kind": cfg.budget_kind,
              "target_tokens": cfg.max_tokens,
              "step": step, "processed_tokens": consumed, "token_budget_shortfall": max(cfg.max_tokens - consumed, 0),
              "sampler_epoch": sampler.epoch, "language_mix": mix_report,
              "checkpoint": str(checkpoint), **training_metrics}
    ledger.write_status({**result, **training_metrics})
    ledger.append({"event": "training_complete" if completed else "training_budget_incomplete", **result})
    accelerator.end_training()
    return result


def run_lm_training(cfg: TrainConfig) -> dict:
    """Run training and persist any Python failure with the latest recorded progress."""
    try:
        return _run_lm_training(cfg)
    except Exception as exc:
        failure_traceback = traceback.format_exc()
        try:
            ledger = RunLedger(cfg.run_dir)
            failure_path = ledger.record_failure(type(exc).__name__, str(exc), failure_traceback)
            print(f"Training failure details recorded in {failure_path}", file=sys.stderr)
        except Exception as logging_error:
            print(f"Could not persist training failure details: {logging_error}", file=sys.stderr)
            sys.stderr.write(failure_traceback)
        raise
