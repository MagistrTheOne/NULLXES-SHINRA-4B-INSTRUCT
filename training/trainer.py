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
from data.shards import ShardIdentityError, assert_single_shard, count_label_tokens, count_target_tokens_post_shift, COUNTER_VERSION, verify_shard_identity
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
    # Explicit pass-through (no silent drops): these default to ShinraConfig
    # defaults but are now provenance-visible when set in architecture YAML.
    cfg.initializer_range = float(raw.get("initializer_range", cfg.initializer_range))
    if raw.get("sliding_window", None) is not None:
        cfg.sliding_window = raw.get("sliding_window")
    if raw.get("layer_types", None) is not None:
        cfg.layer_types = raw.get("layer_types")
    cfg._attn_implementation = attention_implementation
    cfg.auto_map = {
        "AutoConfig": "configuration_shinra.ShinraConfig",
        "AutoModel": "modeling_shinra.ShinraModel",
        "AutoModelForCausalLM": "modeling_shinra.ShinraForCausalLM",
    }
    return cfg


def verify_checkpoint_compat(checkpoint_dir: str | Path, arch_yaml: str | Path, tokenizer_dir: str | Path) -> dict:
    """Static C489 compat contract — reads JSON only, never loads weights.

    Checks: required files, geometry vs architecture YAML, special IDs vs
    tokenizer config. Raises SystemExit with an explicit reason on mismatch.
    config.json presence alone is NOT proof of C489.
    """
    import json as _json

    ckpt = Path(checkpoint_dir)
    required = ["config.json"]
    missing = [name for name in required if not (ckpt / name).exists()]
    if missing:
        raise SystemExit(f"checkpoint {ckpt} missing files: {missing}")
    weight_br = sorted(p.name for p in ckpt.glob("*.safetensors")) + sorted(p.name for p in ckpt.glob("*.bin"))
    weight_br = [n for n in weight_br if not n.startswith("optimizer")]
    if not weight_br:
        raise SystemExit(f"checkpoint {ckpt} has no weight files (*.safetensors/*.bin)")
    import yaml as _yaml

    ckpt_cfg = _json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    arch = (_yaml.safe_load(Path(arch_yaml).read_text(encoding="utf-8")) or {}).get("model", {})
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "vocab_size"):
        if key in arch and ckpt_cfg.get(key) != arch[key]:
            raise SystemExit(f"checkpoint geometry mismatch: {key} ckpt={ckpt_cfg.get(key)} arch={arch[key]}")
    head_dim = ckpt_cfg.get("head_dim", arch.get("head_dim", 128))
    if arch.get("num_attention_heads", 0) * head_dim != 4096:
        raise SystemExit("checkpoint Q width != 4096: not the V2 geometry")
    tok_cfg_path = Path(tokenizer_dir) / "tokenizer_config.json"
    if tok_cfg_path.exists():
        tok_cfg = _json.loads(tok_cfg_path.read_text(encoding="utf-8"))
        blob = _json.dumps(tok_cfg, ensure_ascii=False)
        for needle in ("<|bos|>", "<|eot|>", "<|end_of_text|>"):
            if needle not in blob:
                raise SystemExit(f"tokenizer at {tokenizer_dir} missing {needle}")
    return {"checkpoint": str(ckpt), "weights": weight_br, "geometry": "v2-ok"}


def load_model(cfg: TrainConfig) -> ShinraForCausalLM:
    if cfg.stage == "d_en_ru_pilot" and not cfg.resume_from:
        raise SystemExit(
            "d_en_ru_pilot is BLOCKED: set resume_from to the verified C489 path "
            "(docs/C489_LINK.md). Silent random-init fallback is forbidden for this stage."
        )
    if cfg.resume_from and (Path(cfg.resume_from) / "config.json").exists():
        verify_checkpoint_compat(cfg.resume_from, cfg.model_config, cfg.tokenizer_path)
        model = ShinraForCausalLM.from_pretrained(cfg.resume_from, torch_dtype=torch.bfloat16)
    elif cfg.base_from and cfg.stage in {"sft", "dpo"}:
        model = ShinraForCausalLM.from_pretrained(cfg.base_from, torch_dtype=torch.bfloat16)
    elif cfg.sft_from and cfg.stage == "dpo":
        model = ShinraForCausalLM.from_pretrained(cfg.sft_from, torch_dtype=torch.bfloat16)
    else:
        shinra_cfg = config_from_yaml(cfg.model_config, cfg.attention_implementation)
        model = ShinraForCausalLM(shinra_cfg)
        dtype = torch.bfloat16 if torch.cuda.is_available() and not cfg.force_cpu else torch.float32
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
            "shard_id": (extra or {}).get("shard_id"),
            "shard_sha256": (extra or {}).get("shard_sha256"),
            "n_sequences": (extra or {}).get("n_sequences"),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "consumed_trace": (extra or {}).get("consumed_trace"),
        }
        torch.save(blob, ckpt_dir / "trainer_state.pt")
        prune_checkpoints(output_dir)
    accelerator.wait_for_everyone()
    return ckpt_dir


@torch.no_grad()
def eval_heldout_ce(model, loader, max_batches: int = 32) -> dict:
    # Token-weighted heldout_ce = sum(CE_i * n_i) / sum(n_i), n_i = post-shift targets.
    # Single-pass: forward WITHOUT labels (logits only, no internal CE), then one
    # compute_loss_parts(logits, labels). Forward's outputs.loss is ignored here
    # to avoid double CE/logsumexp on the same logits.
    was_training = model.training
    model.eval()
    ce_weighted = 0.0
    z_weighted = 0.0
    tokens = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        labels = batch["labels"]
        n = count_target_tokens_post_shift(labels)
        if n <= 0:
            continue
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        base = model.module if hasattr(model, "module") else model
        parts = base.compute_loss_parts(outputs.logits, labels)
        ce_weighted += float(parts["train_ce"]) * n
        z_weighted += float(parts["z_loss"]) * n
        tokens += n
    if was_training:
        model.train()
    if not tokens:
        return {"eval_loss": None, "heldout_ce": None, "z_loss": None, "eval_tokens": 0, "counter_version": COUNTER_VERSION}
    heldout_ce = ce_weighted / tokens
    z_loss = z_weighted / tokens
    total = heldout_ce + z_loss
    return {"eval_loss": total, "heldout_ce": heldout_ce, "z_loss": z_loss, "eval_ppl": math.exp(min(heldout_ce, 20)), "eval_tokens": tokens, "counter_version": COUNTER_VERSION}


def _shard_ckpt_extra(
    cfg: TrainConfig,
    *,
    n_sequences: int,
    exhausted: bool,
    killed: bool,
    consumed_trace: list[int],
    overshoot_tokens: int,
) -> dict:
    return {
        "shard_id": cfg.active_shard_id,
        "shard_sha256": cfg.active_shard_sha256,
        "n_sequences": n_sequences,
        "exhausted": exhausted,
        "killed": killed,
        "overshoot_tokens": overshoot_tokens,
        "consumed_trace": consumed_trace if cfg.record_consumed_trace else None,
        "target_tokens": cfg.max_tokens,
    }


def _flush_leftover_accumulation(accelerator: Accelerator, model, optimizer, scheduler, grad_clip: float) -> bool:
    gs = getattr(accelerator, "gradient_state", None)
    if gs is None or bool(getattr(gs, "sync_gradients", True)):
        return False
    setter = getattr(gs, "_set_sync_gradients", None)
    if setter is None:
        return False
    setter(True)
    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return True


def run_lm_training(cfg: TrainConfig) -> dict:
    if cfg.stage == "d_en_ru_pilot" and not (cfg.resume_from and (Path(cfg.resume_from) / "config.json").exists()):
        raise SystemExit(
            "d_en_ru_pilot is BLOCKED: C489 path/compat unverified "
            "(docs/C489_LINK.md). No shards, no model load."
        )
    enable_tf32()
    set_seed(cfg.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    use_cuda = torch.cuda.is_available() and not cfg.force_cpu
    mixed = "bf16" if use_cuda else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=mixed,
        log_with="wandb" if cfg.wandb_project else None,
        kwargs_handlers=[ddp_kwargs],
        cpu=not use_cuda,
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

    if cfg.active_shard_id and cfg.active_shard_sha256:
        live = assert_single_shard(cfg.data_dir, cfg.active_shard_id)
        verify_shard_identity(live, cfg.active_shard_id, cfg.active_shard_sha256)

    tok_path = Path(cfg.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=True) if tok_path.exists() else None
    dataset = load_packed_dataset(cfg.data_dir)
    n_sequences = len(dataset)
    dataset_index = 0
    consumed = 0
    step = 0
    resume_blob: dict | None = None
    if cfg.resume_from:
        state_file = Path(cfg.resume_from) / "trainer_state.pt"
        if state_file.exists():
            resume_blob = load_trainer_state(state_file)
            if getattr(cfg, "fresh_stage_ledger", False):
                consumed = 0
                step = 0
                dataset_index = 0
            else:
                consumed = int(resume_blob.get("consumed_tokens", 0))
                step = int(resume_blob.get("step", 0))
                if cfg.active_shard_id:
                    same_shard = (
                        resume_blob.get("shard_id") == cfg.active_shard_id
                        and resume_blob.get("shard_sha256") == cfg.active_shard_sha256
                    )
                    dataset_index = int(resume_blob.get("dataset_index", 0)) if same_shard else 0
                else:
                    dataset_index = int(resume_blob.get("dataset_index", 0))
    if dataset_index > n_sequences:
        raise ShardIdentityError(f"dataset_index {dataset_index} > n_sequences {n_sequences}")
    if dataset_index > 0:
        dataset = Subset(dataset, list(range(dataset_index, n_sequences)))
    loader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=use_cuda,
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
    if resume_blob is not None:
        if resume_blob.get("optimizer") is not None:
            optimizer.load_state_dict(resume_blob["optimizer"])
        if not getattr(cfg, "fresh_stage_ledger", False):
            # Resume-interrupted: same stage/dir → restore schedule position + RNG.
            if resume_blob.get("scheduler") is not None:
                scheduler.load_state_dict(resume_blob["scheduler"])
            if resume_blob.get("rng") is not None:
                torch.set_rng_state(resume_blob["rng"])
            if resume_blob.get("cuda_rng") is not None and use_cuda:
                torch.cuda.set_rng_state_all(resume_blob["cuda_rng"])
        # New-stage start (fresh_stage_ledger): optimizer state kept per policy
        # (no unconditional Adam reset), scheduler fresh, RNG fresh.
        # optimizer.load_state_dict restores saved param-group LRs; sync groups to
        # the scheduler so the effective LR matches the chosen policy exactly.
    try:
        _sched_lrs = scheduler.get_last_lr()
        for _g, _lr in zip(optimizer.param_groups, _sched_lrs):
            _g["lr"] = float(_lr)
    except Exception:
        pass

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(cfg.run_dir)
    lr_start = None
    try:
        lr_start = float(scheduler.get_last_lr()[0])
    except Exception:
        pass
    ledger.write_status(
        {
            "event": "train_start",
            "stage": cfg.stage,
            "step": step,
            "consumed_tokens": consumed,
            "stop_counter": "legacy-honest (labels != -100 pre-shift); 8M limit on this counter",
            "lr_start": lr_start,
            "scheduler_restored": bool(resume_blob and resume_blob.get("scheduler") is not None and not getattr(cfg, "fresh_stage_ledger", False)),
            "optimizer_restored": bool(resume_blob and resume_blob.get("optimizer") is not None),
            "fresh_stage_ledger": bool(getattr(cfg, "fresh_stage_ledger", False)),
            "resume_from": str(cfg.resume_from) if cfg.resume_from else None,
        }
    )
    running = 0.0
    running_grad = 0.0
    t0 = time.time()
    model.train()
    iterator = iter(loader)
    progress = tqdm(
        total=cfg.max_steps,
        initial=step,
        disable=(not accelerator.is_main_process) or cfg.force_cpu,
        desc=cfg.stage,
    )
    exhausted = dataset_index >= n_sequences
    killed = False
    consumed_trace: list[int] = []
    consumed_targets_v1 = 0
    microbatches_in_group = 0
    overshoot_tokens = 0

    while step < cfg.max_steps and not exhausted and not killed:
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
            # Window-mean semantics: sum of microbatch means / accum_steps, so the
            # configured tokens_per_step/LR/schedule match the documented recipe.
            # (History S0/C stepped every microbatch without scaling — see report.)
            acc_steps = max(int(cfg.gradient_accumulation_steps), 1)
            accelerator.backward(loss / acc_steps)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        batch_tokens = count_label_tokens(batch["labels"])
        batch_targets_v1 = count_target_tokens_post_shift(batch["labels"])
        consumed += batch_tokens
        consumed_targets_v1 += batch_targets_v1
        if cfg.record_consumed_trace:
            consumed_trace.append(consumed)
        microbatches_in_group += 1
        if accelerator.sync_gradients:
            step += 1
            microbatches_in_group = 0
            gathered = accelerator.gather(loss.detach())
            loss_val = float(gathered.mean().item())
            running += loss_val
            if grad_norm is not None:
                running_grad += float(grad_norm.detach() if hasattr(grad_norm, "detach") else grad_norm)
            progress.update(1)
            if cfg.max_tokens > 0 and consumed >= cfg.max_tokens:
                overshoot_tokens = consumed - cfg.max_tokens
                break
            if cfg.stop_after_steps is not None and step >= cfg.stop_after_steps:
                killed = True
                break
            if step % cfg.logging_steps == 0 and accelerator.is_main_process:
                elapsed = max(time.time() - t0, 1e-6)
                tps = consumed / elapsed
                avg = running / cfg.logging_steps
                ppl = math.exp(min(avg, 20))
                lr = scheduler.get_last_lr()[0]
                nan_flag = not math.isfinite(avg)
                entropy = last_token_entropy(outputs.logits)
                # Train split, computed from already-available logits (no extra forward;
                # +1 CE/logsumexp on logging steps only, ~1/10 steps overhead).
                # Stop-limit stays on legacy honest `consumed` (pre-shift); v1 is parallel.
                # NOTE: last_token_entropy + parts run without retaining extra graph
                # (detached floats only; same lifetime as `outputs`).
                try:
                    base = accelerator.unwrap_model(model)
                    with torch.no_grad():
                        _parts = base.compute_loss_parts(outputs.logits, batch["labels"])
                    train_ce_avg, z_avg = float(_parts["train_ce"]), float(_parts["z_loss"])
                except Exception:
                    train_ce_avg, z_avg = avg, 0.0
                remaining = None
                if cfg.max_tokens > 0:
                    remaining = max(cfg.max_tokens - consumed, 0)
                metrics = {
                    "loss": avg,
                    "train_ce": train_ce_avg,
                    "z_loss": z_avg,
                    "ppl": ppl,
                    "ce_ppl": math.exp(min(train_ce_avg, 20)),
                    "lr": lr,
                    "consumed_tokens": consumed,
                    "consumed_targets_v1": consumed_targets_v1,
                    "counter_version": COUNTER_VERSION,
                    "tokens_per_sec": tps,
                    "step": step,
                    "nan": int(nan_flag),
                    "grad_norm": running_grad / cfg.logging_steps,
                    "entropy": entropy,
                    "stage": cfg.stage,
                    "overshoot_tokens": overshoot_tokens,
                }
                ledger.log_metrics(metrics)
                ledger.write_status(
                    {
                        "stage": cfg.stage,
                        "step": step,
                        "consumed_tokens": consumed,
                        "remaining_tokens": remaining,
                        "overshoot_tokens": overshoot_tokens,
                        "tokens_per_sec": tps,
                        "loss": avg,
                        "lr": lr,
                        "ppl": ppl,
                        "nan": nan_flag,
                        "exhausted": exhausted,
                        "shard_id": cfg.active_shard_id,
                        "shard_sha256": cfg.active_shard_sha256,
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
                    extra=_shard_ckpt_extra(
                        cfg,
                        n_sequences=n_sequences,
                        exhausted=exhausted,
                        killed=killed,
                        consumed_trace=consumed_trace,
                        overshoot_tokens=overshoot_tokens,
                    ),
                )

    if exhausted and microbatches_in_group > 0:
        if _flush_leftover_accumulation(accelerator, model, optimizer, scheduler, cfg.grad_clip):
            step += 1

    if cfg.max_tokens > 0 and consumed >= cfg.max_tokens:
        overshoot_tokens = consumed - cfg.max_tokens

    extra = _shard_ckpt_extra(
        cfg,
        n_sequences=n_sequences,
        exhausted=exhausted,
        killed=killed,
        consumed_trace=consumed_trace,
        overshoot_tokens=overshoot_tokens,
    )
    ckpt_dir = save_checkpoint(
        accelerator,
        model,
        tokenizer,
        cfg.output_dir / "final",
        step,
        optimizer=optimizer,
        scheduler=scheduler,
        consumed_tokens=consumed,
        dataset_index=dataset_index,
        extra=extra,
    )
    done = (not killed) and (exhausted or (cfg.max_tokens > 0 and consumed >= cfg.max_tokens) or cfg.max_tokens <= 0)
    if accelerator.is_main_process:
        remaining = None
        if cfg.max_tokens > 0:
            remaining = max(cfg.max_tokens - consumed, 0)
        ledger.write_status(
            {
                "stage": cfg.stage,
                "step": step,
                "consumed_tokens": consumed,
                "remaining_tokens": remaining,
                "overshoot_tokens": overshoot_tokens,
                "exhausted": exhausted,
                "killed": killed,
                "done": done and not killed,
                "shard_id": cfg.active_shard_id,
                "shard_sha256": cfg.active_shard_sha256,
                "dataset_index": dataset_index,
            }
        )
        if not killed:
            ledger.append(
                {
                    "event": "train_done",
                    "step": step,
                    "consumed_tokens": consumed,
                    "exhausted": exhausted,
                    "overshoot_tokens": overshoot_tokens,
                    "shard_id": cfg.active_shard_id,
                    "shard_sha256": cfg.active_shard_sha256,
                }
            )
        else:
            ledger.append(
                {
                    "event": "train_killed",
                    "step": step,
                    "consumed_tokens": consumed,
                    "dataset_index": dataset_index,
                    "shard_id": cfg.active_shard_id,
                    "shard_sha256": cfg.active_shard_sha256,
                }
            )
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return {
        "step": step,
        "consumed_tokens": consumed,
        "dataset_index": dataset_index,
        "n_sequences": n_sequences,
        "exhausted": exhausted,
        "killed": killed,
        "checkpoint_dir": str(ckpt_dir),
        "consumed_trace": consumed_trace,
        "overshoot_tokens": overshoot_tokens,
        "shard_id": cfg.active_shard_id,
        "shard_sha256": cfg.active_shard_sha256,
    }
