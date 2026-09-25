"""Dedicated S1-P0 loop. One example per forward. No Accelerate accumulation."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.optim import AdamW

from training.s1_p0_checkpoint import load_resume, publish_latest, write_generation
from training.s1_p0_loss import ObjectiveError, scale_gradients, supervised_sums
from training.s1_p0_reader import DatasetPinError, take_update
from training.s1_p0_tokens import encode_example

PARENT_SUBFOLDER = "exp/stage0-2026-09-23/checkpoints/S0.4.1-parent"
PARENT_WEIGHT = "model.safetensors"
PARENT_SHA256 = "99b2111a2c1437d0bff926ee6c96369447b5844d53be84db3a0dec7197947e56"
CONTRACT_COMMIT = "000813d2294b6ec8f2287d4eb46f64c105d0617c"
SEED = 20260924
LEARNING_RATE = 1.5e-8
BETAS = (0.9, 0.999)
EPS = 1e-6
CLIP_NORM = 1.0


def build_optimizer(model: torch.nn.Module) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPS,
        weight_decay=0.0,
        foreach=False,
    )


def load_parent(model_root: str | Path) -> torch.nn.Module:
    """Load S0.4.1 for a future run. This function is not called by phase 1."""
    from model.modeling_shinra import ShinraForCausalLM

    path = Path(model_root) / PARENT_SUBFOLDER
    weight = path / PARENT_WEIGHT
    if not weight.is_file():
        raise FileNotFoundError(PARENT_WEIGHT)
    import hashlib

    digest = hashlib.sha256()
    with weight.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != PARENT_SHA256:
        raise RuntimeError("parent weight hash mismatch")
    model = ShinraForCausalLM.from_pretrained(str(path), torch_dtype=torch.float16)
    model = model.to(dtype=torch.float16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    return model


def set_seed() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def materialize_example(tokenizer, template, record: dict) -> dict:
    messages = record["messages"]
    user = next(item["content"] for item in messages if item["role"] == "user")
    answer = next(item["content"] for item in messages if item["role"] == "assistant")
    encoded = encode_example(tokenizer, template, user, answer)
    if int(encoded["target_tokens"]) != int(record["target_tokens"]):
        raise ObjectiveError("rendered target count disagrees with the record")
    if int(encoded["prompt_tokens"]) != int(record["prompt_tokens"]):
        raise ObjectiveError("rendered prompt count disagrees with the record")
    return encoded


def apply_update(model, optimizer, examples: list[dict], expected_tokens: int) -> dict:
    """Backward one canonical update, scale once, clip, step."""
    optimizer.zero_grad(set_to_none=True)
    cross_entropy_sum = 0.0
    z_loss_sum = 0.0
    actual = 0
    for example in examples:
        input_ids = example["input_ids"]
        labels = example["labels"]
        if not torch.is_tensor(input_ids):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        if not torch.is_tensor(labels):
            labels = torch.tensor([labels], dtype=torch.long)
        device = next(model.parameters()).device
        logits = model(input_ids=input_ids.to(device), labels=None, use_cache=False).logits
        cross_entropy, z_loss, count = supervised_sums(logits, labels.to(device))
        if count != int(example["target_tokens"]):
            raise ObjectiveError("rendered target count disagrees with the record")
        (cross_entropy + z_loss).backward()
        cross_entropy_sum += float(cross_entropy.detach())
        z_loss_sum += float(z_loss.detach())
        actual += count
    if actual != expected_tokens:
        raise ObjectiveError(f"update tokens {actual} != plan {expected_tokens}")
    scale_gradients(model.parameters(), actual)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    if not torch.isfinite(grad_norm):
        raise ObjectiveError("nonfinite gradient norm")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "target_tokens": actual,
        "cross_entropy": cross_entropy_sum / actual,
        "z_loss": z_loss_sum / actual,
        "loss": (cross_entropy_sum + z_loss_sum) / actual,
        "grad_norm": float(grad_norm),
        "learning_rate": LEARNING_RATE,
        "examples": len(examples),
    }


def run_chunk(model, optimizer, records: list[dict], updates: list[dict], start: int, end: int, workspace: Path, identity: dict, tokenizer, template) -> int:
    """Run whole optimizer updates [start, end). Checkpoint after each one."""
    if not 0 <= start < end <= len(updates):
        raise DatasetPinError("chunk is not a range of finished updates")
    identity = dict(identity)
    identity.update({
        "learning_rate": LEARNING_RATE,
        "betas": list(BETAS),
        "eps": EPS,
        "weight_decay": 0.0,
        "foreach": False,
        "bos_id": 1,
        "eot_id": 2,
        "pad_id": 3,
        "vocab_size": 131072,
    })
    seen_tokens = 0
    for update in updates[start:end]:
        chunk = [materialize_example(tokenizer, template, record) for record in take_update(records, update)]
        stats = apply_update(model, optimizer, chunk, int(update["target_tokens"]))
        stats["update_index"] = int(update["update_index"])
        stats["record_start"] = int(update["record_start"])
        stats["record_end"] = int(update["record_end"])
        seen_tokens += int(stats["target_tokens"])
        stats["cumulative_target_tokens"] = seen_tokens
        print(
            "update {update_index} records {record_start}:{record_end} tokens {target_tokens} "
            "ce {cross_entropy:.6f} z {z_loss:.6f} loss {loss:.6f} grad {grad_norm:.4f} "
            "lr {learning_rate} examples {examples} cumulative {cumulative_target_tokens}".format(**stats),
            flush=True,
        )
        identity["next_update"] = int(update["update_index"]) + 1
        _save_generation(model, optimizer, workspace, identity)
    return identity["next_update"]


def _save_generation(model, optimizer, workspace: Path, identity: dict) -> None:
    import io

    buffer = io.BytesIO()
    torch.save({"state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}}, buffer)
    model_bytes = buffer.getvalue()
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    optimizer_bytes = buffer.getvalue()
    buffer = io.BytesIO()
    torch.save(torch.get_rng_state(), buffer)
    cpu_rng = buffer.getvalue()
    cuda_rng = {"available": False}
    if torch.cuda.is_available():
        cuda_rng = {"available": True, "state": [tensor.cpu().tolist() for tensor in torch.cuda.get_rng_state_all()]}
    index = int(identity["next_update"])
    import json

    import base64
    import pickle

    files = {
        "model.pt": model_bytes,
        "optimizer.pt": optimizer_bytes,
        "rng_python.json": json.dumps({"state": base64.b64encode(pickle.dumps(random.getstate())).decode("ascii")}),
        "rng_torch_cpu.pt": cpu_rng,
        "rng_torch_cuda.json": json.dumps(cuda_rng),
        "cursor.json": json.dumps({"next_update": index}),
    }
    write_generation(workspace, index, files, identity)
    publish_latest(workspace, index, identity)
    load_resume(workspace, identity)
