"""Learning-rate schedules: WSD (warmup-stable-decay) and cosine."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def wsd_lambda(step: int, warmup: int, total: int, stable_ratio: float, min_ratio: float) -> float:
    if total <= 0:
        return 1.0
    if step < warmup:
        return max(step, 1) / max(warmup, 1)
    stable_end = int(total * stable_ratio)
    if step < stable_end:
        return 1.0
    decay_span = max(total - stable_end, 1)
    progress = (step - stable_end) / decay_span
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_ratio + (1.0 - min_ratio) * cosine


def cosine_lambda(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return max(step, 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_ratio + (1.0 - min_ratio) * cosine


def build_scheduler(
    optimizer: Optimizer,
    name: str,
    warmup_steps: int,
    max_steps: int,
    stable_ratio: float,
    min_lr_ratio: float,
) -> LambdaLR:
    name = name.lower()
    if name == "wsd":
        fn = lambda step: wsd_lambda(step, warmup_steps, max_steps, stable_ratio, min_lr_ratio)
    elif name == "cosine":
        fn = lambda step: cosine_lambda(step, warmup_steps, max_steps, min_lr_ratio)
    else:
        raise ValueError(f"Unknown scheduler {name}")
    return LambdaLR(optimizer, lr_lambda=fn)
