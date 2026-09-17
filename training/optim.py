"""AdamW fused optimizer with decay/no-decay parameter groups."""

from __future__ import annotations

import torch
from torch.optim import AdamW

from model.modeling_norm import ShinraRMSNorm


def build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if param_name.endswith("bias") or isinstance(module, (ShinraRMSNorm, torch.nn.Embedding)):
                no_decay.append(param)
            else:
                decay.append(param)
    fused = torch.cuda.is_available()
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=betas,
        eps=eps,
        fused=fused,
    )
