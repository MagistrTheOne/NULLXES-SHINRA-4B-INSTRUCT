"""P0 loss. Sums only. The update divides once, outside this module."""

from __future__ import annotations

import torch
import torch.nn.functional as F

Z_LOSS_COEFFICIENT = 1e-5


class ObjectiveError(RuntimeError):
    pass


def require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ObjectiveError(f"nonfinite {name}")


def supervised_sums(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return sum CE, sum z-loss, and the shifted supervised count."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ObjectiveError("expected one example: logits [1, T, V], labels [1, T]")
    if logits.shape[0] != 1 or labels.shape[0] != 1:
        raise ObjectiveError("P0 forward is one example")
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    supervised = flat_labels != -100
    count = int(supervised.sum().item())
    if count <= 0:
        raise ObjectiveError("no supervised tokens")
    require_finite(flat_logits, "logits")
    cross_entropy = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="sum")
    require_finite(cross_entropy, "cross_entropy")
    log_z = torch.logsumexp(flat_logits[supervised], dim=-1)
    z_loss = Z_LOSS_COEFFICIENT * (log_z.square().sum())
    require_finite(z_loss, "z_loss")
    return cross_entropy, z_loss, count


def scale_gradients(parameters, divisor: int) -> None:
    if divisor <= 0:
        raise ObjectiveError("divisor must be the update target count")
    scaled = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        require_finite(parameter.grad, "gradient")
        parameter.grad.div_(float(divisor))
        scaled = True
    if not scaled:
        raise ObjectiveError("no gradients to scale")
