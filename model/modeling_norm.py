"""RMSNorm used throughout SHINRA (pre-attention, pre-MLP, final, optional QK-norm)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShinraRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(hidden_states, (hidden_states.size(-1),), self.weight, self.variance_epsilon)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.variance_epsilon}"
