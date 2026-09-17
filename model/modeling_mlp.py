"""SwiGLU MLP for SHINRA transformer blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .configuration_shinra import ShinraConfig
except ImportError:
    from configuration_shinra import ShinraConfig


class ShinraMLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x)). No bias."""

    def __init__(self, config: ShinraConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        bias = config.mlp_bias
        self.gate_proj = nn.Linear(hidden, intermediate, bias=bias)
        self.up_proj = nn.Linear(hidden, intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, hidden, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)
