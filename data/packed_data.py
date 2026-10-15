"""Shared training-data errors and token counters for the Parquet pipeline."""

from __future__ import annotations

import torch


class ShardError(RuntimeError):
    """A packed shard failed integrity or format validation."""


class ShardIdentityError(ShardError):
    """A shard does not belong to the frozen corpus and tokenizer identity."""


def count_target_tokens_post_shift(labels: torch.Tensor) -> int:
    """Count only non-masked next-token targets included in causal loss."""
    return int((labels[..., 1:] != -100).sum().item())
