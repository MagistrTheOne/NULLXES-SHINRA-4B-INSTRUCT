"""S1-P0 update math. Not the pretrain or production-SFT objective."""

from __future__ import annotations

S1_P0_SEED = 20260924


def token_weighted_mean(groups: list[list[float]]) -> float:
    """Mean over supervised tokens, not over microbatches."""
    total = 0.0
    count = 0
    for group in groups:
        if not group:
            raise ValueError("empty supervised group")
        total += float(sum(group))
        count += len(group)
    if count == 0:
        raise ValueError("no supervised tokens")
    return total / count


def equal_microbatch_mean(groups: list[list[float]]) -> float:
    """The wrong P0 reduction: each microbatch mean has weight 1."""
    if not groups:
        raise ValueError("no microbatches")
    return sum(sum(group) / len(group) for group in groups) / len(groups)


def gradient_divisor(total_target_tokens: int) -> float:
    """Divide the sum of per-token gradients by this once per update."""
    if total_target_tokens <= 0:
        raise ValueError("target token count must be positive")
    return float(total_target_tokens)


def can_attend(doc_ids: list[int], query: int, key: int) -> bool:
    """Causal attention inside one example. No cross-example attention."""
    if query < 0 or key < 0 or query >= len(doc_ids) or key >= len(doc_ids):
        raise IndexError("position outside the packed row")
    return doc_ids[query] == doc_ids[key] and key <= query
