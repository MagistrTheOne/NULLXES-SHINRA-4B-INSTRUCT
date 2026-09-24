"""P0 Amendment 1. Identity is the unique bank once. Full S1 percentages stay in the spec."""

from __future__ import annotations

P0_BUDGET = 2_500_000
IDENTITY_HARD_MAX = 0.06
NON_IDENTITY_WEIGHTS = {
    "S1-01": 15,
    "S1-02": 15,
    "S1-03": 12,
    "S1-04": 12,
    "S1-05": 10,
    "S1-06": 12,
    "S1-07": 10,
    "S1-08": 5,
    "S1-09": 4,
}
WEIGHT_SUM = 95


def effective_family_targets(identity_tokens: int) -> dict[str, int]:
    """Largest remainder. Ties break by family id. Identity is not rescaled."""
    if identity_tokens < 0 or identity_tokens > P0_BUDGET:
        raise ValueError(identity_tokens)
    if identity_tokens > int(P0_BUDGET * IDENTITY_HARD_MAX):
        raise ValueError("identity bank exceeds the 6% ceiling")
    remaining = P0_BUDGET - identity_tokens
    floors: dict[str, int] = {}
    remainders: dict[str, int] = {}
    for family, weight in NON_IDENTITY_WEIGHTS.items():
        floors[family], remainders[family] = divmod(remaining * weight, WEIGHT_SUM)
    gap = remaining - sum(floors.values())
    winners = sorted(floors, key=lambda family: (-remainders[family], family))[:gap]
    for family in winners:
        floors[family] += 1
    floors["S1-10"] = identity_tokens
    if sum(floors.values()) != P0_BUDGET:
        raise RuntimeError("effective budgets do not sum to the P0 budget")
    return floors
