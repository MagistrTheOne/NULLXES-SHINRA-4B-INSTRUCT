"""Stage layer mix for streaming synth. S0–S2 this cycle are 100% synth."""

from __future__ import annotations

from data.synth.schema import LAYERS

STAGE_LAYER_WEIGHTS: dict[str, dict[str, float]] = {
    "s0": {
        "language_core": 0.70,
        "semantic_primitives": 0.10,
        "compositional": 0.05,
        "knowledge_shaped": 0.05,
        "structured": 0.10,
    },
    "s1": {
        "language_core": 0.60,
        "semantic_primitives": 0.15,
        "compositional": 0.10,
        "knowledge_shaped": 0.05,
        "structured": 0.10,
    },
    "s2": {
        "language_core": 0.20,
        "semantic_primitives": 0.40,
        "compositional": 0.25,
        "knowledge_shaped": 0.05,
        "structured": 0.10,
    },
}


def layer_weights(stage: str) -> dict[str, float]:
    if stage not in STAGE_LAYER_WEIGHTS:
        raise ValueError(f"unknown stage {stage}")
    weights = STAGE_LAYER_WEIGHTS[stage]
    missing = [layer for layer in LAYERS if layer not in weights]
    if missing:
        raise ValueError(f"stage {stage} missing layers {missing}")
    total = sum(weights[layer] for layer in LAYERS)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"stage {stage} weights sum to {total}")
    return weights
