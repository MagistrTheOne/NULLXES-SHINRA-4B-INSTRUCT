"""Streaming synth documents. Deterministic. No 20k fixture cap."""

from __future__ import annotations

from collections.abc import Iterator
from random import Random

from data.synth.generate import generate_record
from data.synth.schema import LANGUAGES
from data.synth.stage_mix import layer_weights


def pick_weighted(rng: Random, weights: dict[str, float]) -> str:
    draw = rng.random()
    cursor = 0.0
    last = next(iter(weights))
    for key, weight in weights.items():
        cursor += weight
        last = key
        if draw <= cursor:
            return key
    return last


def iter_stream_records(stage: str, seed: int, count: int, start_index: int = 0) -> Iterator[dict]:
    weights = layer_weights(stage)
    rng = Random(seed)
    for offset in range(count):
        index = start_index + offset
        layer = pick_weighted(rng, weights)
        language = LANGUAGES[index % 2]
        yield generate_record(seed, layer, language, index)
