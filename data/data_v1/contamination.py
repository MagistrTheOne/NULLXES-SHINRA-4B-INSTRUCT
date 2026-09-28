"""Probe firewall + pretrain specials + S0 v0 generators. Local only."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from tokenizer.special_tokens import PRETRAIN_FORBIDDEN_IDS, PRETRAIN_FORBIDDEN_STRINGS

from data.data_v1 import PhaseBError
from data.data_v1.normalize import contains_raw_url
from data.data_v1.phase_a import (
    S0_V0_GENERATORS,
    PhaseAError,
    assert_text_not_contaminated,
    fingerprint_set,
    load_all_probes,
    validate_sidecar,
)

FORBIDDEN_BODY_IDS = frozenset({2, *range(4, 18)})
assert FORBIDDEN_BODY_IDS == PRETRAIN_FORBIDDEN_IDS


@lru_cache(maxsize=1)
def frozen_probe_fingerprints() -> frozenset[str]:
    return frozenset(fingerprint_set(load_all_probes()))


def special_surface_hits(text: str) -> list[str]:
    hits: list[str] = []
    if "<|" in text:
        hits.append("<|")
    for tok in PRETRAIN_FORBIDDEN_STRINGS:
        if tok in text and tok not in hits:
            hits.append(tok)
    return hits


def forbidden_token_ids(ids: list[int]) -> set[int]:
    return set(ids) & FORBIDDEN_BODY_IDS


def assert_no_probe_overlap(text: str, fingerprints: set[str] | frozenset[str] | None = None) -> None:
    fps = fingerprints if fingerprints is not None else frozen_probe_fingerprints()
    try:
        assert_text_not_contaminated(text, set(fps))
    except PhaseAError as exc:
        raise PhaseBError("probe_overlap") from exc


def assert_no_v0_generator(record: dict[str, Any]) -> None:
    gen = record.get("generator")
    if gen in S0_V0_GENERATORS or (isinstance(gen, str) and gen.endswith(".v0")):
        raise PhaseBError("s0_v0_generator")


def assert_sidecar_offline(sidecar: dict[str, Any]) -> None:
    try:
        validate_sidecar(sidecar)
    except PhaseAError as exc:
        msg = str(exc)
        if "url" in msg.casefold():
            raise PhaseBError("raw_url_sidecar") from exc
        raise PhaseBError(msg) from exc
    if contains_raw_url(repr(sidecar)):
        raise PhaseBError("raw_url_sidecar")
