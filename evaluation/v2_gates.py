"""S0/S1/S2 GO/NO-GO probes. Tiny-model and corpus checks. Not identity SFT."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from data.synth.contamination import ContaminationError, assert_clean_pretrain_text
from tokenizer.special_tokens import PRETRAIN_FORBIDDEN_IDS


class GateFail(AssertionError):
    pass


def assert_packed_clean(ids: list[int]) -> None:
    overlap = PRETRAIN_FORBIDDEN_IDS.intersection(ids)
    if overlap:
        raise GateFail(f"forbidden special ids: {sorted(overlap)}")


def assert_no_identity(text: str) -> None:
    try:
        assert_clean_pretrain_text(text)
    except ContaminationError as exc:
        raise GateFail(str(exc)) from exc


def finite_loss(loss: torch.Tensor) -> None:
    if not torch.isfinite(loss):
        raise GateFail("non-finite loss")


def mean_entropy(logits: torch.Tensor) -> float:
    log_p = F.log_softmax(logits.float(), dim=-1)
    return float((-(log_p.exp() * log_p).sum(dim=-1)).mean().detach().cpu())


def repetition_ratio(ids: list[int], ngram: int = 3) -> float:
    if len(ids) < ngram * 2:
        return 0.0
    grams = [tuple(ids[i : i + ngram]) for i in range(len(ids) - ngram + 1)]
    return 1.0 - (len(set(grams)) / max(len(grams), 1))


def s0_pass(*, loss_start: float, loss_end: float, grad_norm: float, packed_ids: list[int]) -> dict:
    finite_loss(torch.tensor(loss_end))
    if not math.isfinite(grad_norm):
        raise GateFail("non-finite grad_norm")
    assert_packed_clean(packed_ids)
    moved = loss_end < loss_start - 1e-6 or loss_end != loss_start
    return {"loss_moved": bool(moved), "loss_end": loss_end, "grad_norm": grad_norm}


def s1_language_alive(en_ce: float, ru_ce: float, random_ce: float) -> dict:
    if not (math.isfinite(en_ce) and math.isfinite(ru_ce)):
        raise GateFail("non-finite language CE")
    if en_ce >= random_ce and ru_ce >= random_ce:
        raise GateFail("EN/RU CE not below random baseline")
    return {"en_ce": en_ce, "ru_ce": ru_ce, "random_ce": random_ce}


def s2_semantic_alive(report: Mapping[str, float]) -> dict:
    required = ("negation_acc", "relation_acc", "coref_acc")
    missing = [k for k in required if k not in report]
    if missing:
        raise GateFail(f"missing semantic scores {missing}")
    dead = [k for k in required if report[k] <= 0.0]
    if dead:
        raise GateFail(f"dead semantic probes {dead}")
    return dict(report)
