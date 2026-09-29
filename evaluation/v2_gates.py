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


EVAL_GATE_CONTRACT = {
    "use_cache": False,
    "eos_token_id": 18,
    "do_sample": False,
}


def _gate_artifact_path(run_dir, kind: str, ckpt) -> "object":
    from pathlib import Path as _Path

    name = str(ckpt).replace("/", "_").replace("\\", "_").strip("_")
    return _Path(run_dir) / "eval" / f"{kind}_{name}.json"


def check_baseline_gate(run_dir, ckpt) -> dict:
    """Baseline continuation eval for the STARTING checkpoint must exist BEFORE
    the first D update. Called by the drum, not by the training loop."""
    path = _gate_artifact_path(run_dir, "baseline", ckpt)
    if not path.is_file():
        raise GateFail(
            f"D baseline missing for {ckpt}: run evaluation/continuation_eval.py first, "
            f"write {path} (use_cache=false, eos=18, do_sample=false)"
        )
    return {"baseline": str(path)}


def check_stage_gates(run_dir, final_ckpt) -> dict:
    """Mid + final continuation eval must exist before stage D is marked done."""
    missing = []
    found = {}
    for kind, ckpt in (("mid", final_ckpt), ("final", final_ckpt)):
        path = _gate_artifact_path(run_dir, kind, ckpt)
        if path.is_file():
            found[kind] = str(path)
        else:
            missing.append(str(path))
    if missing:
        raise GateFail(f"D stage gates missing: {missing}")
    return found


def language_score(*, en_ce: float, ru_ce: float, gen: Mapping[str, float] | None = None) -> dict:
    """Language score uses ONLY EN/RU CE + generation repetition/UNK stats.

    Negative identity/chat probes (identity_absent/chat_absent passes) are
    reported separately and MUST NOT enter this score.
    """
    if not (math.isfinite(en_ce) and math.isfinite(ru_ce)):
        raise GateFail("non-finite language CE")
    score = {"en_ce": en_ce, "ru_ce": ru_ce}
    if gen:
        score.update({k: gen[k] for k in ("rep_trigram", "unk_rate") if k in gen})
    return score
