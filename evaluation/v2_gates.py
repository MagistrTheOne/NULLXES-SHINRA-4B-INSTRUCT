"""S0/S1/S2 GO/NO-GO probes. Tiny-model and corpus checks. Not identity SFT."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

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


def _gate_artifact_path(run_dir: str | Path, kind: str, ckpt: str | Path) -> Path:
    name = str(ckpt).replace("/", "_").replace("\\", "_").strip("_")
    return Path(run_dir) / "eval" / f"{kind}_{name}.json"


def _validate_gate_artifact(path: Path, ckpt) -> dict:
    """Content validation: existence never opens a gate. Checks identity,
    evaluator version, contract fields, EN/RU coverage with non-zero
    denominators, and finite metrics. Stale/empty/foreign JSON raises."""
    import json as _json
    import math as _math

    try:
        doc = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateFail(f"gate artifact unreadable: {path} ({exc})") from exc
    if doc.get("evaluator_version") != "continuation-v1":
        raise GateFail(f"gate artifact {path}: bad evaluator_version {doc.get('evaluator_version')!r}")
    if str(doc.get("checkpoint", "")) != str(ckpt):
        raise GateFail(f"gate artifact {path}: checkpoint {doc.get('checkpoint')!r} != {ckpt!r}")
    contract = doc.get("contract", {})
    if contract.get("use_cache") is not False or contract.get("eos_token_id") != 18 or contract.get("do_sample") is not False:
        raise GateFail(f"gate artifact {path}: contract mismatch {contract!r}")
    rows = doc.get("rows")
    if not isinstance(rows, list) or not rows:
        raise GateFail(f"gate artifact {path}: empty rows")
    langs = {r.get("lang") for r in rows if isinstance(r, dict)}
    if langs != {"en", "ru"}:
        raise GateFail(f"gate artifact {path}: EN/RU coverage {sorted(str(x) for x in langs)}")
    for row in rows:
        if not isinstance(row, dict):
            raise GateFail(f"gate artifact {path}: malformed row")
        if row.get("eos_id") != 18 or row.get("use_cache") is not False or row.get("do_sample") is not False:
            raise GateFail(f"gate artifact {path}: row contract mismatch")
        unk_rate = row.get("unk_rate")
        if not isinstance(unk_rate, (int, float)) or not _math.isfinite(unk_rate):
            raise GateFail(f"gate artifact {path}: non-finite unk_rate")
        gen = row.get("generated_ids")
        if not isinstance(gen, list) or not gen:
            raise GateFail(f"gate artifact {path}: empty generation")
    per_lang = doc.get("per_lang", {})
    for lang in ("en", "ru"):
        entry = per_lang.get(lang, {})
        if not entry.get("n", 0) > 0:
            raise GateFail(f"gate artifact {path}: zero denominator for {lang}")
    return doc


def check_baseline_gate(run_dir, ckpt) -> dict:
    """Baseline continuation eval for the STARTING checkpoint must exist BEFORE
    the first D update. Called by the drum, not by the training loop.
    No deadlock: the artifact is produced by manually running
    evaluation/continuation_eval.py on the starting checkpoint (GPU + weights,
    no trainer run required). Content-validated; stale/empty JSON is rejected."""
    path = _gate_artifact_path(run_dir, "baseline", ckpt)
    if not path.is_file():
        raise GateFail(
            f"D baseline missing for {ckpt}: run evaluation/continuation_eval.py first, "
            f"write {path} (use_cache=false, eos=18, do_sample=false)"
        )
    _validate_gate_artifact(path, ckpt)
    return {"baseline": str(path)}


def check_stage_gates(run_dir, final_ckpt) -> dict:
    """Mid + final continuation eval must exist before stage D is marked done."""
    missing = []
    found = {}
    for kind, ckpt in (("mid", final_ckpt), ("final", final_ckpt)):
        path = _gate_artifact_path(run_dir, kind, ckpt)
        if path.is_file():
            try:
                _validate_gate_artifact(path, ckpt)
            except GateFail as exc:
                raise GateFail(f"D stage gate invalid ({kind}): {exc}") from exc
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
