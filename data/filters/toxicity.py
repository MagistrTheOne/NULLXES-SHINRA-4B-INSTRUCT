"""Heuristic toxicity / abuse filter.

A word-list + pattern layer runs on every document. Optional classifier
weights can be loaded via TRANSFORMERS if TOXICITY_MODEL is set; the
heuristic is the default production gate so the pipeline never depends
on a missing checkpoint.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

TOXIC_PATTERNS = [
    r"\bkill yourself\b",
    r"\bkys\b",
    r"\bmake a bomb\b",
    r"\bhow to make (?:a )?bomb\b",
    r"\bchild sexual\b",
    r"\bcp\s*content\b",
    r"\brape\b",
    r"\bincest\b",
    r"\bnigger\b",
    r"\bfaggot\b",
    r"\bgas the\b",
    r"\bslit your\b",
    r"\bI will (?:kill|murder|rape)\b",
]

COMPILED = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]

# High-precision English/Russian abuse lexemes. Kept compact on purpose:
# recall is boosted by patterns above, precision by requiring word boundaries.
LEXICON = {
    "nigger",
    "niggers",
    "faggot",
    "faggots",
    "kike",
    "retard",
    "retards",
    "rape",
    "raping",
    "pedophile",
    "paedophile",
    "childporn",
    "suicidechest",
    "пидор",
    "пидарас",
    "ниггер",
    "убить себя",
}


def heuristic_toxicity(text: str) -> dict:
    lowered = text.lower()
    hits: list[str] = []
    for pattern in COMPILED:
        if pattern.search(lowered):
            hits.append(pattern.pattern)
            if len(hits) >= 4:
                break
    for term in LEXICON:
        if term in lowered:
            hits.append(term)
            if len(hits) >= 6:
                break
    score = min(1.0, 0.35 * len(hits))
    return {"toxicity": round(score, 4), "hits": hits[:8], "keep": score < 0.50}


@lru_cache(maxsize=1)
def _hf_classifier(model_id: str):
    from transformers import pipeline

    return pipeline("text-classification", model=model_id, truncation=True, max_length=512)


def score_toxicity(text: str, threshold: float = 0.50) -> dict:
    result = heuristic_toxicity(text)
    model_id = os.environ.get("SHINRA_TOXICITY_MODEL")
    if model_id:
        clf = _hf_classifier(model_id)
        pred = clf(text[:2000])[0]
        label = str(pred["label"]).lower()
        score = float(pred["score"])
        toxic = ("toxic" in label or "hate" in label or label.endswith("_1")) and score >= threshold
        result["model_label"] = pred["label"]
        result["model_score"] = score
        result["keep"] = result["keep"] and not toxic
        result["toxicity"] = max(result["toxicity"], score if toxic else result["toxicity"])
    return result
