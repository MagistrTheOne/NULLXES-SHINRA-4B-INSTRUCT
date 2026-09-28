"""Exact content-addressed dedup + shingle Jaccard near-dedup. No network."""

from __future__ import annotations

from data.data_v1.normalize import document_id, exact_hash
from data.data_v1.phase_a import normalize_fingerprint


def _word_shingles(text: str, n: int = 3) -> set[str]:
    tokens = normalize_fingerprint(text).split()
    if not tokens:
        return set()
    if len(tokens) < n:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _char_shingles(text: str, n: int = 16) -> set[str]:
    folded = normalize_fingerprint(text)
    if not folded:
        return set()
    if len(folded) < n:
        return {folded}
    return {folded[i : i + n] for i in range(len(folded) - n + 1)}


def shingle_set(text: str) -> set[str]:
    words = _word_shingles(text, 3)
    return words if len(words) >= 8 else (_char_shingles(text) or words)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    union = len(left | right)
    return inter / union if union else 0.0


class CanaryDedup:
    def __init__(self, threshold: float = 0.80) -> None:
        self.threshold = threshold
        self.exact: dict[str, str] = {}
        self.near: list[tuple[str, set[str]]] = []
        self.exact_duplicates = 0
        self.near_duplicates = 0

    def check(self, text: str) -> tuple[str, str | None]:
        doc_id = document_id(text)
        digest = exact_hash(text)
        if digest in self.exact:
            self.exact_duplicates += 1
            return "exact", self.exact[digest]
        shingles = shingle_set(text)
        for other_id, other in self.near:
            if jaccard(shingles, other) >= self.threshold:
                self.near_duplicates += 1
                return "near", other_id
        self.exact[digest] = doc_id
        self.near.append((doc_id, shingles))
        return "keep", None
