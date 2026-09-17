"""MinHash-LSH near-duplicate detection over document shards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from datasketch import MinHash, MinHashLSH


def normalize_for_hash(text: str) -> str:
    return " ".join(text.lower().split())


def shingles(text: str, n: int = 5) -> list[str]:
    tokens = normalize_for_hash(text).split()
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def document_minhash(text: str, num_perm: int = 128, ngram: int = 5) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for shingle in shingles(text, n=ngram):
        mh.update(shingle.encode("utf-8"))
    return mh


def exact_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


@dataclass
class DedupIndex:
    threshold: float = 0.80
    num_perm: int = 128
    ngram: int = 5
    lsh: MinHashLSH = field(init=False)
    exact: set[str] = field(default_factory=set)
    kept: int = 0
    dropped_exact: int = 0
    dropped_near: int = 0

    def __post_init__(self) -> None:
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

    def is_duplicate(self, doc_id: str, text: str) -> tuple[bool, str]:
        digest = exact_hash(text)
        if digest in self.exact:
            self.dropped_exact += 1
            return True, "exact"
        mh = document_minhash(text, num_perm=self.num_perm, ngram=self.ngram)
        neighbors = self.lsh.query(mh)
        if neighbors:
            self.dropped_near += 1
            return True, "near"
        self.exact.add(digest)
        self.lsh.insert(doc_id, mh)
        self.kept += 1
        return False, "keep"

    def stats(self) -> dict[str, int]:
        return {
            "kept": self.kept,
            "dropped_exact": self.dropped_exact,
            "dropped_near": self.dropped_near,
            "seen": self.kept + self.dropped_exact + self.dropped_near,
        }

    def dump_stats(self, path: Path) -> None:
        path.write_text(json.dumps(self.stats(), indent=2) + "\n", encoding="utf-8")
