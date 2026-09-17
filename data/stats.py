"""Corpus statistics for cleaned SHINRA shards."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def write_corpus_stats(input_dir: Path, output_path: Path) -> dict:
    files = sorted(input_dir.glob("*.parquet"))
    languages: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    total_chars = 0
    total_docs = 0
    quality_sum = 0.0
    for file in files:
        table = pq.read_table(file)
        n = table.num_rows
        total_docs += n
        cols = set(table.column_names)
        if "n_chars" in cols:
            total_chars += int(sum(table.column("n_chars").to_pylist()))
        if "language" in cols:
            languages.update([x for x in table.column("language").to_pylist() if x])
        if "domain" in cols:
            domains.update([x for x in table.column("domain").to_pylist() if x])
        if "source" in cols:
            sources.update([x for x in table.column("source").to_pylist() if x])
        if "quality_score" in cols:
            quality_sum += float(sum(x or 0.0 for x in table.column("quality_score").to_pylist()))
    stats = {
        "documents": total_docs,
        "chars": total_chars,
        "approx_tokens_4bpt": int(total_chars / 4),
        "mean_quality": (quality_sum / total_docs) if total_docs else 0.0,
        "languages": dict(languages.most_common()),
        "domains": dict(domains.most_common()),
        "sources": dict(sources.most_common()),
        "shards": len(files),
    }
    output_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return stats
