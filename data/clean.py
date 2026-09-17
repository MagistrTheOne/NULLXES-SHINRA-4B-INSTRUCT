"""Document cleaning pipeline: normalize → quality → language → toxicity → code → dedup."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import ftfy
from datasets import Dataset, load_dataset
from tqdm import tqdm

from .dedup import DedupIndex, exact_hash
from .filters.code import score_code
from .filters.language import detect_language
from .filters.quality import QualityThresholds, score_document
from .filters.toxicity import score_toxicity
from .sources import PRETRAIN_MIX


def normalize_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_text(record: dict, text_field: str) -> str:
    value = record.get(text_field)
    if isinstance(value, str):
        return value
    if isinstance(record.get("messages"), list):
        parts = []
        for msg in record["messages"]:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                parts.append(msg["content"])
        return "\n".join(parts)
    return ""


def clean_record(
    record: dict,
    text_field: str,
    domain: str,
    dedup: DedupIndex,
    quality: QualityThresholds,
    path_field: str | None = "path",
    path_suffixes: list[str] | None = None,
    keyword_any: list[str] | None = None,
    forced_language: str | None = None,
) -> dict | None:
    path_val = record.get(path_field) if path_field else None
    if path_suffixes:
        path_str = str(path_val or "")
        if not any(path_str.endswith(suffix) for suffix in path_suffixes):
            return None
    text = normalize_text(extract_text(record, text_field))
    if not text:
        return None
    if keyword_any:
        hay = text[:12000].lower()
        if not any(key.lower() in hay for key in keyword_any):
            return None
    lang = detect_language(text)
    if not lang["keep"]:
        return None
    tox = score_toxicity(text)
    if not tox["keep"]:
        return None
    path_str = str(path_val or "")
    treat_as_code = domain == "code" or path_str.endswith((".cu", ".cuh", ".py", ".c", ".cpp"))
    if treat_as_code:
        code = score_code(text, path=path_str or None)
        if not code["keep"]:
            return None
        quality_info = {"keep": True, "quality_score": 1.0, "drop_reasons": []}
    else:
        quality_info = score_document(text, quality)
        if not quality_info["keep"]:
            return None
        code = None
    doc_id = str(record.get("id") or exact_hash(text))
    is_dup, reason = dedup.is_duplicate(doc_id, text)
    if is_dup:
        return None
    return {
        "id": doc_id,
        "text": text,
        "domain": domain,
        "language": forced_language or lang["language"],
        "script": lang["script"],
        "quality_score": quality_info.get("quality_score", 1.0),
        "toxicity": tox["toxicity"],
        "code_language": None if code is None else code["language"],
        "n_chars": len(text),
        "n_words": quality_info.get("words", len(text.split())),
        "dedup": reason,
    }


def iter_source(spec: dict, max_docs: int | None) -> Iterator[dict]:
    load_kwargs: dict = {"path": spec["hf_id"], "split": spec.get("split", "train")}
    if spec.get("subset"):
        load_kwargs["name"] = spec["subset"]
    load_kwargs["streaming"] = True
    dataset = load_dataset(**load_kwargs)
    for i, row in enumerate(dataset):
        if max_docs is not None and i >= max_docs:
            break
        yield row


def run_clean(
    output_dir: Path,
    sources: dict[str, dict] | None = None,
    max_docs_per_source: int | None = None,
    shard_size: int = 10_000,
) -> dict:
    sources = sources or PRETRAIN_MIX
    output_dir.mkdir(parents=True, exist_ok=True)
    dedup = DedupIndex()
    quality = QualityThresholds()
    stats: Counter[str] = Counter()
    buffer: list[dict] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal buffer, shard_idx
        if not buffer:
            return
        ds = Dataset.from_list(buffer)
        path = output_dir / f"clean-{shard_idx:05d}.parquet"
        ds.to_parquet(str(path))
        stats["shards"] += 1
        stats["written"] += len(buffer)
        buffer = []
        shard_idx += 1

    for name, spec in sources.items():
        domain = spec.get("domain", "web")
        text_field = spec.get("text_field", "text")
        print(f"[clean] source={name} domain={domain}", flush=True)
        for row in tqdm(iter_source(spec, max_docs_per_source), desc=name):
            stats["seen"] += 1
            cleaned = clean_record(
                row,
                text_field=text_field,
                domain=domain,
                dedup=dedup,
                quality=quality,
                path_suffixes=spec.get("path_suffixes"),
                keyword_any=spec.get("keyword_any"),
                forced_language=spec.get("language"),
            )
            if cleaned is None:
                stats["dropped"] += 1
                continue
            cleaned["source"] = name
            buffer.append(cleaned)
            if len(buffer) >= shard_size:
                flush()
    flush()
    report = {"filter": dict(stats), "dedup": dedup.stats()}
    (output_dir / "clean_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    dedup.dump_stats(output_dir / "dedup_stats.json")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean SHINRA pretrain corpora")
    parser.add_argument("--output-dir", default="data/clean/pretrain")
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_clean(
        Path(args.output_dir),
        max_docs_per_source=args.max_docs_per_source,
        shard_size=args.shard_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
