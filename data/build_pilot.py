"""Token-budget SHINRA-COLAB-PILOT builder.

Streams HF sample configs, applies the production filters, stops at a token
cap or disk cap. Does not download FineWeb 1.3T.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm

from .clean import clean_record, iter_source
from .dedup import DedupIndex
from .filters.quality import QualityThresholds
from .pack import pack_documents
from .sources import PILOT_BUCKET_WEIGHTS, PILOT_LANGUAGE_QUOTAS, PILOT_MIX


def _lang_bucket(code: str) -> str:
    if code in {"en", "eng"}:
        return "en"
    if code in {"ru", "uk", "be"}:
        return "ru" if code == "ru" else "other"
    if code in {"zh", "zh-cn", "zh-hans", "ja"}:
        return "zh" if code.startswith("zh") else "other"
    return "other"


def _disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    used = (usage.total - usage.free) / (1024**3)
    return used


def build_pilot(
    output_dir: Path,
    tokenizer_corpus_dir: Path,
    max_tokens: int,
    max_disk_gb: float,
    shard_size: int,
    tokenizer_max_chars: int,
    cache_dir: str | None,
) -> dict:
    if cache_dir:
        import os

        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir) / "datasets")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = tokenizer_corpus_dir / "pilot_corpus.txt"

    bucket_caps = {k: int(max_tokens * w) for k, w in PILOT_BUCKET_WEIGHTS.items()}
    lang_caps = {k: int(max_tokens * w) for k, w in PILOT_LANGUAGE_QUOTAS.items()}
    bucket_counts: Counter[str] = Counter()
    lang_counts: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    dedup = DedupIndex()
    quality = QualityThresholds()
    buffer: list[dict] = []
    shard_idx = 0
    total_tokens = 0
    corpus_chars = 0

    def flush() -> None:
        nonlocal buffer, shard_idx
        if not buffer:
            return
        Dataset.from_list(buffer).to_parquet(str(output_dir / f"clean-{shard_idx:05d}.parquet"))
        stats["shards"] += 1
        stats["written_docs"] += len(buffer)
        buffer = []
        shard_idx += 1

    with corpus_file.open("w", encoding="utf-8") as corpus_handle:
        for name, spec in PILOT_MIX.items():
            bucket = spec["bucket"]
            if bucket_counts[bucket] >= bucket_caps[bucket]:
                continue
            print(f"[pilot] source={name} bucket={bucket} cap={bucket_caps[bucket]}", flush=True)
            for row in tqdm(iter_source(spec, None), desc=name):
                if total_tokens >= max_tokens:
                    stats["stop_reason"] = "tokens"
                    break
                if _disk_gb(output_dir) > max_disk_gb:
                    stats["stop_reason"] = "disk"
                    break
                if bucket_counts[bucket] >= bucket_caps[bucket]:
                    break
                stats["seen"] += 1
                cleaned = clean_record(
                    row,
                    text_field=spec.get("text_field", "text"),
                    domain=spec.get("domain", "web"),
                    dedup=dedup,
                    quality=quality,
                )
                if cleaned is None:
                    stats["dropped"] += 1
                    continue
                lang = _lang_bucket(str(cleaned.get("language") or "und"))
                if lang_counts[lang] >= lang_caps.get(lang, lang_caps["other"]):
                    stats["dropped_lang_quota"] += 1
                    continue
                est_tokens = max(cleaned["n_chars"] // 4, 1)
                if bucket_counts[bucket] + est_tokens > bucket_caps[bucket]:
                    continue
                cleaned["source"] = name
                cleaned["bucket"] = bucket
                buffer.append(cleaned)
                bucket_counts[bucket] += est_tokens
                lang_counts[lang] += est_tokens
                total_tokens += est_tokens
                stats["kept"] += 1
                if corpus_chars < tokenizer_max_chars:
                    line = cleaned["text"].replace("\n", " ")[:8000]
                    corpus_handle.write(line + "\n")
                    corpus_chars += len(line)
                if len(buffer) >= shard_size:
                    flush()
            if total_tokens >= max_tokens:
                break
            if stats.get("stop_reason") == "disk":
                break
    flush()
    report = {
        "tokens_est": total_tokens,
        "max_tokens": max_tokens,
        "buckets": dict(bucket_counts),
        "languages": dict(lang_counts),
        "filter": dict(stats),
        "dedup": dedup.stats(),
        "tokenizer_corpus_chars": corpus_chars,
        "shards": shard_idx,
        "output_dir": str(output_dir),
    }
    (output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SHINRA-COLAB-PILOT shards")
    parser.add_argument("--output-dir", default="data/clean/pilot")
    parser.add_argument("--tokenizer-corpus-dir", default="tokenizer/corpus")
    parser.add_argument("--max-tokens", type=int, default=100_000_000)
    parser.add_argument("--max-disk-gb", type=float, default=150.0)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--tokenizer-max-chars", type=int, default=2_000_000_000)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--pack-dir", default=None)
    parser.add_argument("--sequence-length", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_pilot(
        output_dir=Path(args.output_dir),
        tokenizer_corpus_dir=Path(args.tokenizer_corpus_dir),
        max_tokens=args.max_tokens,
        max_disk_gb=args.max_disk_gb,
        shard_size=args.shard_size,
        tokenizer_max_chars=args.tokenizer_max_chars,
        cache_dir=args.cache_dir,
    )
    if args.tokenizer and args.pack_dir:
        report["pack"] = pack_documents(
            input_dir=Path(args.output_dir),
            tokenizer_path=Path(args.tokenizer),
            output_dir=Path(args.pack_dir),
            sequence_length=args.sequence_length,
        )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
