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

from tokenizer.corpus_friendship import analyze_friendship, write_friendship_report
from tokenizer.special_tokens import ALL_SPECIAL_TOKENS

from .sources import (
    PILOT_BUCKET_WEIGHTS,
    PILOT_LANGUAGE_QUOTAS,
    PILOT_MIX,
    PRODUCTION_MIX_V1,
    assert_mix_weights,
)


def _lang_bucket(code: str) -> str:
    if code in {"en", "eng"}:
        return "en"
    if code == "ru":
        return "ru"
    if code in {"de", "fr", "es", "it", "pt", "nl", "pl", "sv", "cs", "ro", "fi", "hu", "da", "nb", "no"}:
        return "eu"
    return "drop"


def _disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    used = (usage.total - usage.free) / (1024**3)
    return used


def _share_map(counts: Counter[str] | dict[str, int], total: int) -> dict[str, float]:
    denom = float(total) or 1.0
    return {k: round(v / denom, 6) for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))}


def _delta(actual: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(actual) | set(target))
    return {k: round(actual.get(k, 0.0) - target.get(k, 0.0), 6) for k in keys}


def plan_pilot(max_tokens: int) -> dict:
    weight_audit = assert_mix_weights()
    return {
        "dataset": "SHINRA-COLAB-PILOT",
        "max_tokens": max_tokens,
        "target_buckets": dict(PILOT_BUCKET_WEIGHTS),
        "target_languages": dict(PILOT_LANGUAGE_QUOTAS),
        "bucket_caps": {k: int(max_tokens * w) for k, w in PILOT_BUCKET_WEIGHTS.items()},
        "language_caps": {k: int(max_tokens * w) for k, w in PILOT_LANGUAGE_QUOTAS.items()},
        "sources": {
            name: {
                "hf_id": spec["hf_id"],
                "subset": spec.get("subset"),
                "weight": spec["weight"],
                "bucket": spec["bucket"],
                "domain": spec.get("domain"),
                "language": spec.get("language"),
            }
            for name, spec in PILOT_MIX.items()
        },
        "base_v1_not_this_run": dict(PRODUCTION_MIX_V1),
        "weight_audit": weight_audit,
        "not": [
            "FineWeb-Edu 1.3T",
            "TinyStories",
            "Chinese instruction dumps",
            "200B BASE lake",
        ],
    }


def build_pilot(
    output_dir: Path,
    tokenizer_corpus_dir: Path,
    max_tokens: int,
    max_disk_gb: float,
    shard_size: int,
    tokenizer_max_chars: int,
    cache_dir: str | None,
    tokenizer_dir: Path | None = None,
) -> dict:
    from datasets import Dataset
    from tqdm import tqdm

    from .clean import clean_record, iter_source
    from .dedup import DedupIndex
    from .filters.quality import QualityThresholds

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
    source_counts: Counter[str] = Counter()
    script_counts: Counter[str] = Counter()
    leak_counts: Counter[str] = Counter()
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
                    path_suffixes=spec.get("path_suffixes"),
                    keyword_any=spec.get("keyword_any"),
                    forced_language=spec.get("language"),
                )
                if cleaned is None:
                    stats["dropped"] += 1
                    continue
                lang = _lang_bucket(str(cleaned.get("language") or "und"))
                if lang not in lang_caps or lang_counts[lang] >= lang_caps[lang]:
                    stats["dropped_lang_quota"] += 1
                    continue
                est_tokens = max(cleaned["n_chars"] // 4, 1)
                if bucket_counts[bucket] + est_tokens > bucket_caps[bucket]:
                    continue
                text = cleaned["text"]
                for token in ALL_SPECIAL_TOKENS:
                    if token in text:
                        leak_counts[token] += 1
                        stats["special_leaks"] += 1
                cleaned["source"] = name
                cleaned["bucket"] = bucket
                buffer.append(cleaned)
                bucket_counts[bucket] += est_tokens
                lang_counts[lang] += est_tokens
                source_counts[name] += est_tokens
                script_counts[str(cleaned.get("script") or "und")] += est_tokens
                total_tokens += est_tokens
                stats["kept"] += 1
                if spec.get("domain") == "code":
                    stats["code_docs"] += 1
                if spec.get("bucket") == "math_stem":
                    stats["math_docs"] += 1
                if corpus_chars < tokenizer_max_chars:
                    line = text.replace("\n", " ")[:8000]
                    corpus_handle.write(line + "\n")
                    corpus_chars += len(line)
                if len(buffer) >= shard_size:
                    flush()
            if total_tokens >= max_tokens:
                break
            if stats.get("stop_reason") == "disk":
                break
    flush()
    actual_buckets = _share_map(bucket_counts, total_tokens)
    actual_langs = _share_map(lang_counts, total_tokens)
    report = {
        "dataset": "SHINRA-COLAB-PILOT",
        "tokens_est": total_tokens,
        "max_tokens": max_tokens,
        "target_buckets": dict(PILOT_BUCKET_WEIGHTS),
        "actual_buckets": actual_buckets,
        "bucket_delta": _delta(actual_buckets, PILOT_BUCKET_WEIGHTS),
        "target_languages": dict(PILOT_LANGUAGE_QUOTAS),
        "actual_languages": actual_langs,
        "language_delta": _delta(actual_langs, PILOT_LANGUAGE_QUOTAS),
        "buckets_tokens": dict(bucket_counts),
        "languages_tokens": dict(lang_counts),
        "sources_tokens": dict(source_counts),
        "scripts_tokens": dict(script_counts),
        "special_token_leaks": dict(leak_counts),
        "filter": dict(stats),
        "dedup": dedup.stats(),
        "tokenizer_corpus_chars": corpus_chars,
        "shards": shard_idx,
        "output_dir": str(output_dir),
        "base_v1_not_this_run": dict(PRODUCTION_MIX_V1),
    }
    friendship = analyze_friendship(
        corpus_path=corpus_file,
        parquet_dir=output_dir,
        tokenizer_dir=tokenizer_dir,
        extra={
            "actual_buckets": actual_buckets,
            "actual_languages": actual_langs,
            "bucket_delta": report["bucket_delta"],
            "language_delta": report["language_delta"],
        },
    )
    report["tokenizer_friendship"] = {
        "verdict": friendship["verdict"],
        "dna": friendship["dna"],
        "corpus": {
            k: friendship["corpus"].get(k)
            for k in (
                "exists",
                "lines",
                "chars",
                "script_shares",
                "code_like_share",
                "math_like_share",
                "json_like_share",
                "cjk_share",
                "special_leak_rate",
                "approx_bytes_per_whitespace_token",
            )
        },
        "trained_tokenizer": friendship["trained_tokenizer"],
    }
    (output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_friendship_report(friendship, output_dir / "tokenizer_friendship.json")
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
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print mix/DNA plan without streaming Hugging Face datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plan:
        plan = plan_pilot(args.max_tokens)
        friendship = analyze_friendship(
            tokenizer_dir=Path(args.tokenizer) if args.tokenizer else Path("tokenizer/artifacts"),
        )
        payload = {"plan": plan, "tokenizer_friendship": friendship}
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "pilot_plan.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, default=str))
        return
    report = build_pilot(
        output_dir=Path(args.output_dir),
        tokenizer_corpus_dir=Path(args.tokenizer_corpus_dir),
        max_tokens=args.max_tokens,
        max_disk_gb=args.max_disk_gb,
        shard_size=args.shard_size,
        tokenizer_max_chars=args.tokenizer_max_chars,
        cache_dir=args.cache_dir,
        tokenizer_dir=Path(args.tokenizer) if args.tokenizer else Path("tokenizer/artifacts"),
    )
    if args.tokenizer and args.pack_dir:
        from .pack import pack_documents

        report["pack"] = pack_documents(
            input_dir=Path(args.output_dir),
            tokenizer_path=Path(args.tokenizer),
            output_dir=Path(args.pack_dir),
            sequence_length=args.sequence_length,
        )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
