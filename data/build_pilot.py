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

from tokenizer.corpus_friendship import analyze_friendship

from .sources import (
    PILOT_BUCKET_WEIGHTS,
    PILOT_LANGUAGE_QUOTAS,
    PILOT_MIX,
    PRODUCTION_MIX_V1,
    assert_mix_weights,
)


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


def _budget_progress(counts: dict, caps: dict) -> dict:
    return {
        key: {"tokens": counts.get(key, 0), "cap": cap,
              "progress": counts.get(key, 0) / cap if cap else 1.0}
        for key, cap in caps.items()
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
    *,
    resume: bool = False,
    overwrite: bool = False,
    tail_tokens: int = 2048,
    max_source_docs: int = 5_000_000,
) -> dict:
    from .pilot_state import PilotStore, build_lock, shard_files

    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if min(max_tokens, shard_size, max_source_docs) <= 0 or max_disk_gb <= 0:
        raise ValueError("Token, shard, scan and disk limits must be positive")
    if min(tokenizer_max_chars, tail_tokens) < 0:
        raise ValueError("Corpus and tail limits must be nonnegative")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus = tokenizer_corpus_dir / "pilot_corpus.txt"
    with build_lock(output_dir / ".pilot.lock"), build_lock(tokenizer_corpus_dir / ".pilot-corpus.lock"):
        existing = shard_files(output_dir)
        managed = [output_dir / name for name in (
            "pilot_state.json", "pilot_index.sqlite3", "pilot_index.sqlite3-journal",
            "pilot_report.json", "tokenizer_friendship.json", "pilot_corpus_journal.json",
        )]
        if not resume and not overwrite and (
            existing or any(p.exists() for p in managed) or (corpus.exists() and corpus.stat().st_size)
        ):
            raise FileExistsError("Existing pilot data/corpus found. Use --resume; --overwrite explicitly destroys this pilot.")
        if overwrite:
            for path in existing + managed + [corpus]:
                path.unlink(missing_ok=True)
        if cache_dir:
            import os
            os.environ["HF_HOME"] = cache_dir
            os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir) / "datasets")
        store = PilotStore(output_dir, corpus, tokenizer_max_chars)
        try:
            mode = store.restore(resume)
            return _build_pilot_locked(
                store, max_tokens, max_disk_gb, shard_size, tokenizer_dir,
                resume, mode, tail_tokens, max_source_docs,
            )
        finally:
            store.close()


def _build_pilot_locked(
    store, max_tokens, max_disk_gb, shard_size, tokenizer_dir,
    resume, mode, tail_tokens, max_source_docs,
) -> dict:
    import time
    from tqdm import tqdm

    from .clean import clean_record, iter_source
    from .filters.quality import QualityThresholds
    from .pilot_state import (
        PilotDedup, account_document, atomic_json, estimate_tokens, increment,
        lang_bucket, publish_shard, shard_files,
    )

    output_dir, corpus_file, state = store.output, store.corpus, store.state
    bucket_caps = {k: int(max_tokens * w) for k, w in PILOT_BUCKET_WEIGHTS.items()}
    lang_caps = {k: int(max_tokens * w) for k, w in PILOT_LANGUAGE_QUOTAS.items()}
    for counts, caps in ((state["bucket_counts"], bucket_caps), (state["lang_counts"], lang_caps)):
        if any(value > caps.get(key, 0) for key, value in counts.items()):
            raise ValueError("Existing data exceeds requested quotas; preserving shards. Resume with compatible --max-tokens.")
    if state["total_tokens"] > max_tokens:
        raise ValueError("Existing data exceeds --max-tokens")
    initial_docs = state["written_docs"]
    initial_tokens = state["total_tokens"]
    print(f"[pilot] restore={mode} docs={initial_docs:,} tokens_est={initial_tokens:,} "
          f"next_shard={state['shard_idx']}", flush=True)
    if resume:
        print("[pilot resume] replaying HF sources from the beginning; saved IDs/text hashes "
              "and MinHash signatures prevent duplicate output. No remote cursor assumed.", flush=True)
    if not state["scan_counts_complete"]:
        print("[pilot resume] historical scanned/dropped counts are unknown; acceptance/ETA "
              "use this session. Historical seen is a lower bound.", flush=True)
    dedup = PilotDedup(store)
    quality = QualityThresholds()
    buffer: list[dict] = []
    signatures: list[bytes] = []
    stats = state["stats"]
    state["source_stops"] = {}
    budget = {"max_tokens": max_tokens, "bucket_caps": bucket_caps, "language_caps": lang_caps,
              "tail_tokens": tail_tokens}
    bucket_stops = dict(state.get("bucket_stops", {})) if state.get("budget") == budget else {}
    state["bucket_stops"] = bucket_stops
    state["budget"] = budget
    state["tail_policy"] = "stop before first eligible overflowing document; stop at small remainder"
    session = {"seen": 0, "kept": 0, "dropped": 0}
    stop_reason = "sources_exhausted"
    interrupted = False
    publishing = False
    started = time.monotonic()

    def flush() -> None:
        nonlocal publishing
        publishing = True
        if not buffer:
            store.checkpoint()
            publishing = False
            return
        # Do not catch publication errors here: stale checkpoint recovery must
        # reconstruct from Parquet rather than retry this buffer at a new index.
        publish_shard(output_dir, state["shard_idx"], buffer)
        state["shard_idx"] += 1
        for row, signature in zip(buffer, signatures):
            store.remember(row, signature)
        store.append_corpus(buffer)
        store.checkpoint()
        buffer.clear()
        signatures.clear()
        publishing = False

    for name, spec in PILOT_MIX.items():
        bucket = spec["bucket"]
        cap = bucket_caps[bucket]
        # At most 0.1% of a bucket or tail_tokens (whichever is smaller).
        tail = min(tail_tokens, cap // 1000)
        remaining = cap - state["bucket_counts"].get(bucket, 0)
        if bucket in bucket_stops or remaining <= tail:
            state["source_stops"][name] = bucket_stops.setdefault(
                bucket, "bucket_cap" if remaining == 0 else "bucket_tail")
            continue
        forced_lang = lang_bucket(spec["language"]) if spec.get("language") else None
        if forced_lang and state["lang_counts"].get(forced_lang, 0) >= lang_caps.get(forced_lang, 0):
            state["source_stops"][name] = "language_cap"
            continue
        source_session = {"seen": 0, "kept": 0, "dropped": 0}
        source_scan = state["source_scan"].setdefault(name, {"seen": 0, "kept": 0, "dropped": 0})
        initial_bucket = state["bucket_counts"].get(bucket, 0)
        source_started = time.monotonic()
        last_refresh = 0.0
        reason = "source_exhausted"
        bar = tqdm(total=cap, initial=initial_bucket, desc=f"{name} | {bucket}",
                   unit="tok", unit_scale=True, dynamic_ncols=True,
                   bar_format="{desc}: {n_fmt}/{total_fmt} tokens [{percentage:6.2f}%] {postfix}")

        def progress(force: bool = False) -> None:
            nonlocal last_refresh
            now = time.monotonic()
            if not force and now - last_refresh < 0.5:
                return
            last_refresh = now
            current = state["bucket_counts"].get(bucket, 0)
            rate = (current - initial_bucket) / max(now - source_started, 1e-9)
            eta = f"{(cap - current) / rate:.0f}s" if rate > 0 else "unknown"
            acceptance = source_session["kept"] / max(source_session["seen"], 1)
            bar.n = current
            bar.set_postfix_str(
                f"source_tok={state['source_counts'].get(name, 0):,} "
                f"stored_docs={state['source_docs'].get(name, 0):,} "
                f"seen={source_session['seen']:,} kept={source_session['kept']:,} "
                f"dropped={source_session['dropped']:,} acceptance={acceptance:.2%} "
                f"tok/s={rate:.0f} ETA={eta} (session)", refresh=True)

        def dropped(key: str) -> None:
            increment(stats, "dropped")
            increment(stats, key)
            increment(session, "dropped")
            increment(source_session, "dropped")
            increment(source_scan, "dropped")
            progress()

        iterator = iter(iter_source(spec, None))
        try:
            while True:
                progress()
                remaining = cap - state["bucket_counts"].get(bucket, 0)
                if state["total_tokens"] >= max_tokens:
                    reason = stop_reason = "tokens"
                    break
                if remaining <= tail:
                    reason = "bucket_cap" if remaining == 0 else "bucket_tail"
                    bucket_stops[bucket] = reason
                    break
                # Finite raw scan limit also covers all-filtered/all-duplicate
                # streams. It includes replay and is explicitly reported.
                if source_session["seen"] >= max_source_docs:
                    reason = "source_scan_limit"
                    break
                if _disk_gb(output_dir) > max_disk_gb:
                    reason = stop_reason = "disk"
                    break
                try:
                    row = next(iterator)
                except StopIteration:
                    break
                for counter in (stats, session, source_session, source_scan):
                    increment(counter, "seen")
                cleaned = clean_record(
                    row, text_field=spec.get("text_field", "text"),
                    domain=spec.get("domain", "web"), dedup=dedup, quality=quality,
                    path_suffixes=spec.get("path_suffixes"), keyword_any=spec.get("keyword_any"),
                    forced_language=spec.get("language"),
                )
                if cleaned is None:
                    dropped("dropped_filter_or_duplicate")
                    continue
                lang = lang_bucket(str(cleaned.get("language") or "und"))
                tokens = estimate_tokens(cleaned)
                lang_remaining = lang_caps.get(lang, 0) - state["lang_counts"].get(lang, 0)
                if lang_remaining <= 0:
                    dropped("dropped_lang_quota")
                    if forced_lang:
                        reason = "language_cap"
                        break
                    continue
                if tokens > remaining or tokens > max_tokens - state["total_tokens"]:
                    # Whole-document policy: never hunt for a smaller document.
                    dropped("dropped_bucket_tail")
                    reason = "bucket_document_boundary"
                    bucket_stops[bucket] = reason
                    break
                if tokens > lang_remaining:
                    dropped("dropped_lang_quota")
                    if forced_lang:
                        reason = "language_cap_or_document_boundary"
                        break
                    continue
                cleaned.update(source=name, bucket=bucket)
                buffer.append(cleaned)
                signatures.append(dedup.last_signature)
                account_document(state, cleaned)
                for counter in (session, source_session, source_scan):
                    increment(counter, "kept")
                if len(buffer) >= shard_size:
                    flush()
        except KeyboardInterrupt:
            if publishing:
                # Publication may already have committed a shard. Leave the
                # buffer alone; --resume will reconcile it from disk.
                raise
            reason = stop_reason = "interrupted"
            interrupted = True
        finally:
            progress(force=True)
            bar.close()
            close = getattr(iterator, "close", None)
            if close:
                close()
        state["source_stops"][name] = reason
        flush()
        if interrupted or stop_reason in {"disk", "tokens"}:
            break
    if stop_reason == "sources_exhausted":
        if state["total_tokens"] >= max_tokens:
            stop_reason = "tokens"
        elif any(reason == "source_scan_limit" for reason in state["source_stops"].values()):
            stop_reason = "source_scan_limit"
        elif len(bucket_stops) == len(bucket_caps):
            stop_reason = "bucket_caps_or_tails"
    state["stop_reason"] = stop_reason
    flush()
    total_tokens = state["total_tokens"]
    bucket_counts, lang_counts = state["bucket_counts"], state["lang_counts"]
    actual_buckets = _share_map(bucket_counts, total_tokens)
    actual_langs = _share_map(lang_counts, total_tokens)
    report = {
        "dataset": "SHINRA-COLAB-PILOT", "tokens_est": total_tokens, "max_tokens": max_tokens,
        "target_buckets": dict(PILOT_BUCKET_WEIGHTS), "actual_buckets": actual_buckets,
        "bucket_delta": _delta(actual_buckets, PILOT_BUCKET_WEIGHTS),
        "target_languages": dict(PILOT_LANGUAGE_QUOTAS), "actual_languages": actual_langs,
        "language_delta": _delta(actual_langs, PILOT_LANGUAGE_QUOTAS),
        "buckets_tokens": dict(bucket_counts), "languages_tokens": dict(lang_counts),
        "sources_tokens": state["source_counts"], "scripts_tokens": state["script_counts"],
        "buckets": _budget_progress(bucket_counts, bucket_caps),
        "languages": _budget_progress(lang_counts, lang_caps),
        "special_token_leaks": state["leak_counts"], "filter": dict(stats), "dedup": dedup.stats(),
        "seen": stats["seen"], "kept": state["written_docs"], "dropped": stats["dropped"],
        "scan_counts_complete": state["scan_counts_complete"],
        "seen_is_lower_bound": not state["scan_counts_complete"],
        "acceptance_rate": (stats["kept"] / max(stats["seen"], 1)) if state["scan_counts_complete"] else None,
        "session": {**session, "acceptance_rate": session["kept"] / max(session["seen"], 1),
                    "tokens_est": total_tokens - initial_tokens,
                    "tokens_per_second": (total_tokens - initial_tokens) / max(time.monotonic() - started, 1e-9)},
        "source_scan": state["source_scan"], "source_stops": state["source_stops"],
        "stop_reason": stop_reason, "max_source_docs": max_source_docs,
        "tail_policy": state["tail_policy"], "tail_tokens": tail_tokens,
        "tokenizer_corpus_chars": state["corpus_chars"], "shards": len(shard_files(output_dir)),
        "next_shard_idx": state["shard_idx"], "output_dir": str(output_dir),
        "resume": {"requested": resume, "mode": mode, "initial_docs": initial_docs,
                   "initial_tokens_est": initial_tokens, "source_strategy": "replay_with_persistent_dedup"},
        "base_v1_not_this_run": dict(PRODUCTION_MIX_V1),
    }
    # Publish the resumable report before optional tokenizer diagnostics. An
    # unavailable tokenizer or interrupted diagnostic must not hide saved work.
    atomic_json(output_dir / "pilot_report.json", report)
    friendship = analyze_friendship(
        corpus_path=corpus_file, parquet_dir=output_dir, tokenizer_dir=tokenizer_dir,
        extra={"actual_buckets": actual_buckets, "actual_languages": actual_langs,
               "bucket_delta": report["bucket_delta"], "language_delta": report["language_delta"]},
    )
    report["tokenizer_friendship"] = {
        "verdict": friendship["verdict"], "dna": friendship["dna"],
        "corpus": {k: friendship["corpus"].get(k) for k in (
            "exists", "lines", "chars", "script_shares", "code_like_share", "math_like_share",
            "json_like_share", "cjk_share", "special_leak_rate", "approx_bytes_per_whitespace_token",
        )},
        "trained_tokenizer": friendship["trained_tokenizer"],
    }
    atomic_json(output_dir / "tokenizer_friendship.json", friendship)
    atomic_json(output_dir / "pilot_report.json", report)
    if interrupted:
        raise KeyboardInterrupt("Pilot saved; continue with --resume")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SHINRA-COLAB-PILOT shards")
    parser.add_argument("--output-dir", default="data/clean/pilot")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Preserve shards and resume with persistent deduplication")
    mode.add_argument("--overwrite", action="store_true", help="DESTRUCTIVE: remove this pilot's shards, state and corpus")
    parser.add_argument("--tail-tokens", type=int, default=2048,
                        help="Stop at this remaining bucket budget (limited to 0.1%% of bucket cap)")
    parser.add_argument("--max-source-docs", type=int, default=5_000_000,
                        help="Finite raw scan limit per source per invocation, including resume replay")
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
        resume=args.resume,
        overwrite=args.overwrite,
        tail_tokens=args.tail_tokens,
        max_source_docs=args.max_source_docs,
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
