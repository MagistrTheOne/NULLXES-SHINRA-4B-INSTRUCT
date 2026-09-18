"""Deficit scheduling for the existing pilot recipe; never alter data labels/caps.

Primary languages are scheduling hints, not forced document classifications.
FineWeb/code/OpenWebMath are not suitable ways to backfill a Russian/European
quota once English is closed, even if their filters occasionally find outliers.
Unknown sources remain eligible while any language has room.
"""

from __future__ import annotations

from .pilot_state import lang_bucket


PRIMARY_LANGUAGES = {
    "fineweb_edu_10bt": "en",
    "python_edu": "en",
    "stack_smol": "en",
    "openwebmath": "en",
}


def primary_language(name: str, spec: dict) -> str | None:
    return lang_bucket(spec["language"]) if spec.get("language") else PRIMARY_LANGUAGES.get(name)


def tail_limit(cap: int, tail_tokens: int) -> int:
    return min(tail_tokens, cap // 1000)


def rank_sources(
    mix: dict, state: dict, bucket_caps: dict, lang_caps: dict,
    max_tokens: int, tail_tokens: int, bucket_stops: dict,
    finished: dict, language_stops: dict,
) -> tuple[list[str], dict[str, str], dict[str, dict]]:
    """Recalculate eligibility and deficits after each bounded source visit.

    Rank by relative bucket deficit, relative source recipe deficit, then
    language deficit. Source weights are soft priorities, not new hard caps.
    The caller gives every eligible source one visit per round, so rejected
    streams cannot starve other buckets or Wikipedia languages.
    """
    ranked, skipped, deficits = [], {}, {}
    for order, (name, spec) in enumerate(mix.items()):
        bucket = spec["bucket"]
        cap = bucket_caps[bucket]
        remaining = cap - state["bucket_counts"].get(bucket, 0)
        if bucket in bucket_stops or remaining <= tail_limit(cap, tail_tokens):
            skipped[name] = bucket_stops.get(bucket) or ("bucket_cap" if remaining <= 0 else "bucket_tail")
            continue
        if name in finished:
            skipped[name] = finished[name]
            continue
        language = primary_language(name, spec)
        languages = [language] if language else list(lang_caps)
        available = {
            lang: max(0, lang_caps.get(lang, 0) - state["lang_counts"].get(lang, 0))
            for lang in languages
            if lang not in language_stops
        }
        open_languages = {
            lang: value for lang, value in available.items()
            if value > tail_limit(lang_caps.get(lang, 0), tail_tokens)
        }
        if not open_languages:
            skipped[name] = "language_cap_or_tail"
            continue
        target = int(max_tokens * spec["weight"]) if "weight" in spec else cap
        source_remaining = max(0, target - state["source_counts"].get(name, 0))
        lang_fraction = max(value / lang_caps[lang] for lang, value in open_languages.items())
        deficits[name] = {
            "bucket": bucket, "bucket_remaining": remaining,
            "source_target": target, "source_remaining": source_remaining,
            "primary_language": language, "language_remaining": open_languages,
        }
        ranked.append(((remaining / cap, source_remaining / max(target, 1), lang_fraction, -order), name))
    ranked.sort(reverse=True)
    return [name for _, name in ranked], skipped, deficits


def quota_conflicts(mix: dict, state: dict, bucket_caps: dict, lang_caps: dict,
                    bucket_stops: dict, tail_tokens: int) -> list[dict]:
    """Expose incompatible *primary-language* budgets without reallocating them.

    These diagnostics concern the declared scheduling profiles, not a claim
    that every row in a source has the same language.
    """
    groups: dict[tuple[str, ...], dict] = {}
    for bucket, cap in bucket_caps.items():
        needed = cap - state["bucket_counts"].get(bucket, 0)
        if bucket in bucket_stops or needed <= tail_limit(cap, tail_tokens):
            continue
        sources = {name: spec for name, spec in mix.items() if spec["bucket"] == bucket}
        profiles = {primary_language(name, spec) for name, spec in sources.items()}
        if not profiles or None in profiles:
            continue
        key = tuple(sorted(profiles))
        group = groups.setdefault(key, {"buckets": {}, "sources": [], "required_tokens": 0})
        group["buckets"][bucket] = needed
        group["sources"].extend(sources)
        group["required_tokens"] += needed
    conflicts = []
    for languages, group in groups.items():
        available = sum(max(0, lang_caps.get(lang, 0) - state["lang_counts"].get(lang, 0)) for lang in languages)
        if group["required_tokens"] > available:
            conflicts.append({
                **group, "primary_languages": list(languages), "available_tokens": available,
                "shortfall_tokens": group["required_tokens"] - available,
                "reason": "bucket_targets_exceed_primary_language_capacity",
                "action": "Explicit recipe/quota revision is required to meet all targets; no automatic changes.",
            })
    return conflicts
