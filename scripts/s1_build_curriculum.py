"""Build the 20k S1 curriculum pilot. CPU only. No model weights and no GPU."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.s1.blacklist import diagnostic_blacklist
from data.s1.curriculum import (
    TARGET,
    assign_difficulties,
    family_targets,
    generate_native,
    language_quotas,
    tag_public,
)
from data.s1.foundation import ExactDedup, load_registry
from data.s1.loaders import (
    iter_danetqa,
    iter_entailment,
    iter_muserc,
    iter_rucos,
    iter_rwsd,
    iter_tydi_primary,
    iter_tydi_secondary,
)

OUT = ROOT / "data" / "s1" / "pilot_v1"
FORBIDDEN_SOURCES = ("xquad", "belebele", "rubq", "mmlu", "flores", "paws", "xnli")
FAMILIES = tuple(f"S1-{index:02d}" for index in range(1, 11))


def _offer(buckets: dict, record: dict, cap: int, lang_cap: dict[str, int] | None = None) -> None:
    family = record["family"]
    language = record["language"]
    guessed = _script(record["messages"][0]["content"])
    if guessed and guessed != language:
        return
    if len(buckets[family]) >= cap:
        return
    if lang_cap is not None and sum(1 for row in buckets[family] if row["language"] == language) >= lang_cap.get(language, 0):
        return
    buckets[family].append(tag_public(record))


def collect_public() -> tuple[list[dict], dict]:
    targets = family_targets()
    buckets: dict[str, list] = defaultdict(list)
    notes = {}
    for record in iter_tydi_secondary(2200, 2200, scan_limit=400000):
        if record["family"] == "S1-08":
            _offer(buckets, record, targets["S1-08"], {"en": 500, "ru": 500})
        elif record["family"] == "S1-07":
            _offer(buckets, record, targets["S1-07"], {"en": 1000, "ru": 1000})
        if len(buckets["S1-08"]) >= targets["S1-08"] and len(buckets["S1-07"]) >= targets["S1-07"]:
            break
    for record in iter_tydi_primary(1200, 400, scan_limit=80000):
        if record["family"] == "S1-03":
            _offer(buckets, record, targets["S1-03"], {"en": 800, "ru": 400})
        elif record["family"] == "S1-07" and len(buckets["S1-07"]) < targets["S1-07"]:
            _offer(buckets, record, targets["S1-07"], {"en": 1000, "ru": 1000})
    for record in iter_danetqa(targets["S1-03"]):
        _offer(buckets, record, targets["S1-03"])
    terra_cap = min(2200, targets["S1-06"])
    for record in iter_entailment("rsg_terra", "TERRa", terra_cap):
        _offer(buckets, record, targets["S1-06"])
    for record in iter_entailment("rsg_rcb", "RCB", targets["S1-06"] - len(buckets["S1-06"])):
        _offer(buckets, record, targets["S1-06"])
    for record in iter_rucos(1200):
        _offer(buckets, record, targets["S1-05"])
    for record in iter_rwsd(400):
        _offer(buckets, record, targets["S1-05"])
    for record in iter_muserc(800):
        _offer(buckets, record, targets["S1-05"])
    rows = [record for family in ("S1-03", "S1-05", "S1-06", "S1-07", "S1-08") for record in buckets[family]]
    for family in ("S1-03", "S1-05", "S1-06", "S1-07", "S1-08"):
        if len(buckets[family]) != targets[family]:
            notes[family] = {"got": len(buckets[family]), "target": targets[family]}
    return rows, notes


def _script(text: str) -> str | None:
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 8:
        return None
    cyr = sum(1 for char in letters if "\u0400" <= char <= "\u04ff")
    return "ru" if cyr / len(letters) >= 0.4 else "en"


def quality(records: list[dict]) -> dict:
    registry = {source["source_id"] for source in load_registry()["sources"]}
    prompt_counts: dict[str, int] = defaultdict(int)
    pair_counts: dict[str, int] = defaultdict(int)
    checks = Counter()
    shinra = Counter()
    nullxes = Counter()
    for record in records:
        user = record["messages"][0]["content"]
        assistant = record["messages"][1]["content"]
        prompt_counts[user] += 1
        pair_counts[user + "\n" + assistant] += 1
        if not assistant.strip():
            checks["empty_target"] += 1
        if user == assistant:
            checks["identical_user_assistant"] += 1
        if any(message["role"] == "system" for message in record["messages"]):
            checks["system_messages"] += 1
        if record["family"] not in FAMILIES:
            checks["unexpected_family"] += 1
        if record["difficulty"] not in ("A", "B", "C", "D", "E"):
            checks["unexpected_difficulty"] += 1
        if record["source_split"] != "train":
            checks["forbidden_split"] += 1
        if record["source"] not in registry:
            checks["unapproved_source"] += 1
        if any(name in record["source"].lower() for name in FORBIDDEN_SOURCES):
            checks["unapproved_source"] += 1
        blob = user + "\n" + assistant
        if "SHINRA" in blob.upper():
            shinra[record["family"]] += 1
        if "NULLXES" in blob.upper():
            nullxes[record["family"]] += 1
        if record["family"] != "S1-10" and ("SHINRA" in blob.upper() or "NULLXES" in blob.upper()):
            checks["forbidden_entity"] += 1
        guessed = _script(user)
        if guessed and guessed != record["language"]:
            checks["language_mismatch"] += 1
        if len(assistant) >= 12 and assistant in user and record["metadata"].get("source_type") == "NATIVE" and record["family"] == "S1-10":
            checks["identity_target_in_prompt"] += 1
    checks["duplicate_prompts"] = sum(count - 1 for count in prompt_counts.values() if count > 1)
    checks["duplicate_prompt_target"] = sum(count - 1 for count in pair_counts.values() if count > 1)
    return {
        "checks": dict(checks),
        "shinra_by_family": dict(shinra),
        "nullxes_by_family": dict(nullxes),
    }


def audit_sample(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for record in sorted(records, key=lambda row: (row["family"], row["language"], row["id"])):
        key = (record["family"], record["language"])
        if len(grouped[key]) < 5:
            grouped[key].append(record)
    sample = [record for key in sorted(grouped) for record in grouped[key]]
    return sample


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    public, shortfalls = collect_public()
    quotas = language_quotas(public)
    native = generate_native(quotas)
    difficulty_gap = assign_difficulties(native, public)
    dedup = ExactDedup(diagnostic_blacklist())
    kept = [record for record in public + native if dedup.accept(record)]
    kept.sort(key=lambda row: row["id"])
    path = OUT / "s1_curriculum_pilot_v1.jsonl"
    payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in kept)
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    report = quality(kept)
    chars = sum(len(record["messages"][0]["content"]) + len(record["messages"][1]["content"]) for record in kept)
    families = Counter(record["family"] for record in kept)
    languages = Counter(record["language"] for record in kept)
    difficulties = Counter(record["difficulty"] for record in kept)
    sources = Counter(record["metadata"].get("source_type", "PUBLIC") for record in kept)
    targets = family_targets()
    family_gap = {family: families[family] - targets[family] for family in FAMILIES}
    sample = audit_sample(kept)
    (OUT / "audit_sample.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in sample),
        encoding="utf-8",
    )
    manifest = {
        "builder_version": "s1-curriculum-pilot-v1",
        "seed": "s1-curriculum-pilot-v1",
        "record_count": len(kept),
        "sha256": digest,
        "languages": dict(languages),
        "families": {family: families[family] for family in FAMILIES},
        "family_targets": targets,
        "family_gap": family_gap,
        "difficulties": {level: difficulties[level] for level in "ABCDE"},
        "difficulty_gap": difficulty_gap,
        "source_mix": dict(sources),
        "public_shortfalls": shortfalls,
        "duplicate_removed": dedup.removed,
        "diagnostic_blocked": dedup.blocked,
        "approx_characters": chars,
        "approx_tokens_chars_div_4": chars // 4,
        "token_estimate": "approximation chars//4; SHINRA tokenizer not used",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(kept), "sha256": digest, "family_gap": family_gap, "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
