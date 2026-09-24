"""S1 curriculum pilot assembly. Native generation is deterministic and offline."""

from __future__ import annotations

from collections import Counter

from data.s1.foundation import FROZEN_FAMILY_PERCENT
from data.s1.native import (
    controlled_record,
    identity_record,
    instruction_record,
    paraphrase_record,
    relation_record,
    relation_reversal_pair,
)

TARGET = 20_000
NATIVE_FAMILIES = ("S1-01", "S1-02", "S1-04", "S1-09", "S1-10")
DIFFICULTIES = ("A", "B", "C", "D", "E")
# Frozen approximate mix from S1_DATA_SPEC.md section 5, as percent of records.
DIFFICULTY_PERCENT = {"A": 15, "B": 20, "C": 30, "D": 25, "E": 10}
REVERSAL_PAIRS = 100


def family_targets(total: int = TARGET) -> dict[str, int]:
    targets = {family: total * percent // 100 for family, percent in FROZEN_FAMILY_PERCENT.items()}
    gap = total - sum(targets.values())
    if gap:
        targets["S1-01"] += gap
    return targets


def difficulty_targets(total: int = TARGET) -> dict[str, int]:
    targets = {level: total * percent // 100 for level, percent in DIFFICULTY_PERCENT.items()}
    gap = total - sum(targets.values())
    if gap:
        targets["C"] += gap
    return targets


def _split_languages(count: int, en_budget: int, ru_budget: int) -> tuple[int, int]:
    count = min(count, max(0, en_budget) + max(0, ru_budget))
    if count <= 0 or en_budget + ru_budget <= 0:
        return 0, 0
    en = min(en_budget, int(round(count * en_budget / (en_budget + ru_budget))))
    en = max(0, min(count, en))
    ru = count - en
    if ru > ru_budget:
        en += ru - ru_budget
        ru = ru_budget
    if en > en_budget:
        ru += en - en_budget
        en = en_budget
    return en, ru


def language_quotas(public_records: list[dict], total: int = TARGET) -> dict[str, tuple[int, int]]:
    targets = family_targets(total)
    public_counts = Counter(record["family"] for record in public_records)
    en_public = sum(1 for record in public_records if record["language"] == "en")
    ru_public = sum(1 for record in public_records if record["language"] == "ru")
    en_budget = max(0, total // 2 - en_public)
    ru_budget = max(0, total // 2 - ru_public)
    quotas = {}
    for family in NATIVE_FAMILIES:
        need = max(0, targets[family] - public_counts[family])
        en, ru = _split_languages(need, en_budget, ru_budget)
        quotas[family] = (en, ru)
        en_budget -= en
        ru_budget -= ru
    return quotas


def _schedule(count: int, levels: list[str]) -> list[str]:
    if count > len(levels):
        raise ValueError(f"difficulty pool {len(levels)} < {count}")
    chosen = levels[:count]
    del levels[:count]
    return chosen


def generate_native(quotas: dict[str, tuple[int, int]]) -> list[dict]:
    """Build native rows. Open difficulty labels are assigned later."""
    records: list[dict] = []

    def take_difficulty(forced: str | None = None) -> str:
        return forced or "C"

    en_rev, ru_rev = _split_languages(REVERSAL_PAIRS, quotas.get("S1-02", (0, 0))[0] // 2, quotas.get("S1-02", (0, 0))[1] // 2)
    for language, pairs in (("en", en_rev), ("ru", ru_rev)):
        for index in range(pairs):
            forward, backward = relation_reversal_pair(index, language)
            records.append(forward)
            records.append(backward)

    produced = Counter((record["family"], record["language"]) for record in records)
    for family, builder in (
        ("S1-01", instruction_record),
        ("S1-02", relation_record),
        ("S1-04", paraphrase_record),
        ("S1-09", controlled_record),
        ("S1-10", identity_record),
    ):
        for language, count in zip(("en", "ru"), quotas.get(family, (0, 0))):
            already = produced[(family, language)]
            for index in range(max(0, count - already)):
                forced = "D" if family == "S1-10" and index % 8 in (4, 5) else None
                records.append(builder(index, language, take_difficulty(forced)))
    return records


def assign_difficulties(native_records: list[dict], public_records: list[dict], total: int = TARGET) -> dict[str, int]:
    """Fill non-contrast native rows toward the frozen difficulty mix. Returns the residual."""
    targets = difficulty_targets(total)
    used = Counter(record["difficulty"] for record in public_records)
    open_rows = []
    for record in native_records:
        task = str(record["metadata"].get("latent_task") or "")
        fact = str(record["metadata"].get("latent_fact") or "")
        if task.startswith("reversal") or fact == "distinction":
            record["difficulty"] = "D"
            used["D"] += 1
        else:
            open_rows.append(record)
    pool: list[str] = []
    for level in DIFFICULTIES:
        pool.extend([level] * max(0, targets[level] - used[level]))
    if len(pool) < len(open_rows):
        pool.extend(["C"] * (len(open_rows) - len(pool)))
    for record, level in zip(open_rows, pool):
        record["difficulty"] = level
        used[level] += 1
    return {level: used[level] - targets[level] for level in DIFFICULTIES}


def tag_public(record: dict) -> dict:
    record["metadata"]["source_type"] = "PUBLIC"
    record["metadata"]["builder_version"] = "s1-curriculum-pilot-v1"
    return record
