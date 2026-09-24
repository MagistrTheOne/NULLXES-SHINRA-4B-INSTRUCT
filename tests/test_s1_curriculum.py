"""CPU checks for the S1 curriculum pilot. No datasets, GPU, or model weights."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "s1" / "pilot_v1" / "s1_curriculum_pilot_v1.jsonl"

from data.s1.curriculum import TARGET, family_targets, generate_native
from data.s1.native import instruction_record, paraphrase_record, relation_reversal_pair


def _quotas():
    targets = family_targets(TARGET)
    return {family: (targets[family] // 2, targets[family] - targets[family] // 2) for family in ("S1-01", "S1-02", "S1-04", "S1-09", "S1-10")}


def test_native_rebuild_is_byte_identical():
    quotas = _quotas()
    first = generate_native(quotas)
    second = generate_native(quotas)
    left = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in first)
    right = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in second)
    assert left == right
    assert len(first) == sum(sum(pair) for pair in quotas.values())


def test_relation_reversal_targets_differ():
    forward, backward = relation_reversal_pair(3, "en")
    assert forward["messages"][1]["content"] != backward["messages"][1]["content"]
    assert forward["messages"][0]["content"] != backward["messages"][0]["content"]
    ru_forward, ru_backward = relation_reversal_pair(3, "ru")
    assert ru_forward["messages"][1]["content"] != ru_backward["messages"][1]["content"]


def test_paraphrase_negatives_follow_latent_mode():
    negative = paraphrase_record(5, "en", "C")
    assert negative["metadata"]["latent_task"] == "hard_negative"
    assert negative["messages"][1]["content"] == "no"
    ru = paraphrase_record(5, "ru", "C")
    assert ru["messages"][1]["content"] == "нет"
    equivalent = paraphrase_record(0, "en", "B")
    assert equivalent["metadata"]["latent_task"] == "equivalent"
    assert equivalent["messages"][1]["content"] == "yes"


def test_instruction_contrast_changes_the_target():
    left = instruction_record(0, "en", "A")
    right = instruction_record(1, "en", "B")
    assert left["metadata"]["contrast_group"] == right["metadata"]["contrast_group"]
    assert left["metadata"]["latent_task"] != right["metadata"]["latent_task"]
    assert left["messages"][1]["content"] != right["messages"][1]["content"]
    assert "SHINRA" not in left["messages"][0]["content"].upper()


def test_native_rows_have_no_system_role():
    rows = generate_native({"S1-01": (2, 2), "S1-02": (4, 4), "S1-04": (2, 2), "S1-09": (2, 2), "S1-10": (8, 8)})
    for row in rows:
        assert [message["role"] for message in row["messages"]] == ["user", "assistant"]
        assert row["messages"][1]["content"].strip()
        if row["family"] != "S1-10":
            blob = row["messages"][0]["content"] + row["messages"][1]["content"]
            assert "SHINRA" not in blob.upper()
            assert "NULLXES" not in blob.upper()


def test_curriculum_file_contracts():
    assert PILOT.exists(), "run scripts/s1_build_curriculum.py first"
    rows = [json.loads(line) for line in PILOT.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == TARGET
    families = Counter(row["family"] for row in rows)
    languages = Counter(row["language"] for row in rows)
    difficulties = Counter(row["difficulty"] for row in rows)
    assert set(families) == {f"S1-{index:02d}" for index in range(1, 11)}
    for family, target in family_targets().items():
        assert abs(families[family] - target) <= TARGET * 0.02
    for language in ("en", "ru"):
        assert abs(languages[language] - TARGET // 2) <= TARGET * 0.03
    for level, percent in (("A", 15), ("B", 20), ("C", 30), ("D", 25), ("E", 10)):
        assert abs(difficulties[level] - TARGET * percent // 100) <= TARGET * 0.03
    assert families["S1-10"] <= TARGET * 0.06
    for row in rows:
        blob = row["messages"][0]["content"] + "\n" + row["messages"][1]["content"]
        assert row["source_split"] == "train"
        assert all(message["role"] != "system" for message in row["messages"])
        if row["family"] != "S1-10":
            assert "SHINRA" not in blob.upper()
            assert "NULLXES" not in blob.upper()
        source = row["source"].lower()
        for name in ("xquad", "belebele", "rubq", "mmlu", "flores", "paws", "xnli"):
            assert name not in source
    prompts = [row["messages"][0]["content"] for row in rows]
    assert len(prompts) == len(set(prompts))


if __name__ == "__main__":
    test_native_rebuild_is_byte_identical()
    test_relation_reversal_targets_differ()
    test_paraphrase_negatives_follow_latent_mode()
    test_instruction_contrast_changes_the_target()
    test_native_rows_have_no_system_role()
    test_curriculum_file_contracts()
    print("test_s1_curriculum: PASS")
