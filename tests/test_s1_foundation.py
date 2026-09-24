"""CPU tests for the S1 public-core foundation. No datasets and no weights."""

from __future__ import annotations

import json
from pathlib import Path

from data.s1.blacklist import QA_PROMPTS, diagnostic_blacklist
from data.s1 import foundation
from data.s1.foundation import (
    EN_INSTRUCTIONS,
    ExactDedup,
    RU_INSTRUCTIONS,
    assert_train_source,
    instruction_for,
    make_record,
    sha256_text,
)
from data.s1.transforms import qa_extractive

ROOT = Path(__file__).resolve().parents[1]


def test_registry_guards():
    assert_train_source("tydiqa_secondary", "train")
    for source, split in (
        ("tydiqa_secondary", "validation"),
        ("rsg_danetqa", "test"),
        ("rsg_terra", "validation"),
    ):
        try:
            assert_train_source(source, split)
        except PermissionError:
            pass
        else:
            raise AssertionError(f"accepted {source} {split}")
    registry = json.loads((ROOT / "data" / "s1" / "source_registry.json").read_text(encoding="utf-8"))
    for source in registry["sources"]:
        assert source["license_status"] == "APPROVED"
        assert "test" not in source["allowed_splits"]
        assert source["license"] in (
            "apache-2.0",
            "mit",
            "nullxes-internal",
            "google-paws-dataset+cc-by-sa-3.0",
            "cc-by-sa-4.0",
            "cc-by-2.0",
        )


def test_eval_only_status_is_rejected_before_approval():
    original = foundation.source_by_id
    foundation.source_by_id = lambda source_id: {
        "source_id": source_id,
        "license_status": "EVAL_ONLY",
        "license": "other",
        "allowed_splits": ["train"],
        "forbidden_splits": ["validation", "test"],
    }
    try:
        assert_train_source("custom_eval_set", "train")
    except PermissionError as exc:
        assert "eval-only" in str(exc)
    else:
        raise AssertionError("EVAL_ONLY source was accepted")
    finally:
        foundation.source_by_id = original


def test_eval_only_and_unknown_are_absent_from_registry():
    text = (ROOT / "data" / "s1" / "source_registry.json").read_text(encoding="utf-8")
    for name in ("xquad", "belebele", "rubq", "mmlu", "flores", "paws_qqp", "paws-x", "xnli"):
        assert name not in text
    assert "paws_wiki" in text


def test_transform_is_deterministic_and_user_only():
    first = qa_extractive("tydiqa_secondary", "secondary_task", "row-1", "en", "A dock.", "Where?", "Greyhaven", "S1-08", "A", "apache-2.0")
    second = qa_extractive("tydiqa_secondary", "secondary_task", "row-1", "en", "A dock.", "Where?", "Greyhaven", "S1-08", "A", "apache-2.0")
    assert first == second
    assert first["id"].startswith("s1v1-")
    assert [item["role"] for item in first["messages"]] == ["user", "assistant"]
    assert first["messages"][1]["content"] == "Greyhaven"
    assert "<|system|>" not in first["messages"][0]["content"]
    assert instruction_for(first["id"], "en") in EN_INSTRUCTIONS
    assert instruction_for(first["id"], "ru") in RU_INSTRUCTIONS
    assert len(EN_INSTRUCTIONS) == 8 and len(RU_INSTRUCTIONS) == 8
    ru = make_record(
        source_id="rsg_danetqa",
        source_config="DaNetQA",
        source_split="train",
        source_row_id="7",
        language="ru",
        family="S1-03",
        difficulty="D",
        context="Текст.",
        question="Верно?",
        answer="Нет",
        license_name="mit",
        transform="yes_no.v1",
    )
    assert ru["language"] == "ru" and ru["family"] == "S1-03"


def test_template_selection_depends_on_id_only():
    assert instruction_for("s1v1-abc", "en") == instruction_for("s1v1-abc", "en")
    found = False
    for index in range(32):
        if instruction_for(f"s1v1-{index}", "en") != instruction_for("s1v1-0", "en"):
            found = True
            break
    assert found


def test_dedup_and_diagnostic_blacklist():
    record = qa_extractive("tydiqa_secondary", "secondary_task", "row-9", "en", "Port.", "Who?", "Mira", "S1-08", "A", "apache-2.0")
    dedup = ExactDedup()
    assert dedup.accept(record) is True
    assert dedup.accept(record) is False
    assert dedup.removed == 1
    blocked = ExactDedup({sha256_text(record["messages"][0]["content"])})
    assert blocked.accept(record) is False
    source = (ROOT / "data" / "s1" / "blacklist.py").read_text(encoding="utf-8")
    assert "maxon" not in source
    assert "C:\\Users" not in source
    blacklist = diagnostic_blacklist()
    assert len(QA_PROMPTS) == 12
    assert sha256_text(QA_PROMPTS[0]) in blacklist
    assert len(blacklist) >= 12 + 160


def test_spec_hash():
    import hashlib

    data = (ROOT / "data" / "specs" / "S1_DATA_SPEC.md").read_bytes()
    assert len(data) == 9750
    assert hashlib.sha256(data).hexdigest() == "2c5b4e2702b19a89849d6be4c47db65057f92e6ae2623031495c8d21486deb68"


if __name__ == "__main__":
    test_registry_guards()
    test_eval_only_status_is_rejected_before_approval()
    test_eval_only_and_unknown_are_absent_from_registry()
    test_transform_is_deterministic_and_user_only()
    test_template_selection_depends_on_id_only()
    test_dedup_and_diagnostic_blacklist()
    test_spec_hash()
    print("test_s1_foundation: PASS")
