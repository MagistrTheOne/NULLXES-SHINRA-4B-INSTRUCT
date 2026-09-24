"""CPU checks for the S1-P0 update rule. No weights and no GPU."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from data.s1.blacklist import QA_PROMPTS, diagnostic_blacklist
from data.s1.foundation import sha256_text
from training.s1_p0_objective import (
    S1_P0_SEED,
    can_attend,
    equal_microbatch_mean,
    gradient_divisor,
    token_weighted_mean,
)

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "s1" / "p0_dev" / "s1_dev_v1.jsonl"


def test_token_weighted_mean_is_not_the_microbatch_mean():
    groups = [[10.0], [0.0] * 100]
    assert equal_microbatch_mean(groups) == 5.0
    assert abs(token_weighted_mean(groups) - (10.0 / 101.0)) < 1e-12
    assert gradient_divisor(101) == 101.0


def test_packing_blocks_cross_example_attention():
    docs = [0, 0, 1, 1]
    assert can_attend(docs, 1, 0)
    assert can_attend(docs, 3, 2)
    assert not can_attend(docs, 2, 0)
    assert not can_attend(docs, 2, 1)
    pack = (ROOT / "data" / "pack.py").read_text(encoding="utf-8")
    assert '"attention_mask": [1] * sequence_length' in pack


def test_seed_is_frozen():
    assert S1_P0_SEED == 20260924
    assert S1_P0_SEED != 7407
    assert S1_P0_SEED != 42


def test_dev_holdout_contract():
    assert DEV.exists(), "run scripts/s1_build_dev.py first"
    rows = [json.loads(line) for line in DEV.read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads((DEV.parent / "manifest.json").read_text(encoding="utf-8"))
    assert len(rows) == 2000
    assert manifest["sha256"]
    assert manifest["seed"] == S1_P0_SEED
    families = Counter(row["family"] for row in rows)
    languages = Counter(row["language"] for row in rows)
    difficulties = Counter(row["difficulty"] for row in rows)
    assert set(families) == {f"S1-{index:02d}" for index in range(1, 11)}
    assert languages["en"] == 1000 and languages["ru"] == 1000
    assert set(difficulties) == set("ABCDE")
    assert families["S1-10"] == 100
    prompts = [row["messages"][0]["content"] for row in rows]
    assert len(prompts) == len(set(prompts))
    blocked = diagnostic_blacklist()
    for prompt in prompts:
        assert sha256_text(prompt) not in blocked
    for row in rows:
        blob = row["messages"][0]["content"] + row["messages"][1]["content"]
        assert all(message["role"] != "system" for message in row["messages"])
        if row["family"] != "S1-10":
            assert "SHINRA" not in blob.upper()
            assert "NULLXES" not in blob.upper()
    assert len(QA_PROMPTS) == 12


if __name__ == "__main__":
    test_token_weighted_mean_is_not_the_microbatch_mean()
    test_packing_blocks_cross_example_attention()
    test_seed_is_frozen()
    test_dev_holdout_contract()
    print("test_s1_p0_objective: PASS")
