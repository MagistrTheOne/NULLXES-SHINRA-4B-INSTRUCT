"""P0 Amendment 1. CPU only. No model weights."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download
from jinja2 import Environment
from tokenizers import Tokenizer

from data.s1.foundation import sha256_text
from data.s1.native import identity_record
from data.s1.p0_amendment import (
    IDENTITY_HARD_MAX,
    NON_IDENTITY_WEIGHTS,
    P0_BUDGET,
    WEIGHT_SUM,
    effective_family_targets,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "data" / "specs" / "S1_DATA_SPEC.md"
CONTRACT = ROOT / "data" / "specs" / "S1_P0_TRAIN_CONTRACT.md"
DEV = ROOT / "data" / "s1" / "p0_dev" / "s1_dev_v1.jsonl"
REVISION = "efae04115951d9473ebc2a90a2b2c684115408e7"
SPEC_SHA = "2c5b4e2702b19a89849d6be4c47db65057f92e6ae2623031495c8d21486deb68"


def _tokenizer():
    tokenizer_path = hf_hub_download(
        "MagistrTheOne/NULLXES-SHINRA-4B-INSTRUCT",
        "tokenizer.json",
        revision=REVISION,
        local_files_only=True,
    )
    template_path = ROOT / "tokenizer" / "chat_template.jinja"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    template = Environment(autoescape=False).from_string(template_path.read_text(encoding="utf-8"))
    return tokenizer, template


def _target_tokens(tokenizer, template, user: str, answer: str) -> int:
    prompt = template.render(
        messages=[{"role": "user", "content": user}],
        add_generation_prompt=True,
        bos_token="<|bos|>",
    )
    full = template.render(
        messages=[{"role": "user", "content": user}, {"role": "assistant", "content": answer}],
        add_generation_prompt=False,
        bos_token="<|bos|>",
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    full_ids = tokenizer.encode(full, add_special_tokens=False).ids
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise AssertionError("prompt prefix mismatch")
    if full_ids.count(1) != 1 or 2 not in full_ids[len(prompt_ids) :]:
        raise AssertionError("BOS/EOT contract failed")
    return len(full_ids) - len(prompt_ids)


def _bank():
    tokenizer, template = _tokenizer()
    prompts: set[str] = set()
    tokens = 0
    for language in ("en", "ru"):
        for index in range(1152):
            record = identity_record(index, language, "D")
            user = record["messages"][0]["content"]
            digest = sha256_text(user)
            if digest in prompts:
                raise AssertionError(f"duplicate identity prompt {language} {index}")
            prompts.add(digest)
            tokens += _target_tokens(tokenizer, template, user, record["messages"][1]["content"])
    return prompts, tokens


def test_identity_argument_is_required():
    try:
        effective_family_targets()
    except TypeError:
        return
    raise AssertionError("missing identity_tokens fell through")


def test_unique_bank_and_budgets():
    prompts, tokens = _bank()
    assert len(prompts) == 2304
    assert tokens == 12816
    assert tokens <= int(P0_BUDGET * IDENTITY_HARD_MAX)
    targets = effective_family_targets(tokens)
    assert sum(targets.values()) == P0_BUDGET
    assert targets["S1-10"] == tokens
    assert sum(NON_IDENTITY_WEIGHTS.values()) == WEIGHT_SUM
    remaining = P0_BUDGET - tokens
    for family, weight in NON_IDENTITY_WEIGHTS.items():
        quotient, remainder = divmod(remaining * weight, WEIGHT_SUM)
        assert targets[family] in (quotient, quotient + 1)
    text = CONTRACT.read_text(encoding="utf-8")
    assert "P0 Amendment 1" in text
    assert "2026-09-24" in text
    for family, budget in targets.items():
        assert f"{family}: {budget:,}" in text or f"{family}: {budget}" in text
    spec = SPEC.read_bytes()
    assert len(spec) == 9750
    assert hashlib.sha256(spec).hexdigest() == SPEC_SHA
    assert subprocess.check_output(["git", "diff", "HEAD", "--", "data/specs/S1_DATA_SPEC.md"], cwd=ROOT) == b""
    assert subprocess.check_output(["git", "diff", "HEAD", "--", str(DEV.relative_to(ROOT))], cwd=ROOT) == b""
    assert DEV.is_file() and DEV.stat().st_size > 0


if __name__ == "__main__":
    test_identity_argument_is_required()
    test_unique_bank_and_budgets()
    print("test_s1_p0_identity_amendment: PASS")
