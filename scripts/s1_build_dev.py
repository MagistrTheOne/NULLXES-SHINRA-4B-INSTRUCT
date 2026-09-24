"""Freeze the S1 DEV holdout. No training and no dataset download."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.s1.dev_holdout import DEV_COUNTS, generate_dev
from data.s1.foundation import sha256_text
from training.s1_p0_objective import S1_P0_SEED

OUT = ROOT / "data" / "s1" / "p0_dev"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = generate_dev()
    rows.sort(key=lambda row: row["id"])
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path = OUT / "s1_dev_v1.jsonl"
    path.write_text(payload, encoding="utf-8")
    prompts = [row["messages"][0]["content"] for row in rows]
    hashes = sorted({sha256_text(prompt) for prompt in prompts})
    (OUT / "dev_prompt_hashes.json").write_text(
        json.dumps({"count": len(hashes), "prompt_sha256": hashes}, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "split": "dev",
        "builder": "s1-p0-dev-v1",
        "seed": S1_P0_SEED,
        "record_count": len(rows),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "languages": dict(Counter(row["language"] for row in rows)),
        "families": {family: Counter(row["family"] for row in rows)[family] for family in DEV_COUNTS},
        "difficulties": {level: Counter(row["difficulty"] for row in rows)[level] for level in "ABCDE"},
        "unique_prompts": len(set(prompts)),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
