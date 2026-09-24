"""Build the 5k S1 public-core pilot. CPU only. No SHINRA weights."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.s1.blacklist import diagnostic_blacklist
from data.s1.foundation import BUILDER_VERSION, FROZEN_FAMILY_PERCENT, ExactDedup
from data.s1.loaders import (
    iter_danetqa,
    iter_entailment,
    iter_muserc,
    iter_rucos,
    iter_rwsd,
    iter_tydi_primary,
    iter_tydi_secondary,
)

OUT = ROOT / "data" / "s1" / "pilot"
TARGET = 5000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dedup = ExactDedup(diagnostic_blacklist())
    kept = []
    for factory in (
        lambda: iter_tydi_secondary(1800, 800, scan_limit=250000),
        lambda: iter_tydi_primary(400, 200),
        lambda: iter_danetqa(450),
        lambda: iter_entailment("rsg_terra", "TERRa", 400),
        lambda: iter_entailment("rsg_rcb", "RCB", 400),
        lambda: iter_rucos(350),
        lambda: iter_rwsd(250),
        lambda: iter_muserc(300),
    ):
        for record in factory():
            if dedup.accept(record):
                kept.append(record)
            if len(kept) >= TARGET:
                break
        if len(kept) >= TARGET:
            break
    kept = kept[:TARGET]
    jsonl = OUT / "s1_public_core_v1.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    raw = jsonl.read_bytes()
    families = Counter(row["family"] for row in kept)
    languages = Counter(row["language"] for row in kept)
    sources = Counter(row["source"] for row in kept)
    manifest = {
        "builder_version": BUILDER_VERSION,
        "record_count": len(kept),
        "family_counts": dict(sorted(families.items())),
        "language_counts": dict(sorted(languages.items())),
        "source_counts": dict(sorted(sources.items())),
        "duplicate_counts": dedup.removed,
        "diagnostic_blocked": dedup.blocked,
        "excluded_split_counts": 0,
        "license_status": "APPROVED",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "frozen_family_percent": FROZEN_FAMILY_PERCENT,
        "gaps": [name for name, _pct in FROZEN_FAMILY_PERCENT.items() if families.get(name, 0) == 0],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
