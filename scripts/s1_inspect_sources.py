"""CPU metadata inspection. Does not build the training corpus."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download

from data.s1.foundation import ROOT as REPO
from data.s1.foundation import load_registry

OUT = REPO / "data" / "s1" / "source_inspection.json"


def _zip_train(config: str, repo: str) -> dict:
    path = hf_hub_download(repo, f"data/{config}.zip", repo_type="dataset")
    archive = zipfile.ZipFile(path)
    name = f"{config}/train.jsonl"
    names = [item for item in archive.namelist() if item.endswith(".jsonl")]
    sample = json.loads(archive.open(name).readline().decode("utf-8"))
    count = sum(1 for line in archive.open(name) if line.strip())
    return {
        "config": config,
        "splits_present": sorted({Path(item).name.split(".")[0] for item in names}),
        "train_rows": count,
        "schema": sorted(sample.keys()),
        "opened_split": "train",
    }


def main() -> None:
    registry = load_registry()
    report = {"sources": []}
    tydi = [row for row in registry["sources"] if row["hf_repo"].endswith("tydiqa")]
    from datasets import load_dataset

    for source in tydi:
        features = load_dataset(source["hf_repo"], source["config"], split="train", streaming=True)
        row = next(iter(features))
        report["sources"].append(
            {
                "source_id": source["source_id"],
                "config": source["config"],
                "allowed_splits": source["allowed_splits"],
                "schema": sorted(row.keys()),
                "sample_language_field": row.get("language") or str(row.get("id", ""))[:24],
            }
        )
    rsg_repo = "RussianNLP/russian_super_glue"
    for config in ("DaNetQA", "TERRa", "RCB", "RuCoS", "RWSD", "MuSeRC"):
        item = _zip_train(config, rsg_repo)
        item["source_id"] = "rsg_" + config.lower()
        report["sources"].append(item)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
