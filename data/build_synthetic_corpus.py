"""Generate, verify, tokenize, and pack an audit-only NULLXES synthetic corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from configs.loader import CONFIG_PATH, load_config
from data.hashing import sha256_file
from data.pack import pack_documents
from data.synthetic_engine import generate_corpus


class SyntheticBuildError(RuntimeError):
    """The synthetic corpus build could not be safely published."""


def _manifest_sha(value: dict[str, Any]) -> str:
    body = dict(value)
    body.pop("manifest_sha256", None)
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _report_sha(value: dict[str, Any]) -> str:
    body = dict(value)
    body.pop("report_sha256", None)
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _verify_packed_split(path: Path, split: str, corpus_sha: str) -> dict[str, Any]:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyntheticBuildError(f"packed {split} manifest is missing or invalid: {exc}") from exc
    if manifest.get("schema") != "nullxes.shinra.packed.synthetic.v1" \
            or manifest.get("manifest_sha256") != _manifest_sha(manifest) \
            or manifest.get("split") != split \
            or manifest.get("corpus_manifest_sha256") != corpus_sha:
        raise SyntheticBuildError(f"packed {split} manifest identity or self-hash is invalid")
    files, hashes = manifest.get("shard_files"), manifest.get("shard_sha256")
    if not isinstance(files, list) or not files or not isinstance(hashes, dict) or set(files) != set(hashes):
        raise SyntheticBuildError(f"packed {split} shard inventory is incomplete")
    for name in files:
        shard = path / name
        if not shard.is_file() or sha256_file(shard) != hashes[name]:
            raise SyntheticBuildError(f"packed {split} shard checksum mismatch: {name}")
    return manifest


def build_synthetic_corpus(tokenizer_path: str | Path, output_dir: str | Path, *,
                           config_path: str | Path = CONFIG_PATH,
                           records_per_family: int | None = None,
                           root_seed: int | None = None,
                           minimum_free_gb: float | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    output = Path(output_dir).resolve()
    stage = output.parent / f".{output.name}.synthetic-build-partial"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed synthetic corpus: {output}")
    if stage.exists():
        raise FileExistsError(f"synthetic partial build already exists and is preserved: {stage}")
    output.parent.mkdir(parents=True, exist_ok=True)
    reserve_gb = float(config["runtime"]["minimum_free_disk_gb"]
                       if minimum_free_gb is None else minimum_free_gb)
    if reserve_gb <= 0:
        raise SyntheticBuildError("minimum_free_gb must be positive")
    available_gb = shutil.disk_usage(output.parent).free / (1024 ** 3)
    if available_gb < reserve_gb:
        raise SyntheticBuildError(
            f"insufficient free space before synthetic build: available={available_gb:.2f} GiB, "
            f"required={reserve_gb:.2f} GiB"
        )
    stage.mkdir()
    corpus_dir = stage / "corpus"
    synthetic_manifest = generate_corpus(
        corpus_dir, config_path=config_path, records_per_family=records_per_family, root_seed=root_seed
    )
    corpus_sha = synthetic_manifest["manifest_sha256"]
    packed_manifests: dict[str, str] = {}
    reports: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "blind"):
        reports[split] = pack_documents(
            corpus_dir, Path(tokenizer_path), stage / split,
            int(config["model"]["train_sequence_length"]),
            int(config["data"]["packed_shard_tokens"]),
            config_path=Path(config_path), required_split=split, config=config,
            minimum_free_gb=reserve_gb,
        )
        packed = _verify_packed_split(stage / split, split, corpus_sha)
        packed_manifests[split] = packed["manifest_sha256"]
    build_report: dict[str, Any] = {
        "schema": "nullxes.shinra.synthetic-build-report.v1",
        "admission": "audit_only",
        "training_enabled": False,
        "corpus_manifest_sha256": corpus_sha,
        "packed_manifest_sha256": packed_manifests,
        "record_counts": synthetic_manifest["record_counts"],
        "record_counts_by_language": synthetic_manifest["record_counts_by_language"],
        "packed_reports": reports,
    }
    build_report["report_sha256"] = _report_sha(build_report)
    if build_report["report_sha256"] != _report_sha(build_report):
        raise SyntheticBuildError("synthetic build report self-hash could not be reproduced")
    (stage / "build_report.json").write_text(
        json.dumps(build_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The trainer reads this exact corpus manifest from the packed-data root.
    (stage / "manifest.json").write_bytes((corpus_dir / "manifest.json").read_bytes())
    os.replace(stage, output)
    return build_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and pack an audit-only NULLXES Synthetic Engine corpus"
    )
    parser.add_argument("--tokenizer", required=True, help="verified Native SentencePiece bundle")
    parser.add_argument("--output-dir", required=True, help="new, empty destination on the approved runtime")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--records-per-family", type=int, default=None)
    parser.add_argument("--root-seed", type=int, default=None)
    parser.add_argument("--minimum-free-gb", type=float, default=None)
    args = parser.parse_args()
    result = build_synthetic_corpus(
        args.tokenizer, args.output_dir, config_path=args.config,
        records_per_family=args.records_per_family, root_seed=args.root_seed,
        minimum_free_gb=args.minimum_free_gb,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
