"""Pinned canonical/v1 reader. No discovery, no exclusions, no shuffle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATASET_REPO = "MagistrTheOne/NULLXES-SHINRA-S1-P0"
DATASET_COMMIT = "0b99e4568fd5195facea7323249bf4c8b1c02594"
MANIFEST_PATH = "canonical/v1/canonical_manifest.json"
UPDATE_PLAN_PATH = "canonical/v1/update_plan.json"
UPDATE_TARGET = 32_768


class DatasetPinError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_pins(
    dataset_repo: str,
    dataset_commit: str,
    canonical_manifest_path: str,
    update_plan_path: str,
) -> None:
    if dataset_repo != DATASET_REPO:
        raise DatasetPinError("dataset repo mismatch")
    if dataset_commit != DATASET_COMMIT:
        raise DatasetPinError("dataset commit mismatch")
    if canonical_manifest_path != MANIFEST_PATH:
        raise DatasetPinError("canonical manifest path mismatch")
    if update_plan_path != UPDATE_PLAN_PATH:
        raise DatasetPinError("update plan path mismatch")


def load_authority(
    dataset_repo: str,
    dataset_commit: str,
    canonical_manifest_path: str,
    update_plan_path: str,
    manifest_file: Path,
    plan_file: Path,
    expected_manifest_sha256: str,
    expected_plan_sha256: str,
    expected_train_sha256: str,
) -> tuple[dict, dict]:
    """Read the two pinned JSON files and nothing else."""
    require_pins(dataset_repo, dataset_commit, canonical_manifest_path, update_plan_path)
    manifest = load_json(manifest_file)
    plan = load_json(plan_file)
    verify_authority(
        manifest,
        plan,
        manifest_sha256=sha256_file(manifest_file),
        plan_sha256=sha256_file(plan_file),
        expected_manifest_sha256=expected_manifest_sha256,
        expected_plan_sha256=expected_plan_sha256,
        expected_train_sha256=expected_train_sha256,
    )
    return manifest, plan


def verify_authority(
    manifest: dict,
    plan: dict,
    *,
    manifest_sha256: str,
    plan_sha256: str,
    expected_manifest_sha256: str,
    expected_plan_sha256: str,
    expected_train_sha256: str,
) -> None:
    if manifest.get("dataset_repo") != DATASET_REPO:
        raise DatasetPinError("dataset repo mismatch")
    if manifest_sha256 != expected_manifest_sha256:
        raise DatasetPinError("manifest hash mismatch")
    if plan_sha256 != expected_plan_sha256:
        raise DatasetPinError("update plan hash mismatch")
    if manifest.get("train_sha256") != expected_train_sha256:
        raise DatasetPinError("TRAIN SHA256 mismatch")
    if plan.get("train_sha256") != expected_train_sha256:
        raise DatasetPinError("update plan TRAIN SHA256 mismatch")
    shards = manifest.get("shards") or []
    if [row.get("canonical_index") for row in shards] != list(range(len(shards))):
        raise DatasetPinError("manifest shard order is not canonical")
    updates = plan.get("updates") or []
    cursor = 0
    for index, update in enumerate(updates):
        if update.get("update_index") != index:
            raise DatasetPinError("update index gap")
        if update.get("record_start") != cursor:
            raise DatasetPinError("update does not continue the previous record")
        end = int(update["record_end"])
        if end <= cursor:
            raise DatasetPinError("empty or backwards update")
        if int(update["record_count"]) != end - cursor:
            raise DatasetPinError("record_count does not match the boundary")
        target = int(update["target_tokens"])
        if target <= 0:
            raise DatasetPinError("update has no supervised tokens")
        if index < len(updates) - 1 and target < UPDATE_TARGET:
            raise DatasetPinError("closed update is under the token target")
        cursor = end
    if cursor != int(manifest["total_records"]):
        raise DatasetPinError("update plan does not cover every canonical record")


def verify_shard(path: Path, expected: dict) -> int:
    digest = sha256_file(path)
    if digest != expected["sha256"]:
        raise DatasetPinError(f"shard hash mismatch {path.name}")
    records = 0
    target = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records += 1
            target += int(row["target_tokens"])
    if records != int(expected["records"]) or target != int(expected["target_tokens"]):
        raise DatasetPinError(f"shard accounting mismatch {path.name}")
    if path.stat().st_size != int(expected["bytes"]):
        raise DatasetPinError(f"shard size mismatch {path.name}")
    return records


def iter_records(manifest: dict, root: Path):
    """Yield records in manifest order. root holds the shard files by manifest path name."""
    for shard in manifest["shards"]:
        path = root / shard["path"]
        verify_shard(path, shard)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def take_update(records, update: dict) -> list[dict]:
    start = int(update["record_start"])
    end = int(update["record_end"])
    chunk = list(records[start:end])
    if len(chunk) != int(update["record_count"]):
        raise DatasetPinError("record boundary mismatch")
    actual = sum(int(row["target_tokens"]) for row in chunk)
    if actual != int(update["target_tokens"]):
        raise DatasetPinError("update target-token mismatch")
    return chunk
