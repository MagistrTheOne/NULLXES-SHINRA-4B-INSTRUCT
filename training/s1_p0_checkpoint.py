"""Immutable P0 resume generations. latest.json moves only after COMPLETE."""

from __future__ import annotations

import json
import os
from pathlib import Path

REQUIRED_FILES = (
    "model.pt",
    "optimizer.pt",
    "rng_python.json",
    "rng_torch_cpu.pt",
    "rng_torch_cuda.json",
    "cursor.json",
    "identity.json",
)


class CheckpointError(RuntimeError):
    pass


def generation_dir(root: Path, index: int) -> Path:
    if index < 1:
        raise CheckpointError("generation index starts at 1")
    return root / "resume" / f"gen-{index:06d}"


def latest_path(root: Path) -> Path:
    return root / "state" / "latest.json"


def write_generation(root: Path, index: int, files: dict[str, bytes | str], identity: dict) -> Path:
    path = generation_dir(root, index)
    if path.exists():
        raise CheckpointError(f"{path.name} already exists")
    path.mkdir(parents=True)
    for name, payload in files.items():
        target = path / name
        if isinstance(payload, str):
            target.write_text(payload, encoding="utf-8")
        else:
            target.write_bytes(payload)
    (path / "identity.json").write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if missing:
        raise CheckpointError(f"incomplete generation {missing}")
    (path / "COMPLETE").write_text("ok\n", encoding="utf-8")
    return path


def publish_latest(root: Path, index: int, identity: dict) -> Path:
    path = generation_dir(root, index)
    if not (path / "COMPLETE").is_file():
        raise CheckpointError("refusing to publish a generation without COMPLETE")
    stored = json.loads((path / "identity.json").read_text(encoding="utf-8"))
    for key in (
        "dataset_commit",
        "train_sha256",
        "manifest_sha256",
        "update_plan_sha256",
        "contract_commit",
    ):
        if stored.get(key) != identity.get(key):
            raise CheckpointError(f"identity mismatch {key}")
    pointer = {
        "generation": path.name,
        "next_update": identity["next_update"],
        "dataset_commit": identity["dataset_commit"],
        "train_sha256": identity["train_sha256"],
        "manifest_sha256": identity["manifest_sha256"],
        "update_plan_sha256": identity["update_plan_sha256"],
        "contract_commit": identity["contract_commit"],
    }
    destination = latest_path(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(pointer, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def load_resume(root: Path, identity: dict) -> dict:
    pointer_path = latest_path(root)
    if not pointer_path.is_file():
        raise CheckpointError("latest.json missing")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    path = root / "resume" / pointer["generation"]
    if not (path / "COMPLETE").is_file():
        raise CheckpointError("latest generation is not COMPLETE")
    stored = json.loads((path / "identity.json").read_text(encoding="utf-8"))
    for key in (
        "dataset_commit",
        "train_sha256",
        "manifest_sha256",
        "update_plan_sha256",
        "contract_commit",
    ):
        if pointer.get(key) != identity.get(key) or stored.get(key) != identity.get(key):
            raise CheckpointError(f"resume pin mismatch {key}")
    return pointer
