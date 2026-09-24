"""Persistent workspace paths. No model, no datasets, no network."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DATASET_REPO = "MagistrTheOne/NULLXES-SHINRA-S1-P0"
BUCKET_URI = "hf://buckets/MagistrTheOne/nullxes-shinra-workspace"
MODEL_REPO = "MagistrTheOne/NULLXES-SHINRA-4B-INSTRUCT"
STANDARD_MOUNT = "/workspace"
EPHEMERAL_FALLBACK = "/tmp/shinra-workspace"
MOUNT_SPEC = f"{BUCKET_URI}:{STANDARD_MOUNT}"

LAYOUT = (
    "s1-p0/build-state",
    "s1-p0/scratch",
    "s1-p0/manifests",
    "training/p0/checkpoints",
    "training/p0/optimizer",
    "training/p0/scheduler",
    "training/p0/rng",
    "training/p0/logs",
    "training/p0/state",
    "eval/p0",
    "tmp-persistent",
)

LATEST_FIELDS = (
    "checkpoint",
    "update",
    "target_tokens_seen",
    "dataset_commit",
    "train_sha256",
    "contract_commit",
    "created_at",
)


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkspaceStatus:
    path: str
    exists: bool
    writable: bool
    free_bytes: int | None
    mode: str


def _env_flag(environ: Mapping[str, str], name: str) -> bool:
    return environ.get(name) == "1"


def _writable(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    probe = path / ".shinra-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _free_bytes(path: Path) -> int | None:
    target = path if path.exists() else path.parent
    if not target.exists():
        return None
    try:
        return shutil.disk_usage(target).free
    except OSError:
        return None


def mount_ready(path: Path = Path(STANDARD_MOUNT)) -> bool:
    return path.is_dir() and _writable(path)


def resolve_workspace(
    environ: Mapping[str, str] | None = None,
    *,
    bucket_mounted: bool | None = None,
) -> tuple[Path, str]:
    """Return the workspace path and BUCKET or EPHEMERAL.

    Persistent mode refuses to run without a writable /workspace mount.
    """
    env = os.environ if environ is None else environ
    mounted = mount_ready() if bucket_mounted is None else bucket_mounted
    if _env_flag(env, "SHINRA_WORKSPACE_PERSISTENT"):
        if not mounted:
            raise WorkspaceError(
                "SHINRA_WORKSPACE_PERSISTENT=1 but the bucket mount /workspace is unavailable"
            )
        return Path(STANDARD_MOUNT), "BUCKET"
    chosen = env.get("SHINRA_WORKSPACE")
    if chosen:
        path = Path(chosen)
        if path.as_posix() == STANDARD_MOUNT and mounted:
            return path, "BUCKET"
        return path, "EPHEMERAL"
    if mounted:
        return Path(STANDARD_MOUNT), "BUCKET"
    return Path(EPHEMERAL_FALLBACK), "EPHEMERAL"


def inspect_workspace(
    path: Path | None = None,
    *,
    mode: str | None = None,
    environ: Mapping[str, str] | None = None,
    bucket_mounted: bool | None = None,
) -> WorkspaceStatus:
    if path is None:
        path, mode = resolve_workspace(environ, bucket_mounted=bucket_mounted)
    elif mode is None:
        mode = "BUCKET" if path.as_posix() == STANDARD_MOUNT and _writable(path) else "EPHEMERAL"
    return WorkspaceStatus(
        path=str(path),
        exists=path.exists(),
        writable=_writable(path) if path.exists() else False,
        free_bytes=_free_bytes(path),
        mode=mode,
    )


def ensure_layout(workspace: Path) -> None:
    for relative in LAYOUT:
        (workspace / relative).mkdir(parents=True, exist_ok=True)


def checkpoint_paths(workspace: Path, index: int) -> tuple[Path, Path]:
    if index < 0:
        raise WorkspaceError("checkpoint index must be >= 0")
    base = workspace / "training" / "p0" / "checkpoints"
    return base / f"checkpoint-{index}.tmp", base / f"checkpoint-{index}"


def latest_path(workspace: Path) -> Path:
    return workspace / "training" / "p0" / "state" / "latest.json"


def validate_resume(payload: Mapping[str, object]) -> None:
    missing = [field for field in LATEST_FIELDS if field not in payload]
    if missing:
        raise WorkspaceError(f"resume metadata missing {', '.join(missing)}")
    for field in ("checkpoint", "dataset_commit", "train_sha256", "contract_commit", "created_at"):
        if not str(payload[field]).strip():
            raise WorkspaceError(f"resume field {field} is empty")
    for field in ("update", "target_tokens_seen"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise WorkspaceError(f"resume field {field} must be a non-negative integer")


def write_latest(workspace: Path, payload: Mapping[str, object]) -> Path:
    validate_resume(payload)
    path = latest_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def begin_checkpoint(workspace: Path, index: int) -> Path:
    temporary, final = checkpoint_paths(workspace, index)
    if final.exists():
        raise WorkspaceError(f"checkpoint {index} already exists")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    return temporary


def commit_checkpoint(workspace: Path, index: int, payload: Mapping[str, object]) -> Path:
    """Publish a finished checkpoint directory, then write latest.json."""
    validate_resume(payload)
    temporary, final = checkpoint_paths(workspace, index)
    if not temporary.is_dir() or not any(temporary.iterdir()):
        raise WorkspaceError("checkpoint temp directory is empty")
    if str(payload["checkpoint"]) != final.name:
        raise WorkspaceError("latest checkpoint name does not match the directory")
    for file in temporary.rglob("*"):
        if file.is_file():
            with file.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
    os.replace(temporary, final)
    write_latest(workspace, payload)
    return final
