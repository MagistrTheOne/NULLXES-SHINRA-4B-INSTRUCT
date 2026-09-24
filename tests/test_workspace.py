"""Filesystem checks for the bucket workspace. No GPU, no Hub, no datasets."""

from __future__ import annotations

import json

import pytest

from runtime.workspace import (
    BUCKET_URI,
    DATASET_REPO,
    EPHEMERAL_FALLBACK,
    MODEL_REPO,
    MOUNT_SPEC,
    STANDARD_MOUNT,
    WorkspaceError,
    begin_checkpoint,
    commit_checkpoint,
    ensure_layout,
    inspect_workspace,
    latest_path,
    resolve_workspace,
    validate_resume,
    write_latest,
)


def _payload(index: int = 3) -> dict:
    return {
        "checkpoint": f"checkpoint-{index}",
        "update": index,
        "target_tokens_seen": 32768 * index,
        "dataset_commit": "abc123",
        "train_sha256": "deadbeef",
        "contract_commit": "000813d2294b6ec8f2287d4eb46f64c105d0617c",
        "created_at": "2026-09-25T00:00:00Z",
    }


def test_constants_keep_the_three_homes_separate():
    assert DATASET_REPO == "MagistrTheOne/NULLXES-SHINRA-S1-P0"
    assert MODEL_REPO == "MagistrTheOne/NULLXES-SHINRA-4B-INSTRUCT"
    assert BUCKET_URI == "hf://buckets/MagistrTheOne/nullxes-shinra-workspace"
    assert MOUNT_SPEC == f"{BUCKET_URI}:{STANDARD_MOUNT}"


def test_fallback_is_ephemeral_when_bucket_is_absent():
    path, mode = resolve_workspace({}, bucket_mounted=False)
    status = inspect_workspace(path, mode=mode)
    assert path.as_posix() == EPHEMERAL_FALLBACK
    assert mode == "EPHEMERAL"
    assert status.mode == "EPHEMERAL"


def test_persistent_requirement_fails_closed_without_mount():
    with pytest.raises(WorkspaceError):
        resolve_workspace({"SHINRA_WORKSPACE_PERSISTENT": "1"}, bucket_mounted=False)


def test_persistent_requirement_uses_the_bucket_mount():
    path, mode = resolve_workspace({"SHINRA_WORKSPACE_PERSISTENT": "1"}, bucket_mounted=True)
    assert path.as_posix() == STANDARD_MOUNT
    assert mode == "BUCKET"


def test_explicit_scratch_path_stays_ephemeral(tmp_path):
    path, mode = resolve_workspace({"SHINRA_WORKSPACE": str(tmp_path)}, bucket_mounted=False)
    assert path == tmp_path
    assert mode == "EPHEMERAL"


def test_layout_and_checkpoint_paths(tmp_path):
    ensure_layout(tmp_path)
    assert (tmp_path / "s1-p0" / "build-state").is_dir()
    assert (tmp_path / "training" / "p0" / "checkpoints").is_dir()
    assert (tmp_path / "eval" / "p0").is_dir()
    temporary = begin_checkpoint(tmp_path, 4)
    assert temporary.name == "checkpoint-4.tmp"
    status = inspect_workspace(tmp_path, mode="EPHEMERAL")
    assert status.exists and status.writable
    assert status.free_bytes is not None and status.free_bytes > 0


def test_latest_roundtrip_and_resume_validation(tmp_path):
    path = write_latest(tmp_path, _payload())
    assert path == latest_path(tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_resume(loaded)
    with pytest.raises(WorkspaceError):
        validate_resume({"update": 1})
    with pytest.raises(WorkspaceError):
        validate_resume({**_payload(), "update": -1})


def test_commit_is_atomic_and_rejects_invalid_metadata(tmp_path):
    begin_checkpoint(tmp_path, 2)
    temporary, final = __import__("runtime.workspace", fromlist=["checkpoint_paths"]).checkpoint_paths(tmp_path, 2)
    (temporary / "weights.ok").write_text("ready", encoding="utf-8")
    with pytest.raises(WorkspaceError):
        commit_checkpoint(tmp_path, 2, {**_payload(2), "checkpoint": "checkpoint-9"})
    assert temporary.is_dir()
    assert not final.exists()
    assert not latest_path(tmp_path).exists()
    published = commit_checkpoint(tmp_path, 2, _payload(2))
    assert published == final
    assert not temporary.exists()
    assert (final / "weights.ok").read_text(encoding="utf-8") == "ready"
    assert json.loads(latest_path(tmp_path).read_text(encoding="utf-8"))["update"] == 2
