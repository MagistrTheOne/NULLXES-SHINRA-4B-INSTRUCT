"""Gated FineWeb-Edu EN frozen-file fetch. One shard, one revision. No canary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from data.data_v1.acquisition import (
    MIN_FREE_BYTES,
    SOURCE_MAX_MATERIALIZED_BYTES,
    AcquisitionError,
    assert_allowed_acquisition_root,
    validate_acquired_artifact,
)
from data.data_v1.plan_fineweb_edu_en import (
    BOUND_BYTES,
    EXPECTED_RAW_SHA256,
    FILENAME,
    RAW_FORMAT,
    REPOSITORY,
    REVISION,
    SOURCE_ID,
    STRATEGY,
    UPSTREAM_PATH,
    frozen_plan,
)
from data.data_v1.sources import load_allowlist

FetchFn = Callable[[str, str, str, Path], None]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _default_hf_fetch(repo: str, revision: str, repo_path: str, dest: Path) -> None:
    if (repo, revision, repo_path) != (REPOSITORY, REVISION, UPSTREAM_PATH):
        raise AcquisitionError("gated fetch refuses unplanned repo/revision/path")
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=repo,
        filename=repo_path,
        repo_type="dataset",
        revision=revision,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    inner = dest.with_name(dest.name + ".download")
    shutil.copyfile(cached, inner)
    os.replace(inner, dest)


def acquire_fineweb_edu_en(
    *,
    acquisition_root: str | Path,
    allowlist: dict[str, Any] | None = None,
    fetch_file: FetchFn | None = None,
    free_bytes: int | None = None,
    extra_s0: tuple[Path, ...] = (),
    extra_materialize: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Download the single frozen parquet shard, verify SHA, write sibling manifest.

    Does not adapt, materialize, or run canary.
    """
    plan = frozen_plan()
    if plan["bound_bytes"] > SOURCE_MAX_MATERIALIZED_BYTES:
        raise AcquisitionError("plan bound exceeds 4 GiB")
    root = Path(acquisition_root)
    assert_allowed_acquisition_root(root, extra_s0=extra_s0, extra_materialize=extra_materialize)
    doc = allowlist if allowlist is not None else load_allowlist()
    row = next(s for s in doc["sources"] if s["source_id"] == SOURCE_ID)
    if row["upstream"]["repository"] != REPOSITORY or row["upstream"]["revision"] != REVISION:
        raise AcquisitionError("governance freeze mismatch")

    folder = root / SOURCE_ID
    dest = folder / FILENAME
    manifest_path = folder / f"{FILENAME}.acquisition.json"
    fetch = fetch_file or _default_hf_fetch

    if dest.is_file() and manifest_path.is_file() and _sha256_file(dest) == EXPECTED_RAW_SHA256:
        return validate_acquired_artifact(
            SOURCE_ID,
            manifest_path,
            root,
            doc,
            extra_s0=extra_s0,
            extra_materialize=extra_materialize,
            free_bytes=free_bytes,
        )

    measure = root if root.exists() else (root.parent if root.parent.exists() else Path.cwd())
    available = int(free_bytes) if free_bytes is not None else shutil.disk_usage(measure).free
    if available < MIN_FREE_BYTES:
        raise AcquisitionError("free disk below 20 GiB")

    folder.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        tmp.unlink()
    fetch(REPOSITORY, REVISION, UPSTREAM_PATH, tmp)
    if not tmp.is_file() or tmp.stat().st_size < 1:
        raise AcquisitionError("empty raw artifact")
    if tmp.stat().st_size > BOUND_BYTES:
        tmp.unlink(missing_ok=True)
        raise AcquisitionError("input exceeds 4 GiB")
    actual = _sha256_file(tmp)
    if actual != EXPECTED_RAW_SHA256:
        tmp.unlink(missing_ok=True)
        raise AcquisitionError("raw SHA mismatch")
    if dest.exists():
        dest.unlink()
    os.replace(tmp, dest)

    manifest = {
        "schema_version": "shinra-data-v1-acquisition-manifest",
        "source_id": SOURCE_ID,
        "upstream": {
            "repository": REPOSITORY,
            "subset": None,
            "revision": REVISION,
        },
        "artifact": {
            "filename": FILENAME,
            "format": RAW_FORMAT,
            "bytes": dest.stat().st_size,
            "raw_content_sha256": actual,
        },
        "selection": {
            "strategy": STRATEGY,
            "bound_bytes": BOUND_BYTES,
            "path": UPSTREAM_PATH,
            "expected_raw_sha256": EXPECTED_RAW_SHA256,
        },
        "status": "complete",
    }
    man_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    man_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(man_tmp, manifest_path)
    return validate_acquired_artifact(
        SOURCE_ID,
        manifest_path,
        root,
        doc,
        extra_s0=extra_s0,
        extra_materialize=extra_materialize,
        free_bytes=free_bytes,
    )
