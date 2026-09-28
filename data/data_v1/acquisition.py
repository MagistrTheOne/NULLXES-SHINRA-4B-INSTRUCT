"""DATA V1 acquisition contract: local raw artifact + sibling manifest.

Does not download. Does not enumerate remotes. Does not adapt formats.
Does not call the materializer, Phase B, canary, tokenizer, or trainer.
Does not mutate git allowlist.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from data.data_v1 import PhaseBError
from data.data_v1.sources import (
    ALLOWLIST_PATH,
    GLOBAL_MAX_MATERIALIZED_BYTES,
    SHA256_RE,
    SOURCE_MAX_MATERIALIZED_BYTES,
    load_allowlist,
    source_index,
)

MANIFEST_SCHEMA_VERSION = "shinra-data-v1-acquisition-manifest"
PRODUCTION_ACQUISITION_ROOT = Path("/content/shinra_scratch/data_v1_acquisition")
PRODUCTION_MATERIALIZE_ROOT = Path("/content/shinra_scratch/data_v1")
S0_SCRATCH = Path("/content/shinra_scratch/s0")
MIN_FREE_BYTES = 20 * (1024**3)
RAW_FORMATS = ("parquet", "arrow", "jsonl")
SELECTION_STRATEGY = "bounded_bytes"
INCOMPLETE_SUFFIXES = (".tmp", ".part", ".partial")
REMOTE_RE = re.compile(r"^(https?|hf|s3|gs|ftp)://", re.IGNORECASE)
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_REVISION_LABELS = frozenset(
    {"head", "main", "master", "latest", "current", "pending", "unknown"}
)
MANIFEST_KEYS = ("schema_version", "source_id", "upstream", "artifact", "selection", "status")
ARTIFACT_KEYS = ("filename", "format", "bytes", "raw_content_sha256")
SELECTION_KEYS = ("strategy", "bound_bytes")


class AcquisitionError(PhaseBError):
    """Acquisition contract failure. Fail closed. Not a downloader error."""


def _posix(path: Path) -> str:
    return path.as_posix().replace("\\", "/").casefold()


def _descends_from(path: Path, root: Path) -> bool:
    try:
        text = _posix(path.resolve())
    except OSError:
        text = _posix(path)
    try:
        root_text = _posix(root.resolve()) if root.exists() else _posix(root)
    except OSError:
        root_text = _posix(root)
    root_text = root_text.rstrip("/")
    return text == root_text or text.startswith(root_text + "/")


def assert_allowed_acquisition_root(
    acquisition_root: Path,
    *,
    extra_s0: tuple[Path, ...] = (),
    extra_materialize: tuple[Path, ...] = (),
) -> None:
    root = Path(acquisition_root)
    s0_roots = (S0_SCRATCH, *extra_s0)
    mat_roots = (PRODUCTION_MATERIALIZE_ROOT, *extra_materialize)
    for forbidden in s0_roots:
        if _descends_from(root, forbidden):
            raise AcquisitionError("refuses to touch S0 scratch")
    for forbidden in mat_roots:
        if _descends_from(root, forbidden):
            raise AcquisitionError("materialization root is not an acquisition root")


def _no_url(obj: Any) -> None:
    blob = json.dumps(obj, ensure_ascii=False).casefold()
    if "http://" in blob or "https://" in blob:
        raise AcquisitionError("raw URL is forbidden in acquisition manifest")


def _assert_local_ref(raw: str) -> None:
    text = str(raw).strip()
    if REMOTE_RE.match(text):
        raise AcquisitionError("non-local artifact path refused")
    if "://" in text and not text.startswith("file:"):
        raise AcquisitionError("non-local artifact path refused")


def _basename_only(filename: str) -> None:
    name = str(filename)
    _assert_local_ref(name)
    if not name or name != Path(name).name:
        raise AcquisitionError("artifact.filename must be a basename")
    if "/" in name or "\\" in name or ".." in name:
        raise AcquisitionError("artifact.filename must be a basename")
    lowered = name.casefold()
    if lowered.endswith(INCOMPLETE_SUFFIXES):
        raise AcquisitionError("temp/partial artifact is not acquired")


def revision_is_forbidden_label(revision: str | None) -> bool:
    if revision is None:
        return False
    return revision.strip().casefold() in FORBIDDEN_REVISION_LABELS


def revision_is_immutable(revision: str | None) -> bool:
    if not isinstance(revision, str):
        return False
    return bool(IMMUTABLE_REVISION_RE.match(revision))


def is_production_acquisition_ready(row: dict[str, Any]) -> bool:
    rev = row.get("upstream", {}).get("revision")
    if rev is None:
        return False
    if revision_is_forbidden_label(rev):
        return False
    return revision_is_immutable(rev)


def validate_manifest_document(doc: dict[str, Any]) -> None:
    extra = set(doc) - set(MANIFEST_KEYS)
    if "content_sha256" in extra or (
        isinstance(doc.get("artifact"), dict) and "content_sha256" in doc["artifact"]
    ):
        raise AcquisitionError("content_sha256 is materialization identity, not raw acquisition")
    if extra:
        raise AcquisitionError(f"manifest extra keys: {sorted(extra)}")
    missing = [k for k in MANIFEST_KEYS if k not in doc]
    if missing:
        raise AcquisitionError(f"manifest missing keys: {missing}")
    if doc["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise AcquisitionError("manifest schema_version")
    if doc["status"] != "complete":
        raise AcquisitionError("status != complete")
    up = doc["upstream"]
    if not isinstance(up, dict) or set(up) != {"repository", "subset", "revision"}:
        raise AcquisitionError("manifest upstream")
    if revision_is_forbidden_label(up.get("revision")):
        raise AcquisitionError("revision is not an immutable upstream identity")
    if up.get("revision") is not None and not revision_is_immutable(up["revision"]):
        raise AcquisitionError("revision is not an immutable upstream identity")
    art = doc["artifact"]
    if not isinstance(art, dict) or set(art) != set(ARTIFACT_KEYS):
        raise AcquisitionError("manifest artifact")
    _basename_only(str(art["filename"]))
    if art["format"] not in RAW_FORMATS:
        raise AcquisitionError("artifact.format")
    if not isinstance(art["bytes"], int) or art["bytes"] < 1:
        raise AcquisitionError("artifact.bytes")
    if not isinstance(art["raw_content_sha256"], str) or not SHA256_RE.match(art["raw_content_sha256"]):
        raise AcquisitionError("malformed raw_content_sha256")
    sel = doc["selection"]
    if not isinstance(sel, dict) or set(sel) != set(SELECTION_KEYS):
        raise AcquisitionError("manifest selection")
    if sel["strategy"] != SELECTION_STRATEGY:
        raise AcquisitionError("selection.strategy")
    if not isinstance(sel["bound_bytes"], int) or sel["bound_bytes"] < 1:
        raise AcquisitionError("selection.bound_bytes")
    if sel["bound_bytes"] > SOURCE_MAX_MATERIALIZED_BYTES:
        raise AcquisitionError("selection.bound_bytes exceeds 4 GiB")
    if art["bytes"] > sel["bound_bytes"]:
        raise AcquisitionError("artifact bytes exceed selection.bound_bytes")
    _no_url(doc)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _governance_row(allowlist: dict[str, Any], source_id: str) -> dict[str, Any]:
    row = source_index(allowlist).get(source_id)
    if row is None:
        raise AcquisitionError("unknown source_id")
    cap = int(row["max_materialized_bytes"])
    if cap > SOURCE_MAX_MATERIALIZED_BYTES:
        raise AcquisitionError("source bound exceeds 4 GiB")
    return row


def _upstream_of(row: dict[str, Any]) -> dict[str, Any]:
    up = row["upstream"]
    return {"repository": up["repository"], "subset": up["subset"], "revision": up["revision"]}


def iter_complete_acquisitions(
    acquisition_root: Path,
    allowlist: dict[str, Any],
    *,
    extra_s0: tuple[Path, ...] = (),
    extra_materialize: tuple[Path, ...] = (),
    free_bytes: int | None = None,
    completed_bytes: int | None = None,
    skip_production_ready: bool = True,
) -> list[dict[str, Any]]:
    """Yield validated complete artifacts. Invalid pairs are skipped, not counted."""
    found: list[dict[str, Any]] = []
    root = Path(acquisition_root)
    if not root.exists():
        return found
    for manifest_path in sorted(root.rglob("*.acquisition.json")):
        try:
            found.append(
                validate_acquired_artifact(
                    json.loads(manifest_path.read_text(encoding="utf-8")).get("source_id", ""),
                    manifest_path,
                    root,
                    allowlist,
                    extra_s0=extra_s0,
                    extra_materialize=extra_materialize,
                    free_bytes=free_bytes,
                    completed_bytes=completed_bytes,
                    require_production_ready=not skip_production_ready,
                )
            )
        except (AcquisitionError, json.JSONDecodeError, OSError, TypeError, ValueError):
            continue
    return found


def completed_acquired_bytes(
    acquisition_root: Path,
    allowlist: dict[str, Any],
    **kwargs: Any,
) -> int:
    return sum(int(item["bytes"]) for item in iter_complete_acquisitions(acquisition_root, allowlist, **kwargs))


def validate_acquired_artifact(
    source_id: str,
    manifest_path: str | Path,
    acquisition_root: str | Path,
    allowlist: dict[str, Any] | None = None,
    *,
    extra_s0: tuple[Path, ...] = (),
    extra_materialize: tuple[Path, ...] = (),
    free_bytes: int | None = None,
    input_bytes: int | None = None,
    completed_bytes: int | None = None,
    require_production_ready: bool = True,
) -> dict[str, Any]:
    """Prove a completed local raw artifact satisfies the acquisition contract.

    Does not write files. Does not convert formats. Does not materialize or canary.
    """
    doc = allowlist if allowlist is not None else load_allowlist(ALLOWLIST_PATH)
    row = _governance_row(doc, source_id)
    root = Path(acquisition_root)
    assert_allowed_acquisition_root(root, extra_s0=extra_s0, extra_materialize=extra_materialize)

    man_ref = str(manifest_path).strip()
    _assert_local_ref(man_ref)
    man = Path(manifest_path)
    if not man.exists() or not man.is_file():
        raise AcquisitionError("manifest not found")

    try:
        manifest = json.loads(man.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AcquisitionError("manifest is not JSON") from exc
    if not isinstance(manifest, dict):
        raise AcquisitionError("manifest is not an object")
    validate_manifest_document(manifest)
    if manifest["source_id"] != source_id:
        raise AcquisitionError("manifest source_id mismatch")

    gov_up = _upstream_of(row)
    if manifest["upstream"] != gov_up:
        raise AcquisitionError("upstream identity mismatch")
    if manifest["upstream"]["revision"] != gov_up["revision"]:
        raise AcquisitionError("revision mismatch")

    sel_bound = int(manifest["selection"]["bound_bytes"])
    source_cap = int(row["max_materialized_bytes"])
    if sel_bound > source_cap:
        raise AcquisitionError("selection.bound_bytes exceeds source cap")
    if source_cap > SOURCE_MAX_MATERIALIZED_BYTES:
        raise AcquisitionError("source bound exceeds 4 GiB")

    filename = str(manifest["artifact"]["filename"])
    raw_path = (man.parent / filename)
    _assert_local_ref(str(raw_path))
    try:
        raw_resolved = raw_path.resolve()
        root_resolved = root.resolve() if root.exists() else root
        man_resolved = man.resolve()
    except OSError as exc:
        raise AcquisitionError("cannot resolve acquisition path") from exc
    if not _descends_from(raw_resolved, root_resolved) or not _descends_from(man_resolved, root_resolved):
        raise AcquisitionError("artifact is outside acquisition root")
    if not raw_path.exists() or not raw_path.is_file():
        raise AcquisitionError("raw local file missing")
    if raw_path.is_dir():
        raise AcquisitionError("raw local file missing")

    nbytes = int(input_bytes) if input_bytes is not None else raw_path.stat().st_size
    if nbytes < 1:
        raise AcquisitionError("empty raw artifact")
    if nbytes != int(manifest["artifact"]["bytes"]):
        raise AcquisitionError("raw byte-size mismatch")
    actual_sha = _sha256_file(raw_path)
    if actual_sha != manifest["artifact"]["raw_content_sha256"]:
        raise AcquisitionError("raw SHA mismatch")

    measure = root if root.exists() else (root.parent if root.parent.exists() else Path.cwd())
    available = int(free_bytes) if free_bytes is not None else shutil.disk_usage(measure).free
    if available < MIN_FREE_BYTES:
        raise AcquisitionError("free disk below 20 GiB")

    done = int(completed_bytes) if completed_bytes is not None else 0
    if completed_bytes is None and root.exists():
        done = 0
        for other in root.rglob("*.acquisition.json"):
            if other.resolve() == man_resolved:
                continue
            try:
                other_doc = json.loads(other.read_text(encoding="utf-8"))
                if not isinstance(other_doc, dict) or other_doc.get("status") != "complete":
                    continue
                sibling = other.parent / str(other_doc.get("artifact", {}).get("filename", ""))
                if sibling.is_file() and sibling.stat().st_size == int(other_doc["artifact"]["bytes"]):
                    if _sha256_file(sibling) == other_doc["artifact"]["raw_content_sha256"]:
                        done += int(other_doc["artifact"]["bytes"])
            except (AcquisitionError, KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                continue
    if done + nbytes > GLOBAL_MAX_MATERIALIZED_BYTES:
        raise AcquisitionError("projected global completed raw exceeds 8 GiB")

    if require_production_ready and not is_production_acquisition_ready(row):
        raise AcquisitionError("production acquisition closed: immutable revision not frozen")

    return {
        "source_id": source_id,
        "path": raw_resolved,
        "manifest_path": man_resolved,
        "format": manifest["artifact"]["format"],
        "bytes": nbytes,
        "raw_content_sha256": actual_sha,
        "upstream": dict(gov_up),
        "selection": dict(manifest["selection"]),
        "status": "complete",
        "production_acquisition_ready": True,
    }


def production_sources_readiness(allowlist: dict[str, Any] | None = None) -> dict[str, bool]:
    doc = allowlist if allowlist is not None else load_allowlist(ALLOWLIST_PATH)
    return {row["source_id"]: is_production_acquisition_ready(row) for row in doc["sources"]}
