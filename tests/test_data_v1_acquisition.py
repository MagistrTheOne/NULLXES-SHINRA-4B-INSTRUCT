"""DATA V1 acquisition contract: local manifest + raw metadata. No network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data.data_v1.acquisition import (
    GLOBAL_MAX_MATERIALIZED_BYTES,
    MIN_FREE_BYTES,
    SOURCE_MAX_MATERIALIZED_BYTES,
    AcquisitionError,
    is_production_acquisition_ready,
    production_sources_readiness,
    validate_acquired_artifact,
    validate_manifest_document,
)
from data.data_v1.sources import ALLOWLIST_PATH, load_allowlist

ROOT = Path(__file__).resolve().parents[1]
ACQ_PY = ROOT / "data" / "data_v1" / "acquisition.py"
SOURCE_ID = "fineweb-edu-en"
FREE = 50 * 1024**3
FROZEN_REV = "a" * 40
PAYLOAD = b"PAR1-test-bytes\n"
PHASE_A_MANIFEST = "c18fba5f054475fc9f87242aa048039e1ae9e3cbc860852af152647ef7d035dc"
PHASE_A_BUNDLE = "eb6d0b7552f646e2782d744ae0542f240fb0c350f980b05fb6a337b7bffc87b9"


def _allowlist() -> dict:
    return load_allowlist()


def _frozen_allowlist() -> dict:
    doc = json.loads(json.dumps(_allowlist()))
    for row in doc["sources"]:
        row["upstream"] = dict(row["upstream"])
        row["upstream"]["revision"] = FROZEN_REV
    return doc


def _sha(payload: bytes = PAYLOAD) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _manifest(
    *,
    source_id: str = SOURCE_ID,
    filename: str = "slice.parquet",
    fmt: str = "parquet",
    payload: bytes = PAYLOAD,
    revision: str | None = None,
    bound: int | None = None,
    status: str = "complete",
) -> dict:
    row = next(r for r in _allowlist()["sources"] if r["source_id"] == source_id)
    up = dict(row["upstream"])
    up["revision"] = revision if revision is not None else row["upstream"]["revision"]
    nbytes = len(payload)
    return {
        "schema_version": "shinra-data-v1-acquisition-manifest",
        "source_id": source_id,
        "upstream": up,
        "artifact": {
            "filename": filename,
            "format": fmt,
            "bytes": nbytes,
            "raw_content_sha256": _sha(payload),
        },
        "selection": {"strategy": "bounded_bytes", "bound_bytes": bound or max(nbytes, 1024)},
        "status": status,
    }


def _plant(tmp_path: Path, doc: dict, payload: bytes = PAYLOAD) -> tuple[Path, Path, Path]:
    root = tmp_path / "acq"
    folder = root / doc["source_id"]
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / doc["artifact"]["filename"]
    raw.write_bytes(payload)
    manifest = folder / "slice.acquisition.json"
    manifest.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root, raw, manifest


def test_schema_accepts_structurally_valid_completed_manifest():
    validate_manifest_document(_manifest(revision=None))


def test_unknown_source_id_rejected(tmp_path: Path):
    doc = _manifest()
    doc["source_id"] = "wikipedia-en"
    root, _, man = _plant(tmp_path, doc)
    with pytest.raises(AcquisitionError, match="unknown source"):
        validate_acquired_artifact("wikipedia-en", man, root, _frozen_allowlist(), free_bytes=FREE)


def test_repository_subset_revision_mismatch(tmp_path: Path):
    frozen = _frozen_allowlist()
    doc = _manifest(revision=FROZEN_REV)
    doc["upstream"] = dict(doc["upstream"])
    doc["upstream"]["repository"] = "HuggingFaceFW/fineweb-2"
    root, _, man = _plant(tmp_path, doc)
    with pytest.raises(AcquisitionError, match="upstream identity mismatch"):
        validate_acquired_artifact(SOURCE_ID, man, root, frozen, free_bytes=FREE)

    doc = _manifest(revision=FROZEN_REV)
    doc["upstream"] = dict(doc["upstream"])
    doc["upstream"]["subset"] = "rus_Cyrl"
    root, _, man = _plant(tmp_path / "sub", doc)
    with pytest.raises(AcquisitionError, match="upstream identity mismatch"):
        validate_acquired_artifact(SOURCE_ID, man, root, frozen, free_bytes=FREE)

    doc = _manifest(revision=FROZEN_REV)
    doc["upstream"] = dict(doc["upstream"])
    doc["upstream"]["revision"] = "b" * 40
    root, _, man = _plant(tmp_path / "rev", doc)
    with pytest.raises(AcquisitionError, match="upstream identity mismatch|revision mismatch"):
        validate_acquired_artifact(SOURCE_ID, man, root, frozen, free_bytes=FREE)


def test_null_revision_is_not_moving_head_and_not_production_ready():
    doc = _allowlist()
    assert production_sources_readiness(doc) == {"fineweb-edu-en": False, "fineweb2-ru": False}
    for row in doc["sources"]:
        assert row["upstream"]["revision"] is None
        assert is_production_acquisition_ready(row) is False
    schema_doc = _manifest(revision=None)
    validate_manifest_document(schema_doc)
    assert schema_doc["upstream"]["revision"] is None


def test_handoff_closed_when_governance_revision_null(tmp_path: Path):
    doc = _manifest(revision=None)
    root, _, man = _plant(tmp_path, doc)
    with pytest.raises(AcquisitionError, match="immutable revision not frozen"):
        validate_acquired_artifact(SOURCE_ID, man, root, _allowlist(), free_bytes=FREE)


def test_missing_empty_manifest_and_raw(tmp_path: Path):
    frozen = _frozen_allowlist()
    doc = _manifest(revision=FROZEN_REV)
    root, raw, man = _plant(tmp_path, doc)
    with pytest.raises(AcquisitionError, match="manifest not found"):
        validate_acquired_artifact(SOURCE_ID, root / SOURCE_ID / "nope.acquisition.json", root, frozen, free_bytes=FREE)
    raw.unlink()
    with pytest.raises(AcquisitionError, match="raw local file missing"):
        validate_acquired_artifact(SOURCE_ID, man, root, frozen, free_bytes=FREE)

    empty_doc = _manifest(revision=FROZEN_REV)
    empty_root, empty_raw, empty_man = _plant(tmp_path / "empty", empty_doc)
    empty_raw.write_bytes(b"")
    with pytest.raises(AcquisitionError, match="empty raw artifact"):
        validate_acquired_artifact(SOURCE_ID, empty_man, empty_root, frozen, free_bytes=FREE)

    only_raw = tmp_path / "raw-only" / SOURCE_ID
    only_raw.mkdir(parents=True)
    (only_raw / "slice.parquet").write_bytes(PAYLOAD)
    assert not list(only_raw.glob("*.acquisition.json"))
    with pytest.raises(AcquisitionError, match="manifest not found"):
        validate_acquired_artifact(
            SOURCE_ID, only_raw / "slice.acquisition.json", tmp_path / "raw-only", frozen, free_bytes=FREE
        )


def test_status_and_temp_rejected():
    with pytest.raises(AcquisitionError, match="status"):
        validate_manifest_document(_manifest(status="incomplete"))
    with pytest.raises(AcquisitionError, match="temp/partial"):
        validate_manifest_document(_manifest(filename="slice.parquet.part"))
    with pytest.raises(AcquisitionError, match="temp/partial"):
        validate_manifest_document(_manifest(filename="slice.parquet.tmp"))


def test_url_hf_s3_rejected():
    with pytest.raises(AcquisitionError, match="basename|non-local"):
        validate_manifest_document(_manifest(filename="https://example.com/x.parquet"))
    with pytest.raises(AcquisitionError, match="non-local"):
        validate_acquired_artifact(SOURCE_ID, "hf://datasets/x", Path("."), _frozen_allowlist(), free_bytes=FREE)
    with pytest.raises(AcquisitionError, match="non-local"):
        validate_acquired_artifact(SOURCE_ID, "s3://bucket/x.acquisition.json", Path("."), _frozen_allowlist(), free_bytes=FREE)
    with pytest.raises(AcquisitionError, match="non-local"):
        validate_acquired_artifact(SOURCE_ID, "https://example.com/m.json", Path("."), _frozen_allowlist(), free_bytes=FREE)


def test_raw_sha_and_bytes_mismatch(tmp_path: Path):
    frozen = _frozen_allowlist()
    doc = _manifest(revision=FROZEN_REV)
    doc["artifact"] = dict(doc["artifact"])
    doc["artifact"]["raw_content_sha256"] = "sha256:" + "b" * 64
    root, _, man = _plant(tmp_path, doc)
    with pytest.raises(AcquisitionError, match="raw SHA mismatch"):
        validate_acquired_artifact(SOURCE_ID, man, root, frozen, free_bytes=FREE)

    doc = _manifest(revision=FROZEN_REV)
    doc["artifact"] = dict(doc["artifact"])
    doc["artifact"]["bytes"] = len(PAYLOAD) + 7
    root, _, man = _plant(tmp_path / "sz", doc)
    with pytest.raises(AcquisitionError, match="byte-size mismatch"):
        validate_acquired_artifact(SOURCE_ID, man, root, frozen, free_bytes=FREE)


def test_malformed_sha_rejected():
    doc = _manifest()
    doc["artifact"] = dict(doc["artifact"])
    doc["artifact"]["raw_content_sha256"] = "sha256:not-a-hash"
    with pytest.raises(AcquisitionError, match="malformed"):
        validate_manifest_document(doc)
    doc["artifact"]["raw_content_sha256"] = "deadbeef"
    with pytest.raises(AcquisitionError, match="malformed"):
        validate_manifest_document(doc)


def test_bounds_rejected_without_allocating():
    with pytest.raises(AcquisitionError, match="4 GiB"):
        validate_manifest_document(_manifest(bound=SOURCE_MAX_MATERIALIZED_BYTES + 1))
    doc = _manifest()
    doc["artifact"] = dict(doc["artifact"])
    doc["selection"] = dict(doc["selection"])
    doc["selection"]["bound_bytes"] = 8
    doc["artifact"]["bytes"] = 9
    with pytest.raises(AcquisitionError, match="artifact bytes exceed"):
        validate_manifest_document(doc)


def test_global_and_disk_and_source_cap(tmp_path: Path):
    frozen = _frozen_allowlist()
    doc = _manifest(revision=FROZEN_REV)
    root, _, man = _plant(tmp_path, doc)
    with pytest.raises(AcquisitionError, match="8 GiB"):
        validate_acquired_artifact(
            SOURCE_ID, man, root, frozen, free_bytes=FREE, completed_bytes=GLOBAL_MAX_MATERIALIZED_BYTES
        )
    with pytest.raises(AcquisitionError, match="20 GiB"):
        validate_acquired_artifact(SOURCE_ID, man, root, frozen, free_bytes=MIN_FREE_BYTES - 1)
    bloated = json.loads(json.dumps(frozen))
    bloated["sources"][0]["max_materialized_bytes"] = SOURCE_MAX_MATERIALIZED_BYTES + 1
    with pytest.raises(AcquisitionError, match="4 GiB"):
        validate_acquired_artifact(SOURCE_ID, man, root, bloated, free_bytes=FREE)
    tight = json.loads(json.dumps(frozen))
    tight["sources"][0]["max_materialized_bytes"] = 512
    with pytest.raises(AcquisitionError, match="source cap"):
        validate_acquired_artifact(SOURCE_ID, man, root, tight, free_bytes=FREE)


def test_root_injection_s0_and_materialize_namespace(tmp_path: Path):
    frozen = _frozen_allowlist()
    doc = _manifest(revision=FROZEN_REV)
    injected = tmp_path / "injected-acq"
    root, _, man = _plant(injected, doc)
    state = validate_acquired_artifact(SOURCE_ID, man, root, frozen, free_bytes=FREE)
    assert str(injected) in str(state["path"])

    s0 = tmp_path / "shinra_scratch" / "s0"
    with pytest.raises(AcquisitionError, match="S0"):
        validate_acquired_artifact(SOURCE_ID, man, s0 / "acq", frozen, free_bytes=FREE, extra_s0=(s0,))

    mat = tmp_path / "shinra_scratch" / "data_v1"
    with pytest.raises(AcquisitionError, match="materialization root"):
        validate_acquired_artifact(SOURCE_ID, man, mat, frozen, free_bytes=FREE, extra_materialize=(mat,))


def test_validator_does_not_invent_revision(tmp_path: Path):
    doc = _manifest(revision=FROZEN_REV)
    root, _, man = _plant(tmp_path, doc)
    with pytest.raises(AcquisitionError, match="upstream identity mismatch|revision"):
        validate_acquired_artifact(SOURCE_ID, man, root, _allowlist(), free_bytes=FREE)
    assert _allowlist()["sources"][0]["upstream"]["revision"] is None


def test_forbidden_revision_labels_rejected():
    for label in ("HEAD", "main", "latest", "PENDING"):
        doc = _manifest(revision=None)
        doc["upstream"] = dict(doc["upstream"])
        doc["upstream"]["revision"] = label
        with pytest.raises(AcquisitionError, match="immutable"):
            validate_manifest_document(doc)


def test_raw_sha_distinct_and_success_is_metadata_only(tmp_path: Path):
    frozen = _frozen_allowlist()
    doc = _manifest(revision=FROZEN_REV)
    assert "content_sha256" not in doc
    assert "content_sha256" not in doc["artifact"]
    bad = dict(doc)
    bad["content_sha256"] = _sha()
    with pytest.raises(AcquisitionError, match="materialization identity"):
        validate_manifest_document(bad)
    root, raw, man = _plant(tmp_path, doc)
    before = {p.relative_to(tmp_path): p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    state = validate_acquired_artifact(SOURCE_ID, man, root, frozen, free_bytes=FREE)
    after = {p.relative_to(tmp_path): p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    assert state["raw_content_sha256"] == _sha()
    assert state["bytes"] == len(PAYLOAD)
    assert state["format"] == "parquet"
    assert state["upstream"]["revision"] == FROZEN_REV
    assert state["path"] == raw.resolve()
    assert "content_sha256" not in state


def test_production_allowlist_unresolved_and_root_not_created():
    doc = load_allowlist(ALLOWLIST_PATH)
    for row in doc["sources"]:
        assert row["materialization"]["status"] == "not_materialized"
        assert row["materialization"]["slice_id"] is None
        assert row["materialization"]["content_sha256"] is None
        assert row["upstream"]["revision"] is None
        assert is_production_acquisition_ready(row) is False
    assert "mkdir" not in ACQ_PY.read_text(encoding="utf-8")


def test_phase_a_hashes_unchanged():
    from data.data_v1.phase_a import validate_phase_a

    report = validate_phase_a()
    assert report["manifest_sha256"] == PHASE_A_MANIFEST
    assert report["probe_bundle_sha256"] == PHASE_A_BUNDLE


def test_no_network_or_download_in_acquisition_module():
    text = ACQ_PY.read_text(encoding="utf-8")
    for needle in (
        "huggingface_hub",
        "datasets",
        "requests",
        "httpx",
        "urllib.request",
        "subprocess",
        "wget",
        "curl",
        "git clone",
        "snapshot_download",
        "load_dataset",
        "data.data_v1.materialize",
        "data.data_v1.phase_b",
        "pyarrow",
        "pandas",
    ):
        assert needle not in text, needle
