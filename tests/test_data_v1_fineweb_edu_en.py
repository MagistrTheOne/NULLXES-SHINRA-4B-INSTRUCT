"""FineWeb-Edu EN frozen shard: gated acquire, parquet adapter, pipeline. CPU only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.data_v1.acquire_hf import acquire_fineweb_edu_en
from data.data_v1.adapt_parquet import AdapterError, _assert_metadata_offline, adapt_parquet_to_jsonl
from data.data_v1.acquisition import AcquisitionError, validate_manifest_document
from data.data_v1.plan_fineweb_edu_en import (
    BOUND_BYTES,
    DOMAIN,
    EXPECTED_RAW_SHA256,
    FILENAME,
    REVISION,
    SOURCE_ID,
    STRATEGY,
    UPSTREAM_PATH,
    frozen_plan,
)
from data.data_v1.run_fineweb_edu_en import run_fineweb_edu_en_slice
from data.data_v1.sources import load_allowlist

FREE = 50 * 1024**3


def _parquet(path: Path) -> Path:
    table = pa.table(
        {
            "text": [
                "The clay path dried before noon and the archivist closed the grey ledger.",
                "Copper filings stayed in the labelled drawer while the clerk drank water.",
            ],
            "url": ["https://example.com/a", "https://example.com/b"],
        }
    )
    pq.write_table(table, path)
    return path


def test_frozen_plan_identity():
    plan = frozen_plan()
    assert plan["revision"] == REVISION == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert plan["path"] == UPSTREAM_PATH == "sample/10BT/013_00000.parquet"
    assert plan["strategy"] == STRATEGY == "single_frozen_file"
    assert plan["domain"] == DOMAIN == "general"
    assert plan["bound_bytes"] == BOUND_BYTES == 4294967296
    assert plan["expected_raw_sha256"] == EXPECTED_RAW_SHA256
    assert EXPECTED_RAW_SHA256.startswith("sha256:")
    assert "content_sha256" not in plan


def test_single_frozen_file_manifest_roundtrip(tmp_path: Path):
    raw = tmp_path / FILENAME
    raw.write_bytes(b"not-the-hub-bytes")
    sha = "sha256:" + hashlib.sha256(raw.read_bytes()).hexdigest()
    doc = {
        "schema_version": "shinra-data-v1-acquisition-manifest",
        "source_id": SOURCE_ID,
        "upstream": {
            "repository": "HuggingFaceFW/fineweb-edu",
            "subset": None,
            "revision": REVISION,
        },
        "artifact": {
            "filename": FILENAME,
            "format": "parquet",
            "bytes": raw.stat().st_size,
            "raw_content_sha256": sha,
        },
        "selection": {
            "strategy": "single_frozen_file",
            "bound_bytes": BOUND_BYTES,
            "path": UPSTREAM_PATH,
            "expected_raw_sha256": sha,
        },
        "status": "complete",
    }
    validate_manifest_document(doc)


def test_adapter_drops_url_and_writes_jsonl(tmp_path: Path):
    parquet = _parquet(tmp_path / "tiny.parquet")
    out = tmp_path / "out.jsonl"
    state = adapt_parquet_to_jsonl(parquet, out, source_id=SOURCE_ID, snapshot=REVISION)
    assert state["records"] == 2
    lines = out.read_text(encoding="utf-8").splitlines()
    for line in lines:
        rec = json.loads(line)
        assert "text" in rec
        assert rec["source_id"] == SOURCE_ID
        assert rec["language"] == "en"
        assert rec["domain"] == "general"
        assert "url" not in rec
        meta = json.dumps({k: v for k, v in rec.items() if k != "text"}).casefold()
        assert "http://" not in meta and "https://" not in meta
        assert rec["provenance"]["snapshot"] == REVISION


def test_adapter_keeps_http_in_body_text(tmp_path: Path):
    path = tmp_path / "body-url.parquet"
    pq.write_table(
        pa.table(
            {
                "text": ["See https://example.edu/notes for the lecture on clay paths."],
                "url": ["https://example.com/page"],
            }
        ),
        path,
    )
    out = tmp_path / "out.jsonl"
    state = adapt_parquet_to_jsonl(path, out, source_id=SOURCE_ID, snapshot=REVISION)
    assert state["records"] == 1
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "https://example.edu/notes" in rec["text"]
    assert "url" not in rec
    meta = json.dumps({k: v for k, v in rec.items() if k != "text"}).casefold()
    assert "http://" not in meta and "https://" not in meta
    assert rec["provenance"]["uri_hash"].startswith("sha256:")


def test_adapter_refuses_raw_url_in_metadata():
    with pytest.raises(AdapterError, match="raw URL"):
        _assert_metadata_offline(
            {
                "text": "The clay path dried before noon.",
                "provenance": {"uri_hash": "sha256:" + "a" * 64, "snapshot": "https://evil.example/x"},
            }
        )


def test_adapter_requires_text_column(tmp_path: Path):
    path = tmp_path / "no-text.parquet"
    pq.write_table(pa.table({"body": ["nope"]}), path)
    with pytest.raises(AdapterError, match="text column"):
        adapt_parquet_to_jsonl(path, tmp_path / "out.jsonl", source_id=SOURCE_ID, snapshot=REVISION)


def test_gated_acquire_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    parquet = _parquet(tmp_path / "hub.parquet")
    sha = "sha256:" + hashlib.sha256(parquet.read_bytes()).hexdigest()
    monkeypatch.setattr("data.data_v1.acquire_hf.EXPECTED_RAW_SHA256", sha)

    def fetch(repo, revision, path, dest: Path) -> None:
        assert repo == "HuggingFaceFW/fineweb-edu"
        assert revision == REVISION
        assert path == UPSTREAM_PATH
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(parquet.read_bytes())

    acq_root = tmp_path / "acq"
    state = acquire_fineweb_edu_en(
        acquisition_root=acq_root, fetch_file=fetch, free_bytes=FREE
    )
    assert state["raw_content_sha256"] == sha
    assert Path(state["path"]).name == FILENAME
    assert Path(state["manifest_path"]).is_file()
    assert json.loads(Path(state["manifest_path"]).read_text(encoding="utf-8"))["selection"]["strategy"] == "single_frozen_file"


def test_gated_acquire_sha_mismatch(tmp_path: Path):
    def fetch(repo, revision, path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"wrong-bytes")

    with pytest.raises(AcquisitionError, match="raw SHA mismatch"):
        acquire_fineweb_edu_en(acquisition_root=tmp_path / "acq", fetch_file=fetch, free_bytes=FREE)


def test_pipeline_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    parquet = _parquet(tmp_path / "hub.parquet")
    sha = "sha256:" + hashlib.sha256(parquet.read_bytes()).hexdigest()
    monkeypatch.setattr("data.data_v1.acquire_hf.EXPECTED_RAW_SHA256", sha)

    def fetch(repo, revision, path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(parquet.read_bytes())

    out = run_fineweb_edu_en_slice(
        acquisition_root=tmp_path / "acq",
        materialize_root=tmp_path / "mat",
        fetch_file=fetch,
        free_bytes=FREE,
    )
    assert out["adapted_records"] == 2
    jsonl = Path(out["materialized"]["path"])
    receipt = Path(out["materialized"]["receipt_path"])
    assert jsonl.is_file()
    assert receipt.is_file()
    assert json.loads(receipt.read_text(encoding="utf-8"))["source_id"] == SOURCE_ID
    body = jsonl.read_text(encoding="utf-8")
    assert "http" not in body.casefold()
