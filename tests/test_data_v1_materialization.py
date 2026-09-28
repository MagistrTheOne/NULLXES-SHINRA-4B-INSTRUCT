"""DATA V1 materialization ABI: scratch receipt, no network, no git-allowlist mutate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data.data_v1.materialize import (
    CREATED_BY,
    GLOBAL_MAX_MATERIALIZED_BYTES,
    MIN_FREE_BYTES,
    SOURCE_MAX_MATERIALIZED_BYTES,
    MaterializationError,
    materialize,
    resolve_materialization,
    slice_id_for,
)
from data.data_v1.sources import ALLOWLIST_PATH, load_allowlist

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "data" / "data_v1"
SOURCE_ID = "fineweb-edu-en"
FREE = 50 * 1024**3
PHASE_A_MANIFEST = "c18fba5f054475fc9f87242aa048039e1ae9e3cbc860852af152647ef7d035dc"
PHASE_A_BUNDLE = "eb6d0b7552f646e2782d744ae0542f240fb0c350f980b05fb6a337b7bffc87b9"


def _allowlist() -> dict:
    return load_allowlist()


def _write_input(path: Path, records: list[dict] | None = None) -> Path:
    rows = records or [
        {"text": "The clay path dried before noon.", "language": "en", "domain": "general", "z": 1},
        {"text": "Copper filings stayed in the labelled drawer.", "language": "en", "domain": "knowledge"},
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _run(tmp_path: Path, src: Path, **kwargs):
    scratch = kwargs.pop("scratch_root", tmp_path / "scratch")
    return materialize(
        src,
        kwargs.pop("source_id", SOURCE_ID),
        scratch_root=scratch,
        allowlist=_allowlist(),
        free_bytes=kwargs.pop("free_bytes", FREE),
        extra_forbidden=kwargs.pop("extra_forbidden", ()),
        **kwargs,
    )


def test_local_jsonl_materializes(tmp_path: Path):
    src = _write_input(tmp_path / "in.jsonl")
    scratch = tmp_path / "scratch"
    state = _run(tmp_path, src, scratch_root=scratch)
    jsonl = Path(state["path"])
    receipt_path = Path(state["receipt_path"])
    assert jsonl.is_file()
    assert receipt_path.is_file()
    raw = jsonl.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert state["content_sha256"] == f"sha256:{digest}"
    assert state["slice_id"] == f"{SOURCE_ID}-{digest[:16]}"
    assert state["slice_id"] == slice_id_for(SOURCE_ID, state["content_sha256"])
    assert jsonl.name == f"{state['slice_id']}.jsonl"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["bytes"] == len(raw) == state["bytes"]
    assert receipt["records"] == 2
    assert receipt["created_by"] == CREATED_BY
    assert receipt["upstream"] == _allowlist()["sources"][0]["upstream"]
    assert receipt["input"] == {"format": "jsonl"}
    assert "http" not in json.dumps(receipt).casefold()
    resolved = resolve_materialization(SOURCE_ID, _allowlist(), scratch)
    assert resolved["content_sha256"] == state["content_sha256"]


def test_no_overwrite_and_idempotent(tmp_path: Path):
    src = _write_input(tmp_path / "in.jsonl")
    scratch = tmp_path / "scratch"
    first = _run(tmp_path, src, scratch_root=scratch)
    raw = Path(first["path"]).read_bytes()
    mtime = Path(first["path"]).stat().st_mtime_ns
    second = _run(tmp_path, src, scratch_root=scratch)
    assert second["slice_id"] == first["slice_id"]
    assert Path(second["path"]).read_bytes() == raw
    assert Path(second["path"]).stat().st_mtime_ns == mtime


def test_conflicting_existing_artifact_fails(tmp_path: Path):
    src = _write_input(tmp_path / "in.jsonl")
    scratch = tmp_path / "scratch"
    state = _run(tmp_path, src, scratch_root=scratch)
    Path(state["path"]).write_bytes(Path(state["path"]).read_bytes() + b"\n")
    with pytest.raises(MaterializationError, match="collision"):
        _run(tmp_path, src, scratch_root=scratch)


def test_url_and_hf_rejected(tmp_path: Path):
    with pytest.raises(MaterializationError, match="non-local"):
        _run(tmp_path, "https://example.com/slice.jsonl")
    with pytest.raises(MaterializationError, match="non-local"):
        _run(tmp_path, "hf://datasets/HuggingFaceFW/fineweb-edu")
    with pytest.raises(MaterializationError, match="non-local"):
        _run(tmp_path, "s3://bucket/slice.jsonl")


def test_missing_empty_dir_extension(tmp_path: Path):
    with pytest.raises(MaterializationError, match="missing input"):
        _run(tmp_path, tmp_path / "nope.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    with pytest.raises(MaterializationError, match="empty input"):
        _run(tmp_path, empty)
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n\n", encoding="utf-8")
    with pytest.raises(MaterializationError, match="empty input"):
        _run(tmp_path, blank)
    folder = tmp_path / "dir.jsonl"
    folder.mkdir()
    with pytest.raises(MaterializationError, match="directory"):
        _run(tmp_path, folder)
    txt = tmp_path / "x.txt"
    txt.write_text('{"text": "nope"}\n', encoding="utf-8")
    with pytest.raises(MaterializationError, match="unsupported extension"):
        _run(tmp_path, txt)
    parquet = tmp_path / "x.parquet"
    parquet.write_bytes(b"PAR1")
    with pytest.raises(MaterializationError, match="unsupported extension"):
        _run(tmp_path, parquet)


def test_malformed_jsonl_rejected(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(MaterializationError, match="malformed JSONL"):
        _run(tmp_path, bad)
    missing_text = tmp_path / "no-text.jsonl"
    missing_text.write_text('{"language": "en"}\n', encoding="utf-8")
    with pytest.raises(MaterializationError, match="malformed JSONL"):
        _run(tmp_path, missing_text)


def test_free_disk_below_20_gib_fails(tmp_path: Path):
    src = _write_input(tmp_path / "in.jsonl")
    with pytest.raises(MaterializationError, match="20 GiB"):
        _run(tmp_path, src, free_bytes=MIN_FREE_BYTES - 1)


def test_source_raw_over_4_gib_fails_without_allocating(tmp_path: Path):
    src = _write_input(tmp_path / "in.jsonl")
    with pytest.raises(MaterializationError, match="4 GiB"):
        _run(tmp_path, src, input_bytes=SOURCE_MAX_MATERIALIZED_BYTES + 1)


def test_projected_global_over_8_gib_fails_without_allocating(tmp_path: Path):
    src = _write_input(tmp_path / "in.jsonl")
    with pytest.raises(MaterializationError, match="8 GiB"):
        _run(tmp_path, src, completed_bytes=GLOBAL_MAX_MATERIALIZED_BYTES)


def test_crash_before_rename_leaves_temp_no_receipt(tmp_path: Path):
    src = _write_input(tmp_path / "in.jsonl")
    scratch = tmp_path / "scratch"
    with pytest.raises(MaterializationError, match="injected crash after_tmp_write"):
        _run(tmp_path, src, scratch_root=scratch, fail_at="after_tmp_write")
    leftover = list(scratch.rglob("*.jsonl.tmp"))
    assert leftover
    assert not list(scratch.rglob("*.receipt.json"))
    assert not list(p for p in scratch.rglob("*.jsonl") if not p.name.endswith(".tmp"))
    with pytest.raises(MaterializationError, match="unresolved"):
        resolve_materialization(SOURCE_ID, _allowlist(), scratch)


def test_jsonl_without_receipt_is_not_materialized(tmp_path: Path):
    src = _write_input(tmp_path / "in.jsonl")
    scratch = tmp_path / "scratch"
    with pytest.raises(MaterializationError, match="injected crash after_jsonl_rename"):
        _run(tmp_path, src, scratch_root=scratch, fail_at="after_jsonl_rename")
    jsonls = list(p for p in scratch.rglob("*.jsonl") if not p.name.endswith(".tmp"))
    assert jsonls
    assert not list(scratch.rglob("*.receipt.json"))
    with pytest.raises(MaterializationError, match="unresolved"):
        resolve_materialization(SOURCE_ID, _allowlist(), scratch)


def test_receipt_without_jsonl_is_not_materialized(tmp_path: Path):
    scratch = tmp_path / "scratch" / SOURCE_ID
    scratch.mkdir(parents=True)
    receipt = {
        "schema_version": "shinra-data-v1-materialization-receipt",
        "source_id": SOURCE_ID,
        "slice_id": f"{SOURCE_ID}-aaaaaaaaaaaaaaaa",
        "content_sha256": "sha256:" + "a" * 64,
        "bytes": 12,
        "records": 1,
        "created_by": CREATED_BY,
        "upstream": dict(_allowlist()["sources"][0]["upstream"]),
        "input": {"format": "jsonl"},
    }
    (scratch / f"{receipt['slice_id']}.receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(MaterializationError, match="unresolved"):
        resolve_materialization(SOURCE_ID, _allowlist(), tmp_path / "scratch")


def test_receipt_sha_and_bytes_mismatch(tmp_path: Path):
    src = _write_input(tmp_path / "in.jsonl")
    scratch = tmp_path / "scratch"
    state = _run(tmp_path, src, scratch_root=scratch)
    receipt_path = Path(state["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["content_sha256"] = "sha256:" + "b" * 64
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(MaterializationError, match="SHA mismatch"):
        resolve_materialization(SOURCE_ID, _allowlist(), scratch)

    state = _run(tmp_path, _write_input(tmp_path / "in2.jsonl", [{"text": "Steel expands in heat."}]), scratch_root=tmp_path / "scratch2")
    receipt_path = Path(state["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["bytes"] = receipt["bytes"] + 1
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(MaterializationError, match="bytes mismatch"):
        resolve_materialization(SOURCE_ID, _allowlist(), tmp_path / "scratch2")


def test_receipt_source_id_and_upstream_and_filename(tmp_path: Path):
    src = _write_input(tmp_path / "in.jsonl")
    scratch = tmp_path / "scratch"
    state = _run(tmp_path, src, scratch_root=scratch)
    receipt_path = Path(state["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_id"] = "fineweb2-ru"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(MaterializationError, match="source_id mismatch"):
        resolve_materialization(SOURCE_ID, _allowlist(), scratch)

    scratch_u = tmp_path / "up"
    state = _run(tmp_path, src, scratch_root=scratch_u)
    receipt_path = Path(state["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["upstream"] = dict(receipt["upstream"])
    receipt["upstream"]["repository"] = "HuggingFaceFW/fineweb-2"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(MaterializationError, match="upstream mismatch"):
        resolve_materialization(SOURCE_ID, _allowlist(), scratch_u)

    scratch_n = tmp_path / "name"
    state = _run(tmp_path, src, scratch_root=scratch_n)
    receipt_path = Path(state["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["slice_id"] = f"{SOURCE_ID}-ffffffffffffffff"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(MaterializationError, match="wrong slice_id filename"):
        resolve_materialization(SOURCE_ID, _allowlist(), scratch_n)


def test_receipt_cannot_invent_revision(tmp_path: Path):
    src = _write_input(tmp_path / "in.jsonl")
    scratch = tmp_path / "scratch"
    state = _run(tmp_path, src, scratch_root=scratch)
    assert state["upstream"]["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    receipt_path = Path(state["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["upstream"] = dict(receipt["upstream"])
    receipt["upstream"]["revision"] = "deadbeef"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(MaterializationError, match="upstream mismatch"):
        resolve_materialization(SOURCE_ID, _allowlist(), scratch)


def test_scratch_root_injection_and_s0_guard(tmp_path: Path):
    src = _write_input(tmp_path / "in.jsonl")
    injected = tmp_path / "injected-root"
    state = _run(tmp_path, src, scratch_root=injected)
    assert str(injected) in str(state["path"])
    s0 = tmp_path / "shinra_scratch" / "s0"
    with pytest.raises(MaterializationError, match="S0"):
        _run(tmp_path, src, scratch_root=s0 / "data_v1", extra_forbidden=(s0,))


def test_production_allowlist_still_unresolved():
    doc = load_allowlist(ALLOWLIST_PATH)
    assert [row["source_id"] for row in doc["sources"]] == ["fineweb-edu-en", "fineweb2-ru"]
    for row in doc["sources"]:
        assert row["materialization"]["status"] == "not_materialized"
        assert row["materialization"]["slice_id"] is None
        assert row["materialization"]["content_sha256"] is None
    assert doc["sources"][0]["upstream"]["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert doc["sources"][1]["upstream"]["revision"] is None


def test_phase_a_hashes_unchanged():
    from data.data_v1.phase_a import validate_phase_a

    report = validate_phase_a()
    assert report["manifest_sha256"] == PHASE_A_MANIFEST
    assert report["probe_bundle_sha256"] == PHASE_A_BUNDLE


def test_no_network_libraries_in_materializer():
    banned = (
        "huggingface_hub",
        "urllib.request",
        "requests.get",
        "import requests",
        "httpx",
        "datasets.load_dataset",
        "snapshot_download",
        "datasets",
    )
    text = (ENGINE / "materialize.py").read_text(encoding="utf-8")
    for needle in banned:
        assert needle not in text, needle
