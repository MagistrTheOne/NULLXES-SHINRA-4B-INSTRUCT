"""DATA V1 materialization ABI: local JSONL → scratch slice + receipt.

Does not download. Does not call Hugging Face. Does not mutate git allowlist.
Does not open real canary, training, or S1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

RECEIPT_SCHEMA_VERSION = "shinra-data-v1-materialization-receipt"
CREATED_BY = "shinra-data-v1-materializer"
PRODUCTION_SCRATCH = Path("/content/shinra_scratch/data_v1")
S0_SCRATCH = Path("/content/shinra_scratch/s0")
MIN_FREE_BYTES = 20 * (1024**3)
SLICE_SHA_PREFIX = 16
REMOTE_RE = re.compile(r"^(https?|hf|s3|gs|ftp)://", re.IGNORECASE)
RECEIPT_KEYS = (
    "schema_version",
    "source_id",
    "slice_id",
    "content_sha256",
    "bytes",
    "records",
    "created_by",
    "upstream",
    "input",
)


class MaterializationError(PhaseBError):
    """Scratch materialization / receipt failure. Fail closed."""


def _posix(path: Path) -> str:
    return path.as_posix().replace("\\", "/").casefold()


def assert_away_from_s0(path: Path, extra_forbidden: tuple[Path, ...] = ()) -> None:
    resolved = path if path.exists() else path
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise MaterializationError("cannot resolve materialization path") from exc
    text = _posix(resolved)
    roots = (S0_SCRATCH, *extra_forbidden)
    for root in roots:
        try:
            root_text = _posix(root.resolve()) if root.exists() else _posix(root)
        except OSError:
            root_text = _posix(root)
        root_text = root_text.rstrip("/")
        if text == root_text or text.startswith(root_text + "/"):
            raise MaterializationError("refuses to touch S0 scratch")


def _ensure_local_jsonl(raw: str | Path) -> Path:
    text = str(raw).strip()
    if not text:
        raise MaterializationError("missing input")
    if REMOTE_RE.match(text):
        raise MaterializationError("non-local input refused")
    if "://" in text and not text.startswith("file:"):
        raise MaterializationError("non-local input refused")
    path = Path(text)
    if path.exists() and path.is_dir():
        raise MaterializationError("directory when file expected")
    if path.suffix.lower() != ".jsonl":
        raise MaterializationError("unsupported extension")
    if not path.exists() or not path.is_file():
        raise MaterializationError("missing input")
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _digest_hex(content_sha256: str) -> str:
    return content_sha256.split(":", 1)[1]


def slice_id_for(source_id: str, content_sha256: str) -> str:
    hex_digest = _digest_hex(content_sha256)
    if len(hex_digest) != 64 or any(c not in "0123456789abcdef" for c in hex_digest):
        raise MaterializationError("content_sha256 is not a lowercase sha256")
    return f"{source_id}-{hex_digest[:SLICE_SHA_PREFIX]}"


def _dump_record(obj: dict[str, Any]) -> str:
    # Serialization ABI only: sorted keys, compact separators, UTF-8.
    # Field values are not rewritten. Not a corpus semantic transform.
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_jsonl_bytes(source: Path) -> tuple[bytes, int]:
    raw = source.read_bytes()
    if not raw:
        raise MaterializationError("empty input")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MaterializationError("input is not UTF-8") from exc
    lines_out: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MaterializationError(f"malformed JSONL at line {line_no}") from exc
        if not isinstance(obj, dict):
            raise MaterializationError(f"malformed JSONL at line {line_no}")
        if "text" not in obj or not isinstance(obj["text"], str):
            raise MaterializationError(f"malformed JSONL at line {line_no}: text required")
        lines_out.append(_dump_record(obj) + "\n")
    if not lines_out:
        raise MaterializationError("empty input")
    payload = "".join(lines_out).encode("utf-8")
    return payload, len(lines_out)


def _fsync_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _atomic_replace(tmp: Path, dest: Path) -> None:
    os.replace(tmp, dest)


def _no_url(obj: Any) -> None:
    blob = json.dumps(obj, ensure_ascii=False).casefold()
    if "http://" in blob or "https://" in blob:
        raise MaterializationError("raw URL is forbidden in receipt")


def validate_receipt(doc: dict[str, Any]) -> None:
    extra = set(doc) - set(RECEIPT_KEYS)
    if extra:
        raise MaterializationError(f"receipt extra keys: {sorted(extra)}")
    missing = [k for k in RECEIPT_KEYS if k not in doc]
    if missing:
        raise MaterializationError(f"receipt missing keys: {missing}")
    if doc["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise MaterializationError("receipt schema_version")
    if doc["created_by"] != CREATED_BY:
        raise MaterializationError("receipt created_by")
    if not SHA256_RE.match(str(doc["content_sha256"])):
        raise MaterializationError("receipt content_sha256")
    if not isinstance(doc["bytes"], int) or doc["bytes"] < 1:
        raise MaterializationError("receipt bytes")
    if not isinstance(doc["records"], int) or doc["records"] < 1:
        raise MaterializationError("receipt records")
    up = doc["upstream"]
    if not isinstance(up, dict) or set(up) != {"repository", "subset", "revision"}:
        raise MaterializationError("receipt upstream")
    inp = doc["input"]
    if not isinstance(inp, dict) or inp.get("format") != "jsonl" or set(inp) != {"format"}:
        raise MaterializationError("receipt input")
    _no_url(doc)


def _pair_paths(scratch_root: Path, source_id: str, slice_id: str) -> tuple[Path, Path]:
    folder = scratch_root / source_id
    return folder / f"{slice_id}.jsonl", folder / f"{slice_id}.receipt.json"


def _line_count(path: Path) -> int:
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            n += 1
    return n


def validate_artifact_pair(
    jsonl_path: Path,
    receipt_path: Path,
    *,
    source_id: str,
    allowlist_row: dict[str, Any],
) -> dict[str, Any]:
    if not jsonl_path.is_file() or not receipt_path.is_file():
        raise MaterializationError("materialization unresolved")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MaterializationError("receipt is not JSON") from exc
    if not isinstance(receipt, dict):
        raise MaterializationError("receipt is not an object")
    validate_receipt(receipt)
    if receipt["source_id"] != source_id:
        raise MaterializationError("receipt source_id mismatch")
    if jsonl_path.stem != receipt["slice_id"]:
        raise MaterializationError("wrong slice_id filename")
    receipt_stem = (
        receipt_path.name[: -len(".receipt.json")]
        if receipt_path.name.endswith(".receipt.json")
        else receipt_path.stem
    )
    if receipt_stem != receipt["slice_id"]:
        raise MaterializationError("wrong slice_id filename")
    gov_up = allowlist_row["upstream"]
    if receipt["upstream"] != {
        "repository": gov_up["repository"],
        "subset": gov_up["subset"],
        "revision": gov_up["revision"],
    }:
        raise MaterializationError("receipt upstream mismatch")
    actual_sha = _sha256_file(jsonl_path)
    if actual_sha != receipt["content_sha256"]:
        raise MaterializationError("receipt SHA mismatch")
    if receipt["slice_id"] != slice_id_for(source_id, actual_sha):
        raise MaterializationError("receipt slice_id does not match SHA formula")
    nbytes = jsonl_path.stat().st_size
    if nbytes != receipt["bytes"]:
        raise MaterializationError("receipt bytes mismatch")
    records = _line_count(jsonl_path)
    if records != receipt["records"]:
        raise MaterializationError("receipt records mismatch")
    if nbytes > SOURCE_MAX_MATERIALIZED_BYTES:
        raise MaterializationError("source raw exceeds 4 GiB")
    return {
        "status": "materialized",
        "source_id": source_id,
        "slice_id": receipt["slice_id"],
        "path": jsonl_path,
        "receipt_path": receipt_path,
        "content_sha256": actual_sha,
        "bytes": nbytes,
        "records": records,
        "upstream": dict(receipt["upstream"]),
        "receipt": receipt,
    }


def iter_valid_artifacts(
    scratch_root: Path,
    allowlist: dict[str, Any],
    *,
    extra_forbidden: tuple[Path, ...] = (),
) -> list[dict[str, Any]]:
    assert_away_from_s0(scratch_root, extra_forbidden)
    index = source_index(allowlist)
    found: list[dict[str, Any]] = []
    if not scratch_root.exists():
        return found
    for source_dir in sorted(p for p in scratch_root.iterdir() if p.is_dir()):
        sid = source_dir.name
        row = index.get(sid)
        if row is None:
            continue
        for receipt_path in sorted(source_dir.glob("*.receipt.json")):
            slice_id = receipt_path.name[: -len(".receipt.json")]
            jsonl_path = source_dir / f"{slice_id}.jsonl"
            try:
                found.append(
                    validate_artifact_pair(
                        jsonl_path, receipt_path, source_id=sid, allowlist_row=row
                    )
                )
            except MaterializationError:
                continue
    return found


def completed_materialized_bytes(
    scratch_root: Path,
    allowlist: dict[str, Any],
    *,
    extra_forbidden: tuple[Path, ...] = (),
) -> int:
    return sum(
        int(item["bytes"])
        for item in iter_valid_artifacts(scratch_root, allowlist, extra_forbidden=extra_forbidden)
    )


def resolve_materialization(
    source_id: str,
    allowlist: dict[str, Any],
    scratch_root: str | Path,
    *,
    extra_forbidden: tuple[Path, ...] = (),
) -> dict[str, Any]:
    root = Path(scratch_root)
    assert_away_from_s0(root, extra_forbidden)
    index = source_index(allowlist)
    row = index.get(source_id)
    if row is None:
        raise MaterializationError("unknown source_id")
    source_dir = root / source_id
    if not source_dir.is_dir():
        raise MaterializationError("materialization unresolved")
    pairs: list[tuple[Path, Path]] = []
    for receipt_path in sorted(source_dir.glob("*.receipt.json")):
        slice_id = receipt_path.name[: -len(".receipt.json")]
        pairs.append((source_dir / f"{slice_id}.jsonl", receipt_path))
    if not pairs:
        raise MaterializationError("materialization unresolved")
    valid: list[dict[str, Any]] = []
    errors: list[MaterializationError] = []
    for jsonl_path, receipt_path in pairs:
        try:
            valid.append(
                validate_artifact_pair(
                    jsonl_path, receipt_path, source_id=source_id, allowlist_row=row
                )
            )
        except MaterializationError as exc:
            errors.append(exc)
    if len(valid) > 1:
        raise MaterializationError("ambiguous materialization")
    if len(valid) == 1:
        item = valid[0]
        total = completed_materialized_bytes(root, allowlist, extra_forbidden=extra_forbidden)
        if item["bytes"] > SOURCE_MAX_MATERIALIZED_BYTES:
            raise MaterializationError("source raw exceeds 4 GiB")
        if total > GLOBAL_MAX_MATERIALIZED_BYTES:
            raise MaterializationError("projected global materialization exceeds 8 GiB")
        return item
    if errors:
        raise errors[0]
    raise MaterializationError("materialization unresolved")


def _governance_row(allowlist: dict[str, Any], source_id: str) -> dict[str, Any]:
    row = source_index(allowlist).get(source_id)
    if row is None:
        raise MaterializationError("unknown source_id")
    if row.get("acquisition_method") != "local_materialized":
        raise MaterializationError("source is not local_materialized")
    return row


def materialize(
    input_path: str | Path,
    source_id: str,
    *,
    scratch_root: str | Path | None = None,
    allowlist: dict[str, Any] | None = None,
    allowlist_path: str | Path | None = None,
    free_bytes: int | None = None,
    input_bytes: int | None = None,
    completed_bytes: int | None = None,
    extra_forbidden: tuple[Path, ...] = (),
    fail_at: str | None = None,
) -> dict[str, Any]:
    """Write a canonical JSONL slice + receipt under scratch_root.

    fail_at is a test-only crash injector: after_tmp_write | after_jsonl_rename | after_receipt_tmp
    """
    doc = allowlist if allowlist is not None else load_allowlist(Path(allowlist_path) if allowlist_path else ALLOWLIST_PATH)
    row = _governance_row(doc, source_id)
    root = Path(scratch_root) if scratch_root is not None else PRODUCTION_SCRATCH
    assert_away_from_s0(root, extra_forbidden)
    source = _ensure_local_jsonl(input_path)
    assert_away_from_s0(source, extra_forbidden)

    source_dir = root / source_id
    assert_away_from_s0(source_dir, extra_forbidden)

    in_size = int(input_bytes) if input_bytes is not None else source.stat().st_size
    if in_size < 1:
        raise MaterializationError("empty input")
    if in_size > SOURCE_MAX_MATERIALIZED_BYTES:
        raise MaterializationError("input exceeds 4 GiB")

    measure = root if root.exists() else (root.parent if root.parent.exists() else Path.cwd())
    available = int(free_bytes) if free_bytes is not None else shutil.disk_usage(measure).free
    if available < MIN_FREE_BYTES:
        raise MaterializationError("free disk below 20 GiB")

    done = (
        int(completed_bytes)
        if completed_bytes is not None
        else completed_materialized_bytes(root, doc, extra_forbidden=extra_forbidden)
    )
    payload, n_records = _canonical_jsonl_bytes(source)
    out_size = len(payload)
    if out_size > SOURCE_MAX_MATERIALIZED_BYTES:
        raise MaterializationError("input exceeds 4 GiB")
    if done + out_size > GLOBAL_MAX_MATERIALIZED_BYTES:
        raise MaterializationError("projected global materialization exceeds 8 GiB")

    existing = [
        item
        for item in iter_valid_artifacts(root, doc, extra_forbidden=extra_forbidden)
        if item["source_id"] == source_id
    ]

    source_dir.mkdir(parents=True, exist_ok=True)
    tmp_jsonl = source_dir / f"{source_id}.inprogress.jsonl.tmp"
    if tmp_jsonl.exists():
        tmp_jsonl.unlink()
    _fsync_write(tmp_jsonl, payload)
    temp_sha = _sha256_file(tmp_jsonl)
    if fail_at == "after_tmp_write":
        raise MaterializationError("injected crash after_tmp_write")
    if _digest_hex(temp_sha) != hashlib.sha256(payload).hexdigest():
        raise MaterializationError("temp SHA mismatch")

    slice_id = slice_id_for(source_id, temp_sha)
    final_jsonl, final_receipt = _pair_paths(root, source_id, slice_id)
    assert_away_from_s0(final_jsonl, extra_forbidden)
    assert_away_from_s0(final_receipt, extra_forbidden)

    if existing:
        if len(existing) == 1 and existing[0]["slice_id"] == slice_id and existing[0]["content_sha256"] == temp_sha:
            tmp_jsonl.unlink(missing_ok=True)
            return existing[0]
        if any(item["slice_id"] != slice_id or item["content_sha256"] != temp_sha for item in existing):
            tmp_jsonl.unlink(missing_ok=True)
            raise MaterializationError("existing artifact collision")

    if final_jsonl.exists() or final_receipt.exists():
        try:
            state = validate_artifact_pair(
                final_jsonl, final_receipt, source_id=source_id, allowlist_row=row
            )
        except MaterializationError as exc:
            tmp_jsonl.unlink(missing_ok=True)
            raise MaterializationError("existing artifact collision") from exc
        if state["content_sha256"] == temp_sha and state["slice_id"] == slice_id:
            tmp_jsonl.unlink(missing_ok=True)
            return state
        tmp_jsonl.unlink(missing_ok=True)
        raise MaterializationError("existing artifact collision")

    _atomic_replace(tmp_jsonl, final_jsonl)
    if fail_at == "after_jsonl_rename":
        raise MaterializationError("injected crash after_jsonl_rename")
    final_sha = _sha256_file(final_jsonl)
    if final_sha != temp_sha:
        raise MaterializationError("final SHA differs from temp SHA")

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_id": source_id,
        "slice_id": slice_id,
        "content_sha256": final_sha,
        "bytes": final_jsonl.stat().st_size,
        "records": n_records,
        "created_by": CREATED_BY,
        "upstream": {
            "repository": row["upstream"]["repository"],
            "subset": row["upstream"]["subset"],
            "revision": row["upstream"]["revision"],
        },
        "input": {"format": "jsonl"},
    }
    validate_receipt(receipt)
    tmp_receipt = source_dir / f"{slice_id}.receipt.json.tmp"
    if tmp_receipt.exists():
        tmp_receipt.unlink()
    _fsync_write(tmp_receipt, (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    if fail_at == "after_receipt_tmp":
        raise MaterializationError("injected crash after_receipt_tmp")
    _atomic_replace(tmp_receipt, final_receipt)
    return validate_artifact_pair(final_jsonl, final_receipt, source_id=source_id, allowlist_row=row)


def main() -> None:
    parser = argparse.ArgumentParser(description="SHINRA DATA V1 local JSONL materializer (no network)")
    parser.add_argument("--input", required=True, help="already-local UTF-8 JSONL")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--scratch", type=Path, default=PRODUCTION_SCRATCH)
    parser.add_argument("--allowlist", type=Path, default=None)
    args = parser.parse_args()
    state = materialize(
        args.input,
        args.source_id,
        scratch_root=args.scratch,
        allowlist_path=args.allowlist,
    )
    public = {k: (str(v) if isinstance(v, Path) else v) for k, v in state.items() if k != "receipt"}
    print(json.dumps(public, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
