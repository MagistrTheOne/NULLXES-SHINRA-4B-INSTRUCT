"""Parquet → JSONL adapter for DATA V1. No Hugging Face. No canary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from data.data_v1 import PhaseBError
from data.data_v1.normalize import contains_raw_url
from data.data_v1.phase_a import DOMAINS
from data.data_v1.sources import SOURCE_MAX_MATERIALIZED_BYTES


class AdapterError(PhaseBError):
    """Format adapter failure."""


def _uri_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cell(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def _assert_metadata_offline(record: dict[str, Any]) -> None:
    """Sidecar/provenance must not carry a raw URL. Document body may cite http(s)."""
    if "url" in record:
        raise AdapterError("adapter refused to emit a raw URL")
    meta = {key: value for key, value in record.items() if key != "text"}
    if contains_raw_url(json.dumps(meta, ensure_ascii=False)):
        raise AdapterError("adapter refused to emit a raw URL")


def adapt_parquet_to_jsonl(
    parquet_path: str | Path,
    jsonl_path: str | Path,
    *,
    source_id: str,
    snapshot: str,
    domain: str = "general",
    max_bytes: int = SOURCE_MAX_MATERIALIZED_BYTES,
) -> dict[str, Any]:
    """Write local JSONL. Drops FineWeb `url`; hashes it. Body text may contain http(s)."""
    src = Path(parquet_path)
    dest = Path(jsonl_path)
    if src.suffix.lower() != ".parquet":
        raise AdapterError("adapter expects parquet")
    if not src.is_file():
        raise AdapterError("parquet missing")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise AdapterError("pyarrow is required for the parquet adapter") from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    records = 0
    written = 0
    pf = pq.ParquetFile(src)
    names = set(pf.schema_arrow.names)
    if "text" not in names:
        raise AdapterError("parquet missing text column")
    if domain not in DOMAINS:
        raise AdapterError("adapter domain not in frozen enum")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for batch in pf.iter_batches():
                cols = {name: batch.column(name) for name in batch.schema.names}
                for i in range(batch.num_rows):
                    text = _cell(cols["text"][i])
                    if not isinstance(text, str) or not text.strip():
                        continue
                    rec: dict[str, Any] = {
                        "text": text,
                        "source_id": source_id,
                        "source_type": "natural",
                        "domain": domain,
                        "language": "en",
                        "split": "train",
                        "license": {"id": "ODC-By-1.0", "redistribution": True},
                    }
                    url = _cell(cols["url"][i]) if "url" in cols else None
                    rec["provenance"] = {
                        "uri_hash": _uri_hash(url) if isinstance(url, str) and url.strip() else _uri_hash(text),
                        "snapshot": snapshot,
                    }
                    _assert_metadata_offline(rec)
                    line = json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                    blob = line.encode("utf-8")
                    written += len(blob)
                    if written > max_bytes:
                        raise AdapterError("adapter JSONL exceeds 4 GiB")
                    fh.write(line)
                    records += 1
        if records < 1:
            raise AdapterError("adapter produced no records")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return {"path": dest.resolve(), "records": records, "bytes": dest.stat().st_size}
