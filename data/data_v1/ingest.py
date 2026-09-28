"""Load local JSONL / text documents. Refuses URLs and remote schemes."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from data.data_v1 import PhaseBError

REMOTE_RE = re.compile(r"^(https?|hf|s3|gs|ftp)://", re.IGNORECASE)
TEXT_KEYS = ("text", "document", "body", "content")


def ensure_local_path(raw: str | Path) -> Path:
    text = str(raw).strip()
    if REMOTE_RE.match(text):
        raise PhaseBError("Phase B refuses non-local sources (no network)")
    if "://" in text and not text.startswith("file:"):
        raise PhaseBError("Phase B refuses URI sources")
    path = Path(text)
    if not path.exists():
        raise PhaseBError(f"source not found: {path}")
    return path.resolve()


def iter_source_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise PhaseBError(f"not a file or directory: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jsonl", ".txt", ".json"})
    if not files:
        raise PhaseBError(f"no local jsonl/txt/json documents under {root}")
    for path in files:
        if path.name.endswith(".sidecar.json"):
            continue
        yield path


def _text_from_obj(obj: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = obj.get(key)
        if isinstance(value, str):
            return value
    raise PhaseBError("record missing text/document/body")


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    root = ensure_local_path(path)
    for file in iter_source_files(root):
        suffix = file.suffix.lower()
        if suffix == ".jsonl":
            yield from _iter_jsonl(file)
        elif suffix == ".txt":
            yield {
                "text": file.read_text(encoding="utf-8"),
                "source_id": _slug(file.stem),
                "_path": str(file),
            }
        elif suffix == ".json":
            obj = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                raise PhaseBError(f"{file}: json document must be an object")
            if obj.get("schema_version") == "shinra-data-v1-sources":
                continue
            rec = dict(obj)
            rec["text"] = _text_from_obj(obj)
            rec["_path"] = str(file)
            yield rec
        else:
            continue


def _iter_jsonl(file: Path) -> Iterator[dict[str, Any]]:
    for line_no, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PhaseBError(f"{file}:{line_no}: invalid json") from exc
        if not isinstance(obj, dict):
            raise PhaseBError(f"{file}:{line_no}: record must be an object")
        rec = dict(obj)
        rec["text"] = _text_from_obj(obj)
        rec["_path"] = f"{file}:{line_no}"
        if "source_id" not in rec:
            rec["source_id"] = _slug(file.stem)
        yield rec


def _slug(stem: str) -> str:
    out = re.sub(r"[^a-z0-9._-]+", "-", stem.lower()).strip("-")
    return out or "local-document"
