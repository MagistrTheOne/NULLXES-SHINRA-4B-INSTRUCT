"""Number-preserving cleaning and strict local corpus preparation primitives."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator


class DataIntegrityError(ValueError):
    pass


# Keep signs, decimal/group separators, dates, exponents, units and adjacent spans.
NUMERIC_SPAN = re.compile(
    r"(?<![\w])[-+]?(?:\d{1,3}(?:[ ,\u00a0]\d{3})+|\d+)(?:[.,]\d+)?"
    r"(?:[eE][-+]?\d+)?(?:\s?(?:%|‰|°[CFК]?|[A-Za-zА-Яа-яЁёµμΩ]+(?:\s*/\s*[A-Za-zА-Яа-яЁёµμΩ]+)?))?"
    r"(?!\w)"
)


def numeric_spans(text: str) -> list[str]:
    return [match.group(0) for match in NUMERIC_SPAN.finditer(text)]


def clean_text(text: str, *, normalization: str = "NFC") -> str:
    if not isinstance(text, str):
        raise DataIntegrityError("text must be a string")
    # Normalization and newline cleanup only; numbers, signs and punctuation stay byte-for-byte equivalent.
    cleaned = unicodedata.normalize(normalization, text).replace("\x00", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    before = numeric_spans(text)
    after = numeric_spans(cleaned)
    if before != after:
        raise DataIntegrityError(f"numeric spans changed during cleaning: before={before!r}, after={after!r}")
    return cleaned.strip()


def _records(path: Path) -> Iterator[dict[str, Any]]:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".jsonl", ".json", ".txt"})
    if not files:
        raise DataIntegrityError(f"no local source records found: {path}")
    for file in files:
        if file.suffix.lower() == ".txt":
            text = file.read_text(encoding="utf-8", errors="strict")
            yield {"text": text, "source_id": file.stem, "raw_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "_path": str(file)}
        elif file.suffix.lower() == ".jsonl":
            with file.open("r", encoding="utf-8", errors="strict") as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        raise DataIntegrityError(f"{file}:{line_no}: blank records are forbidden")
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise DataIntegrityError(f"{file}:{line_no}: malformed JSON: {exc}") from exc
                    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                        raise DataIntegrityError(f"{file}:{line_no}: record must be an object with string text")
                    if not row.get("raw_sha256"):
                        raise DataIntegrityError(f"{file}:{line_no}: original raw_sha256 is required")
                    if row.get("source_id") == "local-document":
                        row["source_id"] = file.stem
                    row["_path"] = f"{file}:{line_no}"
                    yield row
        else:
            try:
                row = json.loads(file.read_text(encoding="utf-8", errors="strict"))
            except json.JSONDecodeError as exc:
                raise DataIntegrityError(f"{file}: malformed JSON: {exc}") from exc
            if isinstance(row, dict) and isinstance(row.get("text"), str):
                if not row.get("raw_sha256"):
                    raise DataIntegrityError(f"{file}: original raw_sha256 is required")
                if row.get("source_id") == "local-document":
                    row["source_id"] = file.stem
                row["_path"] = str(file)
                yield row
            else:
                raise DataIntegrityError(f"{file}: JSON document must have string text")


def prepare_local_corpus(inputs: Iterable[str | Path], output_dir: str | Path, *, split_salt: str,
                         validation_fraction: float = 0.01, blind_fraction: float = 0.01,
                         normalization: str = "NFC") -> dict[str, Any]:
    """Clean, provenance-stamp, exact-deduplicate and split whole source documents."""
    if not split_salt or not 0 < validation_fraction < 1 or not 0 < blind_fraction < 1 or validation_fraction + blind_fraction >= 1:
        raise DataIntegrityError("invalid immutable split policy")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".corpus-stage-", dir=output.parent))
    seen: dict[str, str] = {}
    counts = {"seen": 0, "kept": 0, "exact_duplicates": 0, "train": 0, "validation": 0, "blind": 0}
    split_paths = {name: stage / f"{name}.jsonl" for name in ("train", "validation", "blind")}
    handles = {name: path.open("w", encoding="utf-8", newline="\n") for name, path in split_paths.items()}
    try:
        for source in inputs:
            root = Path(source)
            if not root.exists():
                raise DataIntegrityError(f"source does not exist: {root}")
            if root.is_dir():
                source_root = root
            else:
                source_root = root.parent
            for row in _records(root):
                counts["seen"] += 1
                text = clean_text(row["text"], normalization=normalization)
                if not text:
                    raise DataIntegrityError(f"{row['_path']}: empty cleaned record")
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    counts["exact_duplicates"] += 1
                    continue
                if not isinstance(row.get("raw_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", row["raw_sha256"]):
                    raise DataIntegrityError(f"{row['_path']}: original raw_sha256 is required")
                raw_sha256 = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                if row["raw_sha256"] != raw_sha256:
                    raise DataIntegrityError(f"{row['_path']}: source raw_sha256 mismatch")
                if not all(isinstance(row.get(field), str) and row[field].strip() for field in ("source_id", "revision", "license")):
                    raise DataIntegrityError(f"{row['_path']}: source_id, immutable revision and license are required")
                bucket = int(hashlib.sha256(f"{split_salt}:{digest}".encode()).hexdigest()[:16], 16) / 0xFFFFFFFFFFFFFFFF
                split = row.get("split")
                if split is None:
                    split = "validation" if bucket < validation_fraction else "blind" if bucket < validation_fraction + blind_fraction else "train"
                if split not in {"train", "validation", "blind"}:
                    raise DataIntegrityError(f"{row['_path']}: unsupported split {split!r}")
                declared_split = row.get("split")
                if declared_split is not None and (row.get("split_salt") != split_salt or row.get("split_unit") not in {"source_document", "work"}):
                    raise DataIntegrityError(f"{row['_path']}: split needs the frozen split_salt and document/work split unit")
                seen[digest] = row["_path"]
                record = {"text": text, "source_id": row["source_id"], "revision": row["revision"],
                          "license": row["license"], "source_record": row.get("document_id", row["_path"]),
                          "language": row.get("language"), "domain": row.get("domain"),
                          "raw_sha256": raw_sha256,
                          "clean_sha256": digest, "numeric_spans": numeric_spans(text), "split": split}
                handles[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                counts["kept"] += 1
                counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    split_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in split_paths.items()}
    manifest = {"schema": "nullxes.shinra.corpus.v1", "split_salt_sha256": hashlib.sha256(split_salt.encode()).hexdigest(),
                "counts": counts, "files": split_hashes, "split_hashes": split_hashes,
                "train_hash": split_hashes["train"], "validation_hash": split_hashes["validation"],
                "blind_hash": split_hashes["blind"]}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = stage / "manifest.json"
    manifest_tmp = stage / ".manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    for name in split_paths:
        destination = output / f".{name}.jsonl.tmp"
        shutil.copyfile(split_paths[name], destination)
        destination.replace(output / f"{name}.jsonl")
    (output / "train").mkdir(exist_ok=True)
    (output / "validation").mkdir(exist_ok=True)
    (output / "blind").mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    temporary_manifest = output / ".manifest.json.tmp"
    shutil.copyfile(stage / "manifest.json", temporary_manifest)
    temporary_manifest.replace(manifest_path)
    shutil.rmtree(stage, ignore_errors=True)
    return manifest
