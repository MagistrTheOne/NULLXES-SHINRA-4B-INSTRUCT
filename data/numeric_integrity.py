"""Number-preserving cleaning and strict local corpus preparation primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from datasketch import MinHash, MinHashLSH
except ImportError as exc:  # pragma: no cover - exercised in minimal VM installations
    raise ImportError("corpus preparation requires the preinstalled 'datasketch' package") from exc

from configs.loader import load_config
from tokenizer.contract import normalize_text


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
    # NFC may change Unicode code points; whitespace and numerical spans are preserved.
    try:
        cleaned = normalize_text(text, normalization)
    except ValueError as exc:
        raise DataIntegrityError(str(exc)) from exc
    before = numeric_spans(text)
    after = numeric_spans(cleaned)
    if before != after:
        raise DataIntegrityError(f"numeric spans changed during cleaning: before={before!r}, after={after!r}")
    return cleaned


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
                         validation_fraction: float | None = None, blind_fraction: float | None = None,
                         normalization: str | None = None) -> dict[str, Any]:
    """Clean, provenance-stamp, deduplicate and split whole source documents."""
    data_config = load_config()["data"]
    cleaning = data_config["cleaning"]
    validation_fraction = float(data_config["splits"]["validation"] if validation_fraction is None else validation_fraction)
    blind_fraction = float(data_config["splits"]["blind"] if blind_fraction is None else blind_fraction)
    normalization = normalization or cleaning["unicode_normalization"]
    threshold = float(cleaning["near_dedup_threshold"])
    num_perm = int(cleaning["near_dedup_num_perm"])
    if not split_salt or not 0 < validation_fraction < 1 or not 0 < blind_fraction < 1 or validation_fraction + blind_fraction >= 1:
        raise DataIntegrityError("invalid immutable split policy")
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    existing_outputs = [output / name for name in (
        "train.jsonl", "validation.jsonl", "blind.jsonl", "duplicates.jsonl", "manifest.json"
    )]
    if any(path.exists() for path in existing_outputs):
        raise DataIntegrityError("refusing to overwrite an existing corpus audit output")
    stage = Path(tempfile.mkdtemp(prefix=".corpus-stage-", dir=output.parent))
    seen: dict[str, str] = {}
    shingles_by_hash: dict[str, set[str]] = {}
    near_index = MinHashLSH(threshold=threshold, num_perm=num_perm)
    duplicate_path = stage / "duplicates.jsonl"
    duplicate_handle = duplicate_path.open("w", encoding="utf-8", newline="\n")
    counts = {"seen": 0, "kept": 0, "exact_duplicates": 0, "near_duplicates": 0,
              "train": 0, "validation": 0, "blind": 0}
    split_paths = {name: stage / f"{name}.jsonl" for name in ("train", "validation", "blind")}
    handles = {name: path.open("w", encoding="utf-8", newline="\n") for name, path in split_paths.items()}
    try:
        for source in inputs:
            root = Path(source)
            if not root.exists():
                raise DataIntegrityError(f"source does not exist: {root}")
            for row in _records(root):
                counts["seen"] += 1
                text = clean_text(row["text"], normalization=normalization)
                if not text:
                    raise DataIntegrityError(f"{row['_path']}: empty cleaned record")
                if not isinstance(row.get("raw_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", row["raw_sha256"]):
                    raise DataIntegrityError(f"{row['_path']}: original raw_sha256 is required")
                raw_sha256 = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                if row["raw_sha256"] != raw_sha256:
                    raise DataIntegrityError(f"{row['_path']}: source raw_sha256 mismatch")
                if not all(isinstance(row.get(field), str) and row[field].strip()
                           for field in ("source_id", "revision", "license", "language")):
                    raise DataIntegrityError(f"{row['_path']}: source_id, revision, license and language are required")
                if row["language"].casefold() not in cleaning["filter_languages"]:
                    raise DataIntegrityError(f"{row['_path']}: language is outside the approved EN/RU set")
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    counts["exact_duplicates"] += 1
                    duplicate_handle.write(json.dumps({"type": "exact", "duplicate_of": seen[digest],
                        "source_record": row["_path"], "source_id": row.get("source_id"), "clean_sha256": digest},
                        ensure_ascii=False, sort_keys=True) + "\n")
                    continue
                words = re.findall(r"[\w]+", text.casefold())
                shingles = {" ".join(words[i:i + 5]) for i in range(max(1, len(words) - 4))}
                if not shingles:
                    shingles = {text.casefold()}
                signature = MinHash(num_perm=num_perm)
                for shingle in shingles:
                    signature.update(shingle.encode("utf-8"))
                near_match = None
                for candidate in near_index.query(signature):
                    other = shingles_by_hash[candidate]
                    similarity = len(shingles & other) / max(len(shingles | other), 1)
                    if similarity >= threshold:
                        near_match = candidate
                        break
                if near_match is not None:
                    counts["near_duplicates"] += 1
                    duplicate_handle.write(json.dumps({"type": "near", "similarity": similarity,
                        "duplicate_of": seen[near_match], "source_record": row["_path"],
                        "source_id": row.get("source_id"), "clean_sha256": digest},
                        ensure_ascii=False, sort_keys=True) + "\n")
                    continue
                declared_split = row.get("split")
                if declared_split is not None and (row.get("split_salt") != split_salt or row.get("split_unit") not in {"source_document", "work"}):
                    raise DataIntegrityError(f"{row['_path']}: split needs the frozen split_salt and document/work split unit")
                split_unit = row.get("split_unit", "source_document")
                split_key = row.get("work_id") or row.get("document_id") or row.get("source_record") or row["_path"]
                bucket = int(hashlib.sha256(f"{split_salt}:{split_unit}:{split_key}".encode()).hexdigest()[:16], 16) / 0xFFFFFFFFFFFFFFFF
                expected_split = "validation" if bucket < validation_fraction else "blind" if bucket < validation_fraction + blind_fraction else "train"
                if declared_split is not None and declared_split != expected_split:
                    raise DataIntegrityError(f"{row['_path']}: declared split conflicts with frozen split policy")
                split = expected_split
                seen[digest] = row["_path"]
                shingles_by_hash[digest] = shingles
                near_index.insert(digest, signature)
                record = {"text": text, "source_id": row["source_id"], "revision": row["revision"],
                          "license": row["license"], "source_record": row.get("document_id", row["_path"]),
                          "work_id": row.get("work_id"), "document_id": row.get("document_id"),
                          "language": row.get("language"), "domain": row.get("domain"),
                          "transform_version": "numeric-integrity.v3-nfc-whitespace-preserved",
                          "raw_sha256": raw_sha256,
                          "clean_sha256": digest, "numeric_spans": numeric_spans(text), "split": split}
                handles[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                counts["kept"] += 1
                counts[split] += 1
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        for handle in handles.values():
            handle.close()
        duplicate_handle.close()
    split_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in split_paths.items()}
    files_hashes = {f"{name}.jsonl": digest for name, digest in split_hashes.items()}
    files_hashes["duplicates.jsonl"] = hashlib.sha256(duplicate_path.read_bytes()).hexdigest()
    manifest = {"schema": "nullxes.shinra.corpus.v1", "split_salt_sha256": hashlib.sha256(split_salt.encode()).hexdigest(),
                "counts": counts, "files": files_hashes, "split_hashes": split_hashes,
                "train_hash": split_hashes["train"], "validation_hash": split_hashes["validation"],
                "blind_hash": split_hashes["blind"]}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = stage / "manifest.json"
    manifest_tmp = stage / ".manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    committed: list[Path] = []
    try:
        for name in split_paths:
            destination = output / f".{name}.jsonl.tmp"
            shutil.copyfile(split_paths[name], destination)
            final_path = output / f"{name}.jsonl"
            destination.replace(final_path)
            committed.append(final_path)
        duplicates_destination = output / ".duplicates.jsonl.tmp"
        shutil.copyfile(duplicate_path, duplicates_destination)
        final_duplicates = output / "duplicates.jsonl"
        duplicates_destination.replace(final_duplicates)
        committed.append(final_duplicates)
        created_dirs = []
        for name in ("train", "validation", "blind"):
            split_dir = output / name
            if not split_dir.exists():
                split_dir.mkdir()
                created_dirs.append(split_dir)
        manifest_path = output / "manifest.json"
        temporary_manifest = output / ".manifest.json.tmp"
        shutil.copyfile(stage / "manifest.json", temporary_manifest)
        temporary_manifest.replace(manifest_path)
        committed.append(manifest_path)
    except Exception:
        for path in committed:
            path.unlink(missing_ok=True)
        for split_dir in locals().get("created_dirs", []):
            try:
                split_dir.rmdir()
            except OSError:
                pass
        for temp_path in output.glob(".*.jsonl.tmp"):
            temp_path.unlink(missing_ok=True)
        (output / ".manifest.json.tmp").unlink(missing_ok=True)
        raise
    shutil.rmtree(stage, ignore_errors=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and prepare a local SHINRA corpus slice")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/shinra.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = prepare_local_corpus(
        args.input,
        args.output_dir,
        split_salt=config["data"]["split_salt"],
        normalization=config["data"]["cleaning"]["unicode_normalization"],
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
