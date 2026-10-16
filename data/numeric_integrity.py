"""Auditable cleaning, disk-backed deduplication and document-level split assignment."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sqlite3
import struct
import tempfile
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from datasketch import MinHash
except ImportError as exc:  # pragma: no cover - depends on the prepared Colab image
    raise ImportError("corpus preparation requires the preinstalled 'datasketch' package") from exc

from configs.loader import CONFIG_PATH, load_config
from data.hashing import sha256_file
from tokenizer.contract import normalize_text


class DataIntegrityError(ValueError):
    pass


NUMERIC_SPAN = re.compile(
    r"(?<![\w])[-+]?(?:\d{1,3}(?:[ ,\u00a0]\d{3})+|\d+)(?:[.,]\d+)?"
    r"(?:[eE][-+]?\d+)?(?:\s?(?:%|‰|°[CFК]?|[A-Za-zА-Яа-яЁёµμΩ]+(?:\s*/\s*[A-Za-zА-Яа-яЁёµμΩ]+)?))?"
    r"(?!\w)"
)
WIKI_REF = re.compile(r"<ref\b[^>]*>.*?</ref\s*>|<ref\b[^>]*/\s*>", re.IGNORECASE | re.DOTALL)
WIKI_LINK = re.compile(r"\[\[([^\]]+)\]\]")
WIKI_EXTERNAL_LINK = re.compile(r"\[(?:https?://\S+)(?:\s+([^\]]+))?\]")
WIKI_FILE_CATEGORY = re.compile(r"\[\[(?:file|image|category):[^\]]*\]\]", re.IGNORECASE)
WIKI_FORMATTING = re.compile(r"'{2,5}")
TABLE_ROW = re.compile(r"^\s*(?:\|.*\||\+[-=+]{3,}\+)\s*$")
CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def numeric_spans(text: str) -> list[str]:
    return [match.group(0) for match in NUMERIC_SPAN.finditer(text)]


def clean_text(text: str, *, normalization: str = "NFC") -> str:
    if not isinstance(text, str):
        raise DataIntegrityError("text must be a string")
    try:
        cleaned = normalize_text(text, normalization)
    except ValueError as exc:
        raise DataIntegrityError(str(exc)) from exc
    before, after = numeric_spans(text), numeric_spans(cleaned)
    if before != after:
        raise DataIntegrityError(f"numeric spans changed during normalization: before={before!r}, after={after!r}")
    return cleaned


def _clean_source_markup(text: str, source: dict[str, Any]) -> str:
    if source.get("cleaning_profile") != "wikimedia_rendered_text":
        return text
    cleaned = html.unescape(text)
    cleaned = WIKI_REF.sub(" ", cleaned)
    cleaned = WIKI_FILE_CATEGORY.sub(" ", cleaned)
    cleaned = WIKI_EXTERNAL_LINK.sub(lambda match: match.group(1) or " ", cleaned)
    cleaned = WIKI_LINK.sub(lambda match: match.group(1).split("|", 1)[-1], cleaned)
    cleaned = WIKI_FORMATTING.sub("", cleaned)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?(?:div|span|small|big|center|blockquote|sup|sub|nowiki)\b[^>]*>", " ", cleaned,
                     flags=re.IGNORECASE)
    return cleaned


def _quarantine_reason(text: str, cleaning: dict[str, Any], source: dict[str, Any]) -> str | None:
    lines = text.splitlines()
    table_rows = sum(bool(TABLE_ROW.fullmatch(line)) for line in lines)
    if cleaning.get("quarantine_table_documents", True) and table_rows >= int(cleaning.get("table_rows_quarantine_threshold", 2)):
        return f"table_structure_without_reliable_context:{table_rows}_rows"
    if source.get("quarantine_profile") == "ocr_book" and "\ufffd" in text:
        return "ocr_or_decode_replacement_character"
    controls = len(CONTROL_CHARACTER.findall(text))
    if text and controls / len(text) > float(cleaning.get("max_control_character_ratio", 0.001)):
        return f"excessive_control_characters:{controls}"
    short_lines = [line.strip() for line in lines if 0 < len(line.strip()) <= 120]
    repeated = max(Counter(short_lines).values(), default=0)
    if source.get("quarantine_profile") == "ocr_book" \
            and repeated >= int(cleaning.get("repeated_short_line_quarantine_threshold", 12)):
        return f"repeated_page_header_or_ocr_line:{repeated}"
    return None


def _record_files(path: Path) -> tuple[Path, Path]:
    root = path if path.is_dir() else path.parent
    records = path / "records.jsonl" if path.is_dir() else path
    manifest = root / "manifest.json"
    if not records.is_file() or not manifest.is_file():
        raise DataIntegrityError(f"{path}: pinned raw records.jsonl and source manifest.json are required")
    return records, manifest


def _load_raw_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIntegrityError(f"{path}: invalid source manifest: {exc}") from exc
    declared = manifest.get("manifest_sha256")
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    actual = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if declared != actual:
        raise DataIntegrityError(f"{path}: source manifest self-hash mismatch")
    records = path.parent / "records.jsonl"
    if sha256_file(records) != manifest.get("records_sha256"):
        raise DataIntegrityError(f"{records}: SHA-256 differs from the raw source manifest")
    if manifest.get("schema") != "nullxes.shinra.raw-source.v1":
        raise DataIntegrityError(f"{path}: unsupported raw source manifest schema")
    return manifest


def _records(path: Path, source: dict[str, Any], raw_manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    records_path, _ = _record_files(path)
    upstream_files = {item["path"]: item["sha256"] for item in raw_manifest.get("files", [])}
    records_seen = 0
    with records_path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise DataIntegrityError(f"{records_path}:{line_no}: blank records are forbidden")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataIntegrityError(f"{records_path}:{line_no}: malformed JSON: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise DataIntegrityError(f"{records_path}:{line_no}: record must be an object with string text")
            if row.get("source_id") != source["id"] or row.get("revision") != source["revision"]:
                raise DataIntegrityError(f"{records_path}:{line_no}: source ID/revision is outside the exact allowlist")
            if raw_manifest.get("source_id") != row["source_id"] or raw_manifest.get("revision") != row["revision"]:
                raise DataIntegrityError(f"{records_path}:{line_no}: raw manifest and row source identity differ")
            if row.get("license") != source.get("license_claim") or row.get("rights_status") != source.get("rights_review_status"):
                raise DataIntegrityError(f"{records_path}:{line_no}: license or rights provenance differs from the allowlist")
            if row.get("upstream_file") not in upstream_files \
                    or upstream_files[row["upstream_file"]] != row.get("upstream_file_sha256"):
                raise DataIntegrityError(f"{records_path}:{line_no}: upstream file SHA differs from the raw source manifest")
            if row.get("language") != source.get("language_by_subset", {}).get(row.get("subset")):
                raise DataIntegrityError(f"{records_path}:{line_no}: subset/language mapping differs from the allowlist")
            upstream_id = row.get("upstream_document_id")
            expected_document_id = hashlib.sha256(
                f"{source['id']}\0{source['revision']}\0{row.get('subset')}\0{upstream_id}".encode("utf-8")
            ).hexdigest() if isinstance(upstream_id, str) and upstream_id else None
            if row.get("document_id") != expected_document_id:
                raise DataIntegrityError(f"{records_path}:{line_no}: stable document ID does not match source identity")
            if type(row.get("upstream_row_index")) is not int or row["upstream_row_index"] < 0:
                raise DataIntegrityError(f"{records_path}:{line_no}: upstream row index is missing or invalid")
            if row.get("source_record") != (
                f"{source['id']}@{source['revision']}/{row.get('upstream_file')}#{row['upstream_row_index']}"
            ):
                raise DataIntegrityError(f"{records_path}:{line_no}: source record provenance is inconsistent")
            raw_sha = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            if row.get("raw_sha256") != raw_sha:
                raise DataIntegrityError(f"{records_path}:{line_no}: raw text SHA-256 mismatch")
            if not row.get("upstream_file_sha256") or not row.get("url_or_archive_id"):
                raise DataIntegrityError(f"{records_path}:{line_no}: upstream file hash and URL/archive provenance are required")
            row["_input_record"] = f"{records_path}:{line_no}"
            records_seen += 1
            yield row
    if records_seen != raw_manifest.get("records"):
        raise DataIntegrityError(f"{records_path}: row count differs from raw source manifest")


def _iter_shingles(text: str, shingle_words: int) -> Iterator[str]:
    words = (match.group(0).casefold() for match in re.finditer(r"[\w]+", text))
    window: deque[str] = deque(maxlen=shingle_words)
    initial: list[str] = []
    for word in words:
        window.append(word)
        initial.append(word)
        if len(initial) == shingle_words:
            break
    if not initial:
        yield text.casefold()
        return
    if len(initial) < shingle_words:
        yield " ".join(initial)
        return
    yield " ".join(window)
    for word in words:
        window.append(word)
        yield " ".join(window)


class DiskBackedDeduplicator:
    """SQLite exact and banded-MinHash index with fixed-size signatures, not corpus-wide shingles."""

    def __init__(self, path: Path, *, threshold: float, num_perm: int, bands: int = 16,
                 shingle_words: int = 5) -> None:
        if bands <= 0 or num_perm <= 0 or shingle_words <= 0 or num_perm % bands:
            raise DataIntegrityError("near_dedup_num_perm must be divisible by the configured LSH band count")
        self.threshold = threshold
        self.num_perm = num_perm
        self.bands = bands
        self.rows_per_band = num_perm // bands
        self.shingle_words = shingle_words
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE documents (clean_sha TEXT PRIMARY KEY, source_record TEXT NOT NULL,
                provenance_json TEXT NOT NULL, signature BLOB NOT NULL);
            CREATE TABLE bands (bucket TEXT NOT NULL, clean_sha TEXT NOT NULL,
                PRIMARY KEY(bucket, clean_sha));
            CREATE INDEX bands_by_digest ON bands(clean_sha);
        """)

    def classify(self, clean_sha: str, text: str, provenance: dict[str, Any]
                 ) -> tuple[str, str | None, float | None, dict[str, Any] | None]:
        exact = self.db.execute("SELECT source_record, provenance_json FROM documents WHERE clean_sha=?",
                                (clean_sha,)).fetchone()
        if exact:
            return "exact", str(exact[0]), 1.0, json.loads(exact[1])
        signature = MinHash(num_perm=self.num_perm, seed=1)
        for shingle in _iter_shingles(text, self.shingle_words):
            signature.update(shingle.encode("utf-8"))
        signature_blob = signature.hashvalues.astype("<u8", copy=False).tobytes()
        bucket_ids = []
        values = signature.hashvalues
        for band in range(self.bands):
            start = band * self.rows_per_band
            raw = values[start:start + self.rows_per_band].astype("<u8", copy=False).tobytes()
            bucket_ids.append(hashlib.sha256(raw).hexdigest())
        placeholders = ",".join("?" for _ in bucket_ids)
        candidate_cursor = self.db.execute(
            f"SELECT DISTINCT clean_sha FROM bands WHERE bucket IN ({placeholders}) ORDER BY clean_sha", bucket_ids
        )
        for (candidate,) in candidate_cursor:
            candidate_signature = self.db.execute(
                "SELECT signature FROM documents WHERE clean_sha=?", (candidate,)
            ).fetchone()[0]
            other_values = struct.unpack("<" + "Q" * self.num_perm, candidate_signature)
            similarity = sum(int(left) == right for left, right in zip(signature.hashvalues, other_values)) / self.num_perm
            if similarity >= self.threshold:
                row = self.db.execute("SELECT source_record, provenance_json FROM documents WHERE clean_sha=?",
                                      (candidate,)).fetchone()
                return "near", str(row[0]), similarity, json.loads(row[1])
        self.db.execute("INSERT INTO documents VALUES(?,?,?,?)",
                        (clean_sha, str(provenance["source_record"]),
                         json.dumps(provenance, ensure_ascii=False, sort_keys=True), signature_blob))
        self.db.executemany("INSERT INTO bands VALUES(?,?)", ((bucket, clean_sha) for bucket in bucket_ids))
        self.db.commit()
        return "kept", None, None, None

    def close(self) -> None:
        self.db.close()


def _split_for(row: dict[str, Any], split_salt: str, fractions: dict[str, float]) -> str:
    unit = row.get("work_id") or row.get("upstream_document_id") or row.get("document_id")
    if not isinstance(unit, str) or not unit:
        raise DataIntegrityError(f"{row.get('_input_record', '<record>')}: deterministic source work/document ID is required")
    bucket = int(hashlib.sha256(f"{split_salt}:{unit}".encode()).hexdigest()[:16], 16) / 0xFFFFFFFFFFFFFFFF
    if bucket < fractions["validation"]:
        return "validation"
    if bucket < fractions["validation"] + fractions["blind"]:
        return "blind"
    return "train"


def prepare_local_corpus(inputs: Iterable[str | Path], output_dir: str | Path, *, split_salt: str | None = None,
                         validation_fraction: float | None = None, blind_fraction: float | None = None,
                         normalization: str | None = None, config_path: str | Path = CONFIG_PATH,
                         admission: str = "audit") -> dict[str, Any]:
    """Prepare pinned raw source snapshots; splits and deduplication happen before token chunking."""
    if admission not in {"audit", "production"}:
        raise DataIntegrityError("admission must be audit or production")
    config = load_config(config_path)
    data_config = config["data"]
    cleaning = data_config["cleaning"]
    split_salt = split_salt or data_config["split_salt"]
    fractions = {
        "validation": float(data_config["splits"]["validation"] if validation_fraction is None else validation_fraction),
        "blind": float(data_config["splits"]["blind"] if blind_fraction is None else blind_fraction),
    }
    if not split_salt or not 0 < fractions["validation"] < 1 or not 0 < fractions["blind"] < 1 \
            or fractions["validation"] + fractions["blind"] >= 1:
        raise DataIntegrityError("invalid frozen document split policy")
    if cleaning.get("unicode_normalization") != "NFC":
        raise DataIntegrityError("corpus cleaner and tokenizer contract require NFC normalization")
    normalization = normalization or cleaning["unicode_normalization"]
    if normalization != cleaning["unicode_normalization"]:
        raise DataIntegrityError("normalization override differs from the single YAML corpus/tokenizer contract")
    sources_by_id = {source["id"]: source for source in data_config["sources"]}
    source_snapshots: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    production_admitted = True
    for value in inputs:
        path = Path(value)
        _, raw_manifest_path = _record_files(path)
        raw_manifest = _load_raw_manifest(raw_manifest_path)
        source = sources_by_id.get(str(raw_manifest.get("source_id")))
        if source is None or source.get("revision") != raw_manifest.get("revision"):
            raise DataIntegrityError(f"{raw_manifest_path}: source revision is outside the configured allowlist")
        if raw_manifest.get("purpose") != "pretraining-candidate-source-audit":
            raise DataIntegrityError(f"{raw_manifest_path}: input purpose is not a pretraining source snapshot")
        if raw_manifest.get("bounded_slice") and admission == "production":
            raise DataIntegrityError(f"{path}: bounded source slices cannot be production-admitted")
        if raw_manifest.get("license_claim") != source.get("license_claim") \
                or raw_manifest.get("rights_review_status") != source.get("rights_review_status") \
                or raw_manifest.get("provenance_review_status") != source.get("provenance_review_status"):
            raise DataIntegrityError(f"{raw_manifest_path}: source license/review metadata differs from the YAML allowlist")
        if source.get("rights_review_status") != "approved" or source.get("provenance_review_status") != "approved" \
                or source.get("status") != "approved" or raw_manifest.get("admission") != "production":
            production_admitted = False
        source_snapshots.append((path, source, raw_manifest))
    if not source_snapshots:
        raise DataIntegrityError("at least one pinned source snapshot is required")
    source_snapshots.sort(key=lambda item: (item[1]["id"], item[1]["revision"]))
    if admission == "production" and not production_admitted:
        raise DataIntegrityError("production corpus admission is blocked by source rights/provenance review")

    threshold = float(cleaning["near_dedup_threshold"])
    num_perm = int(cleaning["near_dedup_num_perm"])
    output = Path(output_dir)
    if output.exists():
        raise DataIntegrityError(f"refusing to overwrite existing corpus output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.corpus-partial-", dir=output.parent))
    handles = {name: (stage / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n")
               for name in ("train", "validation", "blind")}
    quarantine = (stage / "quarantine.jsonl").open("w", encoding="utf-8", newline="\n")
    duplicates = (stage / "duplicates.jsonl").open("w", encoding="utf-8", newline="\n")
    deduper = DiskBackedDeduplicator(stage / "dedup.sqlite3", threshold=threshold, num_perm=num_perm,
                                     bands=int(cleaning.get("near_dedup_bands", 16)),
                                     shingle_words=int(cleaning["near_dedup_shingle_words"]))
    counts = {key: 0 for key in ("seen", "kept", "exact_duplicates", "near_duplicates", "quarantined",
                                "train", "validation", "blind")}
    language_counts = {split: {language: 0 for language in cleaning["filter_languages"]}
                       for split in ("train", "validation", "blind")}
    source_provenance = []
    try:
        for path, source, raw_manifest in source_snapshots:
            source_provenance.append({"source_id": source["id"], "revision": source["revision"],
                                      "license_claim": source["license_claim"],
                                      "raw_manifest_sha256": raw_manifest["manifest_sha256"],
                                      "rights_review_status": raw_manifest.get("rights_review_status"),
                                      "provenance_review_status": raw_manifest.get("provenance_review_status")})
            for row in _records(path, source, raw_manifest):
                counts["seen"] += 1
                raw_text = row["text"]
                if row.get("language", "").casefold() not in cleaning["filter_languages"]:
                    raise DataIntegrityError(f"{row['_input_record']}: language outside configured EN/RU set")
                before = numeric_spans(raw_text)
                source_cleaned = _clean_source_markup(raw_text, source)
                text = clean_text(source_cleaned, normalization=normalization)
                after = numeric_spans(text)
                if before != after:
                    provenance = {key: value for key, value in row.items() if not key.startswith("_") and key != "text"}
                    quarantine.write(json.dumps({**provenance,
                        "reason": "source_markup_cleanup_changes_numeric_spans",
                        "numeric_spans_before": before, "numeric_spans_after": after},
                        ensure_ascii=False, sort_keys=True) + "\n")
                    counts["quarantined"] += 1
                    continue
                if not text.strip():
                    raise DataIntegrityError(f"{row['_input_record']}: empty document after source-aware cleaning")
                reason = _quarantine_reason(text, cleaning, source)
                provenance = {key: value for key, value in row.items() if not key.startswith("_") and key != "text"}
                provenance.update({"raw_sha256": row["raw_sha256"],
                                   "clean_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                   "transform_version": "numeric-integrity.v4-source-aware-nfc",
                                   "numeric_spans": after})
                if reason:
                    quarantine.write(json.dumps({**provenance, "reason": reason}, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["quarantined"] += 1
                    continue
                split = _split_for(row, split_salt, data_config["splits"])
                provenance["split"] = split
                kind, duplicate_of, similarity, duplicate_provenance = deduper.classify(
                    provenance["clean_sha256"], text, provenance)
                if kind != "kept":
                    duplicate_key = "exact_duplicates" if kind == "exact" else "near_duplicates"
                    counts[duplicate_key] += 1
                    duplicates.write(json.dumps({**provenance, "duplicate_type": kind,
                        "duplicate_of_source_record": duplicate_of, "minhash_similarity_estimate": similarity,
                        "duplicate_of_provenance": duplicate_provenance,
                        "duplicate_text_sha256": provenance["clean_sha256"]}, ensure_ascii=False, sort_keys=True) + "\n")
                    continue
                record = {**provenance, "text": text}
                handles[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                counts["kept"] += 1
                counts[split] += 1
                language_counts[split][row["language"].casefold()] += 1
        empty_splits = [split for split, count in counts.items()
                        if split in {"train", "validation", "blind"} and count == 0]
        if admission == "production" and empty_splits:
            raise DataIntegrityError(f"corpus is too small or over-filtered; empty required splits: {empty_splits}")
        if admission == "production":
            missing_languages = {split: [language for language, count in values.items() if count == 0]
                                 for split, values in language_counts.items()}
            missing_languages = {split: languages for split, languages in missing_languages.items() if languages}
            if missing_languages:
                raise DataIntegrityError(f"production corpus requires EN and RU in every split: {missing_languages}")
    except Exception:
        for handle in handles.values():
            handle.close()
        quarantine.close()
        duplicates.close()
        deduper.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise
    for handle in handles.values():
        handle.close()
    quarantine.close()
    duplicates.close()
    deduper.close()
    for suffix in ("", "-wal", "-shm"):
        (stage / f"dedup.sqlite3{suffix}").unlink(missing_ok=True)
    split_hashes = {f"{name}.jsonl": sha256_file(stage / f"{name}.jsonl")
                    for name in handles}
    files = dict(split_hashes)
    files["duplicates.jsonl"] = sha256_file(stage / "duplicates.jsonl")
    files["quarantine.jsonl"] = sha256_file(stage / "quarantine.jsonl")
    for name in ("train", "validation", "blind"):
        (stage / name).mkdir()
    admission_status = "production" if admission == "production" and production_admitted else "audit_only"
    manifest = {
        "schema": "nullxes.shinra.corpus.v2",
        "purpose": "pretraining",
        "admission": admission_status,
        "source_revisions": source_provenance,
        "split_policy": {"unit": "source_work_or_document_before_chunking", "salt_sha256": hashlib.sha256(split_salt.encode()).hexdigest(),
                         "fractions": data_config["splits"]},
        "holdout_contamination_check": {"deduplicate_before_split": True,
                                         "near_duplicate_metric": cleaning["near_dedup_metric"],
                                         "split_unit": "source_work_or_document"},
        "cleaning": {"transform_version": "numeric-integrity.v4-source-aware-nfc",
                     "unicode_normalization": cleaning["unicode_normalization"],
                     "numeric_integrity": "ordered numeric spans must match exactly"},
        "deduplication": {"backend": "sqlite_disk_backed_banded_minhash",
                          "similarity_metric": cleaning["near_dedup_metric"],
                          "threshold": threshold,
                          "num_perm": num_perm, "bands": int(cleaning.get("near_dedup_bands", 16)),
                          "shingle_words": int(cleaning["near_dedup_shingle_words"])},
        "counts": counts,
        "language_counts": language_counts,
        "files": files,
        "split_hashes": {key.removesuffix(".jsonl"): value for key, value in split_hashes.items()},
        "train_hash": split_hashes["train.jsonl"],
        "validation_hash": split_hashes["validation.jsonl"],
        "blind_hash": split_hashes["blind.jsonl"],
    }
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    temporary = stage / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, stage / "manifest.json")
    os.replace(stage, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, split and audit pinned SHINRA corpus sources")
    parser.add_argument("--input", nargs="+", required=True, help="source snapshot directories from data.sources")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--admission", choices=("audit", "production"), default="audit")
    args = parser.parse_args()
    result = prepare_local_corpus(args.input, args.output_dir, config_path=args.config, admission=args.admission)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
