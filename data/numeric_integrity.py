"""Auditable cleaning, disk-backed deduplication and document-level split assignment."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import shutil
import sqlite3
import struct
from collections import deque
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
HTML_BLOCK = re.compile(r"<(script|style|noscript)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
TABLE_ROW = re.compile(r"^\s*(?:\|.*\||\+[-=+]{3,}\+)\s*$")
CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LINE = re.compile(r"[^\r\n]+")
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)
URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
TECHNICAL_SIGNAL = re.compile(
    r"(?i)(?:```|=>|->|:=|\b(?:api|algorithm|arxiv|code|compiler|dataset|doi|equation|function|gradient|"
    r"integral|lemma|matrix|neural network|protocol|python|quantum|tensor|theorem)\b|"
    r"\\(?:begin|frac|int|sum)|^\s*(?:def|class|import|from|select|create table)\b)", re.MULTILINE
)
ENGLISH_FUNCTION_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "for", "from", "has",
    "have", "he", "in", "into", "is", "it", "its", "may", "not", "of", "on", "or", "our", "that",
    "the", "their", "this", "to", "was", "we", "were", "which", "with", "you", "your",
})
CLEANING_PIPELINE_VERSION = "numeric-integrity.v5-fast-quality"


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
    if source.get("cleaning_profile") == "web_crawl_text":
        cleaned = html.unescape(text)
        cleaned = HTML_BLOCK.sub(" ", cleaned)
        return HTML_TAG.sub(" ", cleaned)
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


def _cheap_quality_reason(text: str, quality: dict[str, Any], source: dict[str, Any]
                          ) -> tuple[str | None, dict[str, int | float | bool]]:
    """Bounded-memory content and language checks before numeric scans and MinHash."""
    sample_words: list[str] = []
    unique_words: set[str] = set()
    english_function_words = 0
    for match in WORD.finditer(text):
        word = match.group(0).casefold()
        if len(sample_words) < int(quality["word_sample_limit"]):
            sample_words.append(word)
            unique_words.add(word)
            english_function_words += int(word in ENGLISH_FUNCTION_WORDS)
        else:
            break
    letter_count = sum(1 for _ in LETTER.finditer(text))
    text_length = len(text)
    letter_fraction = letter_count / max(text_length, 1)
    url_characters = sum(match.end() - match.start() for match in URL.finditer(text))
    url_fraction = url_characters / max(text_length, 1)
    technical_signal = bool(TECHNICAL_SIGNAL.search(text))
    unique_fraction = len(unique_words) / max(len(sample_words), 1)
    metrics: dict[str, int | float | bool] = {
        "characters": text_length,
        "letters": letter_count,
        "letter_fraction": letter_fraction,
        "word_sample_count": len(sample_words),
        "unique_word_fraction": unique_fraction,
        "url_character_fraction": url_fraction,
        "technical_signal": technical_signal,
    }
    if text_length < int(quality["minimum_characters"]) or letter_count < int(quality["minimum_letters"]):
        if text_length < int(quality["technical_minimum_characters"]) \
                or letter_count < int(quality["technical_minimum_letters"]) or not technical_signal:
            return "quality_too_short_or_low_letter_count", metrics
    if letter_fraction < float(quality["minimum_letter_fraction"]) and not technical_signal:
        return "quality_low_letter_fraction", metrics
    if len(sample_words) < int(quality["minimum_word_sample"]) and not technical_signal:
        return "quality_too_few_words", metrics
    if len(sample_words) >= int(quality["uniqueness_minimum_sample"]) \
            and unique_fraction < float(quality["minimum_unique_word_fraction"]):
        return "quality_repetitive_word_sample", metrics
    if url_fraction > float(quality["maximum_url_character_fraction"]):
        return "quality_url_or_link_farm", metrics
    repeated_limit = int(quality["maximum_repeated_character_run"])
    repeated_match = re.search(rf"([^\s])\1{{{repeated_limit - 1},}}", text)
    if repeated_match and not technical_signal:
        metrics["longest_repeated_character_run_exceeds_limit"] = True
        return "quality_repeated_character_run", metrics
    if source.get("language_validation_profile") == "english_source_latin_script_guard" \
            and len(sample_words) >= int(quality["dclm_language_minimum_words"]) and not technical_signal:
        english_fraction = english_function_words / len(sample_words)
        metrics["english_function_word_fraction"] = english_fraction
        if english_fraction < float(quality["dclm_minimum_english_function_word_fraction"]):
            return "language_english_function_word_guard", metrics
    return None, metrics


def _quarantine_reason(text: str, cleaning: dict[str, Any], source: dict[str, Any]) -> str | None:
    table_rows = 0
    short_line_counts: dict[str, int] = {}
    repeated = 0
    track_repeated_lines = source.get("quarantine_profile") in {"ocr_book", "web_crawl"}
    for match in LINE.finditer(text):
        line = match.group(0)
        if TABLE_ROW.fullmatch(line):
            table_rows += 1
        if track_repeated_lines:
            stripped = line.strip()
            if 0 < len(stripped) <= 120:
                count = short_line_counts.get(stripped, 0) + 1
                short_line_counts[stripped] = count
                repeated = max(repeated, count)
    if cleaning.get("quarantine_table_documents", True) and table_rows >= int(cleaning.get("table_rows_quarantine_threshold", 2)):
        return f"table_structure_without_reliable_context:{table_rows}_rows"
    if source.get("quarantine_profile") == "ocr_book" and "\ufffd" in text:
        return "ocr_or_decode_replacement_character"
    controls = sum(1 for _ in CONTROL_CHARACTER.finditer(text))
    if text and controls / len(text) > float(cleaning.get("max_control_character_ratio", 0.001)):
        return f"excessive_control_characters:{controls}"
    if source.get("quarantine_profile") == "ocr_book" \
            and repeated >= int(cleaning.get("repeated_short_line_quarantine_threshold", 12)):
        return f"repeated_page_header_or_ocr_line:{repeated}"
    if source.get("quarantine_profile") == "web_crawl" and repeated >= int(
            cleaning.get("repeated_short_line_quarantine_threshold", 12)):
        return f"repeated_web_boilerplate_line:{repeated}"
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
    if manifest.get("schema") not in {"nullxes.shinra.raw-source.v2", "nullxes.shinra.raw-source.v3"}:
        raise DataIntegrityError(f"{path}: unsupported raw source manifest schema")
    accepted, scanned, rejected = (manifest.get(key) for key in ("accepted", "scanned", "rejected"))
    if any(type(value) is not int or value < 0 for value in (accepted, scanned, rejected)) \
            or accepted != manifest.get("records") or scanned != accepted + rejected:
        raise DataIntegrityError(f"{path}: accepted/scanned/rejected counters are inconsistent")
    bounded = manifest.get("bounded_slice")
    maximum = manifest.get("max_accepted_documents")
    target_maximum = manifest.get("requested_max_target_tokens")
    if type(bounded) is not bool or manifest.get("admission") not in {"audit_only", "production"} \
            or (maximum is not None and (type(maximum) is not int or maximum <= 0 or accepted > maximum)) \
            or (target_maximum is not None and (type(target_maximum) is not int or target_maximum <= 0
                or type(manifest.get("acquired_supervised_target_tokens")) is not int
                or not manifest.get("tokenizer_bundle_sha256")
                or manifest.get("admission") != "audit_only"
                or manifest.get("target_limit_reached") is not (
                    int(manifest.get("acquired_supervised_target_tokens", 0)) >= target_maximum))) \
            or (bounded != (maximum is not None or target_maximum is not None)) \
            or (bounded and manifest.get("admission") != "audit_only"):
        raise DataIntegrityError(f"{path}: bounded acquisition/admission metadata is inconsistent")
    rejection_name = manifest.get("rejections_file")
    if rejection_name != "rejections.jsonl":
        raise DataIntegrityError(f"{path}: unsupported or unsafe rejection-ledger path")
    rejection_path = path.parent / rejection_name
    if not rejection_path.is_file() or sha256_file(rejection_path) != manifest.get("rejections_sha256"):
        raise DataIntegrityError(f"{rejection_path}: rejection-ledger SHA-256 mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise DataIntegrityError(f"{path}: upstream source file provenance is missing")
    valid_files = [item for item in files if isinstance(item, dict)
                   and isinstance(item.get("path"), str) and item.get("path")
                   and isinstance(item.get("sha256"), str)
                   and re.fullmatch(r"[a-f0-9]{64}", item["sha256"])]
    upstream_paths = {item["path"] for item in valid_files}
    if len(valid_files) != len(files) or len(upstream_paths) != len(files):
        raise DataIntegrityError(f"{path}: malformed or duplicate upstream file provenance")
    reason_counts: dict[str, int] = {}
    ledger_count = 0
    with rejection_path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                rejection = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataIntegrityError(f"{rejection_path}:{line_no}: malformed rejection JSON: {exc}") from exc
            if not isinstance(rejection, dict) or "text" in rejection:
                raise DataIntegrityError(f"{rejection_path}:{line_no}: invalid or non-redacted rejection record")
            if rejection.get("source_id") != manifest.get("source_id") \
                    or rejection.get("revision") != manifest.get("revision") \
                    or rejection.get("upstream_file") not in upstream_paths \
                    or type(rejection.get("upstream_row_index")) is not int \
                    or rejection["upstream_row_index"] < 0 \
                    or not isinstance(rejection.get("reason"), str) or not rejection["reason"] \
                    or not isinstance(rejection.get("available_identifiers"), dict):
                raise DataIntegrityError(f"{rejection_path}:{line_no}: incomplete rejection provenance")
            reason_counts[rejection["reason"]] = reason_counts.get(rejection["reason"], 0) + 1
            ledger_count += 1
    if ledger_count != rejected or reason_counts != manifest.get("rejected_by_reason"):
        raise DataIntegrityError(f"{rejection_path}: ledger rows or reason counts differ from the raw manifest")
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
            language = row["language"]
            if source.get("language_field") and row.get("upstream_language_label") != source.get(
                    "language_labels", {}).get(language):
                raise DataIntegrityError(f"{records_path}:{line_no}: upstream language label differs from the allowlist")
            if source.get("script_field") and row.get("upstream_script_label") != source.get(
                    "script_labels", {}).get(language):
                raise DataIntegrityError(f"{records_path}:{line_no}: upstream script label differs from the allowlist")
            declared_language_profile = row.get("language_validation_profile")
            if declared_language_profile is None and raw_manifest.get("schema") == "nullxes.shinra.raw-source.v2":
                declared_language_profile = source.get("language_validation_profile")
            if declared_language_profile != source.get("language_validation_profile"):
                raise DataIntegrityError(f"{records_path}:{line_no}: language validation profile differs from the allowlist")
            if source.get("language_validation_profile") == "english_source_latin_script_guard":
                fraction = row.get("latin_script_fraction")
                if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) \
                        or not math.isfinite(float(fraction)) \
                        or float(fraction) < float(source["minimum_latin_script_fraction"]):
                    raise DataIntegrityError(f"{records_path}:{line_no}: English source script guard was not met")
            minimum_score = source.get("minimum_quality_score")
            if minimum_score is not None:
                score = row.get("quality_score")
                if isinstance(score, bool) or not isinstance(score, (int, float)) \
                        or not math.isfinite(float(score)) or float(score) < float(minimum_score):
                    raise DataIntegrityError(f"{records_path}:{line_no}: upstream quality score fails its pinned threshold")
            if row.get("quality_method") != source.get("quality_method"):
                raise DataIntegrityError(f"{records_path}:{line_no}: quality method differs from the source allowlist")
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
            row["_source_line_number"] = line_no
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
                 shingle_words: int = 5, sqlite_cache_mib: int = 256, sqlite_mmap_mib: int = 1024,
                 wal_autocheckpoint_pages: int = 131072) -> None:
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
        self.db.execute(f"PRAGMA cache_size=-{int(sqlite_cache_mib) * 1024}")
        self.db.execute(f"PRAGMA mmap_size={int(sqlite_mmap_mib) * 1024 * 1024}")
        self.db.execute(f"PRAGMA wal_autocheckpoint={int(wal_autocheckpoint_pages)}")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (clean_sha TEXT PRIMARY KEY, source_record TEXT NOT NULL,
                provenance_json TEXT NOT NULL, signature BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS bands (bucket TEXT NOT NULL, clean_sha TEXT NOT NULL,
                PRIMARY KEY(bucket, clean_sha));
            CREATE INDEX IF NOT EXISTS bands_by_digest ON bands(clean_sha);
            CREATE TABLE IF NOT EXISTS pipeline_state (id INTEGER PRIMARY KEY CHECK(id=1), state_json TEXT NOT NULL);
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
            f"SELECT DISTINCT d.clean_sha, d.signature, d.source_record, d.provenance_json "
            f"FROM bands AS b JOIN documents AS d USING(clean_sha) "
            f"WHERE b.bucket IN ({placeholders}) ORDER BY d.clean_sha", bucket_ids
        )
        for candidate, candidate_signature, source_record, provenance_json in candidate_cursor:
            other_values = struct.unpack("<" + "Q" * self.num_perm, candidate_signature)
            similarity = sum(int(left) == right for left, right in zip(signature.hashvalues, other_values)) / self.num_perm
            if similarity >= self.threshold:
                return "near", str(source_record), similarity, json.loads(provenance_json)
        self.db.execute("INSERT INTO documents VALUES(?,?,?,?)",
                        (clean_sha, str(provenance["source_record"]),
                         json.dumps(provenance, ensure_ascii=False, sort_keys=True), signature_blob))
        self.db.executemany("INSERT INTO bands VALUES(?,?)", ((bucket, clean_sha) for bucket in bucket_ids))
        return "kept", None, None, None

    def load_pipeline_state(self) -> dict[str, Any] | None:
        row = self.db.execute("SELECT state_json FROM pipeline_state WHERE id=1").fetchone()
        return json.loads(row[0]) if row else None

    def save_pipeline_state(self, state: dict[str, Any]) -> None:
        self.db.execute("INSERT INTO pipeline_state(id,state_json) VALUES(1,?) "
                        "ON CONFLICT(id) DO UPDATE SET state_json=excluded.state_json",
                        (json.dumps(state, ensure_ascii=False, sort_keys=True),))
        self.db.commit()

    def close(self) -> None:
        self.db.close()


def _verify_resume_state_read_only(path: Path, expected_identity_sha256: str) -> None:
    if not path.is_file():
        raise DataIntegrityError(f"cannot resume cleaning without its deduplication state: {path}")
    connection = None
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        row = connection.execute("SELECT state_json FROM pipeline_state WHERE id=1").fetchone()
    except sqlite3.Error as exc:
        raise DataIntegrityError(f"cannot inspect cleaning resume state read-only: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    if row is None:
        raise DataIntegrityError("partial cleaning has no committed SQLite resume checkpoint")
    try:
        state = json.loads(row[0])
    except (TypeError, json.JSONDecodeError) as exc:
        raise DataIntegrityError("partial cleaning SQLite resume checkpoint is malformed") from exc
    if state.get("identity_sha256") != expected_identity_sha256:
        raise DataIntegrityError(
            "partial cleaning was produced with different cleaning/quality semantics; "
            "preserve it and use a new output directory"
        )


def _write_jsonl(handle, record: dict[str, Any]) -> int:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    written = handle.write(payload)
    if written != len(payload):
        raise DataIntegrityError("short write while staging a clean-corpus JSONL record")
    return written


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
                         admission: str = "audit", resume: bool = False,
                         minimum_free_gb: float | None = None) -> dict[str, Any]:
    """Prepare pinned raw source snapshots; splits and deduplication happen before token chunking."""
    if admission not in {"audit", "production"}:
        raise DataIntegrityError("admission must be audit or production")
    config = load_config(config_path)
    reserve_gb = (float(config["runtime"]["minimum_free_disk_gb"])
                  if minimum_free_gb is None else float(minimum_free_gb))
    if reserve_gb < 0:
        raise DataIntegrityError("minimum_free_gb cannot be negative")
    data_config = config["data"]
    cleaning = data_config["cleaning"]
    quality = data_config["quality"]
    performance = data_config["performance"]
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
        if raw_manifest.get("stream_format", "parquet") != source.get("format", "parquet"):
            raise DataIntegrityError(f"{raw_manifest_path}: stream format differs from the pinned source adapter")
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
    stage = output.parent / f".{output.name}.corpus-partial"
    snapshot_identity = [{"path": str(path.resolve()), "manifest_sha256": manifest["manifest_sha256"]}
                         for path, _, manifest in source_snapshots]
    semantic_contract = {
        "pipeline_version": CLEANING_PIPELINE_VERSION,
        "source_allowlist": data_config["sources"],
        "split_salt": split_salt,
        "fractions": fractions,
        "normalization": normalization,
        "admission": admission,
        "cleaning": cleaning,
        "quality": quality,
        "deduplication": {
            "threshold": threshold,
            "num_perm": num_perm,
            "bands": int(cleaning.get("near_dedup_bands", 16)),
            "shingle_words": int(cleaning["near_dedup_shingle_words"]),
        },
    }
    identity = {"inputs": snapshot_identity,
                "semantic_contract_sha256": hashlib.sha256(
                    json.dumps(semantic_contract, sort_keys=True).encode()).hexdigest()}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if stage.exists() and not resume:
        raise DataIntegrityError(f"partial corpus preparation exists; pass --resume: {stage}")
    if resume and not stage.is_dir():
        raise DataIntegrityError(f"no partial corpus preparation exists to resume: {stage}")
    if not stage.exists():
        if resume:
            raise DataIntegrityError(f"no partial corpus preparation exists to resume: {stage}")
        stage.mkdir()
    if resume:
        _verify_resume_state_read_only(stage / "dedup.sqlite3", identity_sha)
    deduper = DiskBackedDeduplicator(stage / "dedup.sqlite3", threshold=threshold, num_perm=num_perm,
                                     bands=int(cleaning.get("near_dedup_bands", 16)),
                                     shingle_words=int(cleaning["near_dedup_shingle_words"]),
                                     sqlite_cache_mib=int(performance["sqlite_cache_mib"]),
                                     sqlite_mmap_mib=int(performance["sqlite_mmap_mib"]),
                                     wal_autocheckpoint_pages=int(performance["sqlite_wal_autocheckpoint_pages"]))
    checkpoint = deduper.load_pipeline_state() if resume else None
    if checkpoint and checkpoint.get("identity_sha256") != identity_sha:
        deduper.close()
        raise DataIntegrityError("corpus resume sources, split, cleaning config, dedup policy or admission differ")
    if resume and not checkpoint:
        deduper.close()
        raise DataIntegrityError("deduplication database has no committed corpus resume state")
    counts = checkpoint["counts"] if checkpoint else {
        key: 0 for key in ("seen", "kept", "exact_duplicates", "near_duplicates", "quarantined",
                           "quality_rejected",
                           "train", "validation", "blind")}
    language_counts = checkpoint["language_counts"] if checkpoint else {
        split: {language: 0 for language in cleaning["filter_languages"]}
        for split in ("train", "validation", "blind")}
    source_provenance = checkpoint["source_provenance"] if checkpoint else []
    cursor = checkpoint.get("cursor") if checkpoint else None
    output_buffer_bytes = int(performance["output_buffer_bytes"])
    handles = {name: (stage / f"{name}.jsonl").open("ab" if resume else "wb", buffering=output_buffer_bytes)
               for name in ("train", "validation", "blind")}
    quarantine = (stage / "quarantine.jsonl").open("ab" if resume else "wb", buffering=output_buffer_bytes)
    duplicates = (stage / "duplicates.jsonl").open("ab" if resume else "wb", buffering=output_buffer_bytes)
    if resume:
        offsets = checkpoint["output_offsets"]
        for name, handle in {**handles, "quarantine": quarantine, "duplicates": duplicates}.items():
            offset = int(offsets[name])
            handle.flush()
            if handle.tell() < offset:
                raise DataIntegrityError(f"corpus resume journal points beyond output file {name}")
            handle.truncate(offset)
            handle.seek(0, os.SEEK_END)
    pending_rows = 0
    pending_output_bytes = 0
    processed_since_disk_check = 0

    def persist_corpus_state(committed_cursor: dict[str, int] | None) -> None:
        nonlocal pending_output_bytes
        for handle in (*handles.values(), quarantine, duplicates):
            handle.flush()
            os.fsync(handle.fileno())
        state = {"schema": "nullxes.shinra.corpus-build-state.v2",
                 "pipeline_version": CLEANING_PIPELINE_VERSION, "identity_sha256": identity_sha,
                 "cursor": committed_cursor, "counts": counts, "language_counts": language_counts,
                 "source_provenance": source_provenance,
                 "output_offsets": {name: handle.tell() for name, handle in
                                    {**handles, "quarantine": quarantine, "duplicates": duplicates}.items()}}
        deduper.save_pipeline_state(state)
        pending_output_bytes = 0

    def write_output(handle, record: dict[str, Any]) -> None:
        nonlocal pending_output_bytes
        pending_output_bytes += _write_jsonl(handle, record)

    def require_disk_headroom() -> None:
        if not reserve_gb:
            return
        available_gb = shutil.disk_usage(stage).free / (1024 ** 3)
        if available_gb < reserve_gb:
            raise DataIntegrityError(
                f"pausing resumable corpus cleaning: free={available_gb:.2f} GiB, "
                f"configured minimum={reserve_gb:.2f} GiB"
            )

    def mark_processed(current_cursor: dict[str, int]) -> None:
        nonlocal cursor, pending_rows, processed_since_disk_check
        pending_rows += 1
        processed_since_disk_check += 1
        needs_checkpoint = (pending_rows >= int(performance["clean_checkpoint_records"])
                            or pending_output_bytes >= int(performance["clean_checkpoint_bytes"]))
        needs_disk_check = (processed_since_disk_check >= int(performance["disk_check_records"])
                            or needs_checkpoint)
        if needs_disk_check and reserve_gb:
            require_disk_headroom()
            processed_since_disk_check = 0
        if needs_checkpoint:
            persist_corpus_state(current_cursor)
            cursor = current_cursor
            pending_rows = 0

    if not resume:
        persist_corpus_state(None)
    try:
        for source_index, (path, source, raw_manifest) in enumerate(source_snapshots):
            provenance_record = {"source_id": source["id"], "revision": source["revision"],
                                      "license_claim": source["license_claim"],
                                      "raw_manifest_sha256": raw_manifest["manifest_sha256"],
                                      "accepted": raw_manifest["accepted"],
                                      "scanned": raw_manifest["scanned"],
                                      "rejected": raw_manifest["rejected"],
                                      "rejected_by_reason": raw_manifest["rejected_by_reason"],
                                      "rejections_sha256": raw_manifest["rejections_sha256"],
                                      "stream_format": raw_manifest.get("stream_format", "parquet"),
                                      "candidate_tokenizer_bundle_sha256": raw_manifest.get("tokenizer_bundle_sha256"),
                                      "acquired_supervised_target_tokens": raw_manifest.get(
                                          "acquired_supervised_target_tokens"),
                                      "rights_review_status": raw_manifest.get("rights_review_status"),
                                      "provenance_review_status": raw_manifest.get("provenance_review_status")}
            if not any(item.get("raw_manifest_sha256") == raw_manifest["manifest_sha256"]
                       for item in source_provenance):
                source_provenance.append(provenance_record)
            for row in _records(path, source, raw_manifest):
                current_cursor = {"source_index": source_index, "line_number": row["_source_line_number"]}
                if cursor is not None and (source_index, current_cursor["line_number"]) <= (
                        int(cursor["source_index"]), int(cursor["line_number"])):
                    continue
                counts["seen"] += 1
                raw_text = row["text"]
                if row.get("language", "").casefold() not in cleaning["filter_languages"]:
                    raise DataIntegrityError(f"{row['_input_record']}: language outside configured EN/RU set")
                source_cleaned = _clean_source_markup(raw_text, source)
                quality_reason, quality_metrics = _cheap_quality_reason(source_cleaned, quality, source)
                provenance = {key: value for key, value in row.items()
                              if not key.startswith("_") and key != "text"}
                structural_reason = None if quality_reason else _quarantine_reason(source_cleaned, cleaning, source)
                if quality_reason or structural_reason:
                    provenance.update({"raw_sha256": row["raw_sha256"],
                                       "quality_text_sha256": hashlib.sha256(
                                           source_cleaned.encode("utf-8")).hexdigest(),
                                       "transform_version": CLEANING_PIPELINE_VERSION,
                                       "quality_metrics": quality_metrics})
                    write_output(quarantine, {**provenance,
                        "reason": quality_reason or structural_reason})
                    counts["quarantined"] += 1
                    counts["quality_rejected"] += int(quality_reason is not None)
                    mark_processed(current_cursor)
                    continue
                before = numeric_spans(raw_text)
                try:
                    text = normalize_text(source_cleaned, normalization)
                except ValueError as exc:
                    raise DataIntegrityError(str(exc)) from exc
                after = numeric_spans(text)
                if before != after:
                    write_output(quarantine, {**provenance,
                        "reason": "source_markup_cleanup_changes_numeric_spans",
                        "numeric_spans_before": before, "numeric_spans_after": after})
                    counts["quarantined"] += 1
                    mark_processed(current_cursor)
                    continue
                if not text.strip():
                    raise DataIntegrityError(f"{row['_input_record']}: empty document after source-aware cleaning")
                provenance.update({"raw_sha256": row["raw_sha256"],
                                   "clean_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                   "transform_version": CLEANING_PIPELINE_VERSION,
                                   "numeric_spans": after,
                                   "quality_metrics": quality_metrics})
                split = _split_for(row, split_salt, data_config["splits"])
                provenance["split"] = split
                kind, duplicate_of, similarity, duplicate_provenance = deduper.classify(
                    provenance["clean_sha256"], text, provenance)
                if kind != "kept":
                    duplicate_key = "exact_duplicates" if kind == "exact" else "near_duplicates"
                    counts[duplicate_key] += 1
                    write_output(duplicates, {**provenance, "duplicate_type": kind,
                        "duplicate_of_source_record": duplicate_of, "minhash_similarity_estimate": similarity,
                        "duplicate_of_provenance": duplicate_provenance,
                        "duplicate_text_sha256": provenance["clean_sha256"]})
                    mark_processed(current_cursor)
                    continue
                record = {**provenance, "text": text}
                write_output(handles[split], record)
                counts["kept"] += 1
                counts[split] += 1
                language_counts[split][row["language"].casefold()] += 1
                mark_processed(current_cursor)
        if pending_rows:
            require_disk_headroom()
            persist_corpus_state(current_cursor)
            cursor = current_cursor
            pending_rows = 0
        require_disk_headroom()
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
        raise
    for handle in handles.values():
        handle.close()
    quarantine.close()
    duplicates.close()
    deduper.close()
    split_hashes = {f"{name}.jsonl": sha256_file(stage / f"{name}.jsonl")
                    for name in handles}
    files = dict(split_hashes)
    files["duplicates.jsonl"] = sha256_file(stage / "duplicates.jsonl")
    files["quarantine.jsonl"] = sha256_file(stage / "quarantine.jsonl")
    for name in ("train", "validation", "blind"):
        (stage / name).mkdir(exist_ok=True)
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
        "cleaning": {"transform_version": CLEANING_PIPELINE_VERSION,
                     "unicode_normalization": cleaning["unicode_normalization"],
                     "numeric_integrity": "ordered numeric spans must match exactly",
                     "quality_policy_version": quality["policy_version"],
                     "quality_policy": quality,
                     "quality_rejected": counts["quality_rejected"]},
        "performance_profile": performance,
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
    for suffix in ("", "-wal", "-shm"):
        (output / f"dedup.sqlite3{suffix}").unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean, deduplicate, split and audit pinned SHINRA corpus sources")
    parser.add_argument("--input", nargs="+", required=True, help="source snapshot directories from data.sources")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--admission", choices=("audit", "production"), default="audit")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the committed JSONL cursor and SQLite dedup state")
    parser.add_argument("--minimum-free-gb", type=float, default=None,
                        help="pause while preserving the resumable stage below this free-space reserve")
    args = parser.parse_args()
    result = prepare_local_corpus(args.input, args.output_dir, config_path=args.config,
                                  admission=args.admission, resume=args.resume,
                                  minimum_free_gb=args.minimum_free_gb)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
