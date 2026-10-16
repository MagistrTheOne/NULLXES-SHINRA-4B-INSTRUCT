"""Pinned, streaming Hugging Face dataset adapters for SHINRA source records."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Iterator

from configs.loader import CONFIG_PATH, load_config
from data.hashing import sha256_file


class SourceAcquisitionError(RuntimeError):
    pass


COMMIT_RE = re.compile(r"^[a-f0-9]{40}$")


def _content_sha(info: Any) -> str:
    lfs = getattr(info, "lfs", None)
    value = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if not value:
        oid = lfs.get("oid") if isinstance(lfs, dict) else getattr(lfs, "oid", None)
        if isinstance(oid, str):
            candidate = oid.split(":", 1)[1] if oid.startswith("sha256:") else oid
            if re.fullmatch(r"[a-f0-9]{64}", candidate):
                value = candidate
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise SourceAcquisitionError(f"upstream file lacks an immutable LFS SHA-256: {getattr(info, 'path', '?')}")
    return value


def _file_rows(repo_id: str, revision: str, path: str, token: str | None,
               batch_size: int, required_columns: tuple[str, ...], file_format: str
               ) -> Iterator[dict[str, Any]]:
    try:
        from huggingface_hub import HfFileSystem
        if file_format == "parquet":
            import pyarrow.parquet as pq
        elif file_format == "jsonl.zst":
            import zstandard
        else:
            raise SourceAcquisitionError(f"unsupported pinned source format: {file_format}")
    except ImportError as exc:
        dependency = "pyarrow" if file_format == "parquet" else "zstandard"
        raise SourceAcquisitionError(
            f"source acquisition requires the preinstalled optional dependencies huggingface_hub and {dependency}"
        ) from exc

    fs = HfFileSystem(token=token)
    remote = f"datasets/{repo_id}@{revision}/{path}"
    try:
        with fs.open(remote, "rb") as stream:
            if file_format == "parquet":
                parquet = pq.ParquetFile(stream)
                missing_columns = sorted(set(required_columns) - set(parquet.schema_arrow.names))
                if missing_columns:
                    raise SourceAcquisitionError(
                        f"{repo_id}@{revision}/{path}: required Parquet columns are missing: {missing_columns}"
                    )
                for batch in parquet.iter_batches(batch_size=batch_size):
                    yield from batch.to_pylist()
            else:
                decompressor = zstandard.ZstdDecompressor().stream_reader(stream, read_across_frames=True)
                with io.TextIOWrapper(decompressor, encoding="utf-8", errors="strict", newline="") as text_stream:
                    for line_number, line in enumerate(text_stream, 1):
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            yield {"_source_parse_error": "malformed_jsonl", "_source_line": line_number}
                            continue
                        if not isinstance(row, dict):
                            yield {"_source_parse_error": "malformed_jsonl_row", "_source_line": line_number}
                            continue
                        yield row
    except SourceAcquisitionError:
        raise
    except Exception as exc:
        raise SourceAcquisitionError(f"failed streaming pinned {file_format} file {repo_id}@{revision}/{path}: {exc}") from exc


def _available_identifiers(row: Any, source: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    fields = [source.get("document_id_field"), source.get("url_field")]
    result: dict[str, Any] = {}
    for field in dict.fromkeys(name for name in fields if isinstance(name, str) and name):
        value = row.get(field)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            text = str(value)
            result[field] = text if len(text) <= 1024 else {
                "prefix": text[:1024], "truncated": True,
                "value_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
    return result


def _validate_parquet_schema(repo_id: str, revision: str, path: str, token: str | None,
                             required_columns: tuple[str, ...]) -> None:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(token=token)
        with fs.open(f"datasets/{repo_id}@{revision}/{path}", "rb") as stream:
            parquet = pq.ParquetFile(stream)
            missing = sorted(set(required_columns) - set(parquet.schema_arrow.names))
            if missing:
                raise SourceAcquisitionError(
                    f"{repo_id}@{revision}/{path}: required Parquet columns are missing: {missing}"
                )
    except SourceAcquisitionError:
        raise
    except Exception as exc:
        raise SourceAcquisitionError(
            f"failed validating pinned Parquet schema {repo_id}@{revision}/{path}: {exc}"
        ) from exc


def _rejection_reason(row: Any, text_field: str, id_field: str, *, path: str, row_index: int) -> str | None:
    if not isinstance(row, dict):
        return "malformed_row_shape"
    if row.get("_source_parse_error"):
        return str(row["_source_parse_error"])
    if text_field not in row:
        return "missing_text_field"
    text = row[text_field]
    if text is None:
        return "null_text"
    if not isinstance(text, str):
        return "non_string_text"
    if not text.strip():
        return "empty_text"
    if id_field == "__source_file_row__":
        if not path or row_index < 0:
            return "missing_document_identifier"
        return None
    if id_field not in row:
        return "missing_document_identifier"
    identifier = row[id_field]
    if identifier is None:
        return "null_document_identifier"
    if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
        return "non_scalar_document_identifier"
    if not str(identifier).strip():
        return "empty_document_identifier"
    return None


def _source_rows(source: dict[str, Any], token: str | None, batch_size: int
                 ) -> tuple[list[dict[str, str]], Iterator[dict[str, Any]], int]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SourceAcquisitionError("source acquisition requires the preinstalled optional huggingface_hub package") from exc

    repo_id = str(source["id"])
    revision = str(source["revision"])
    if not COMMIT_RE.fullmatch(revision):
        raise SourceAcquisitionError(f"{repo_id}: revision must be a full immutable 40-character commit SHA")
    api = HfApi(token=token)
    repo_info = api.repo_info(repo_id, repo_type="dataset", revision=revision)
    resolved_revision = getattr(repo_info, "sha", None)
    if resolved_revision != revision:
        raise SourceAcquisitionError(
            f"{repo_id}: upstream resolved revision {resolved_revision!r}, expected pinned commit {revision}"
        )
    file_format = source.get("format", "parquet")
    suffix = ".parquet" if file_format == "parquet" else ".jsonl.zst"
    paths = sorted(path for path in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
                   if any(path.startswith(prefix) for prefix in source["parquet_prefixes"])
                   and path.endswith(suffix))
    if not paths:
        raise SourceAcquisitionError(f"{repo_id}@{revision}: no {file_format} files match configured prefixes")
    candidate_file_count = len(paths)
    max_source_files = source.get("max_source_files")
    if max_source_files is not None:
        paths = paths[:int(max_source_files)]
    infos = api.get_paths_info(repo_id, paths, repo_type="dataset", revision=revision)
    file_provenance = [{"path": info.path, "sha256": _content_sha(info), "format": file_format}
                       for info in infos]
    if {item["path"] for item in file_provenance} != set(paths):
        raise SourceAcquisitionError(f"{repo_id}@{revision}: upstream metadata omitted one or more selected files")
    required_columns = [source["text_field"]]
    if source["document_id_field"] != "__source_file_row__":
        required_columns.append(source["document_id_field"])
    if source.get("language_field"):
        required_columns.append(source["language_field"])
    if source.get("script_field"):
        required_columns.append(source["script_field"])
    if source.get("quality_field"):
        required_columns.append(source["quality_field"])
    if file_format == "parquet":
        for path in paths:
            _validate_parquet_schema(repo_id, revision, path, token, tuple(required_columns))

    def rows() -> Iterator[dict[str, Any]]:
        for path in paths:
            file_sha = next(item["sha256"] for item in file_provenance if item["path"] == path)
            subset = source["subset_by_path"].get(next((prefix.rstrip("/") for prefix in source["parquet_prefixes"]
                                                        if path.startswith(prefix)), "default"), "default")
            for row_index, row in enumerate(_file_rows(
                repo_id, revision, path, token, batch_size, tuple(required_columns), file_format
            )):
                text_field = source["text_field"]
                id_field = source["document_id_field"]
                reason = _rejection_reason(row, text_field, id_field, path=path, row_index=row_index)
                if reason:
                    yield {"_outcome": "rejected", "rejection": {
                        "source_id": repo_id, "revision": revision, "upstream_file": path,
                        "upstream_row_index": row_index, "reason": reason,
                        "available_identifiers": _available_identifiers(row, source),
                    }, "_cursor": {"path": path, "row_index": row_index}}
                    continue
                text = row[text_field]
                subset = source["subset_by_path"].get(next((prefix.rstrip("/") for prefix in source["parquet_prefixes"]
                                                            if path.startswith(prefix)), "default"), "default")
                language = source["language_by_subset"].get(subset)
                if language not in {"en", "ru"}:
                    raise SourceAcquisitionError(f"{repo_id}: no configured language for subset {subset!r}")
                language_field = source.get("language_field")
                if language_field:
                    actual_language = row.get(language_field)
                    expected_label = source.get("language_labels", {}).get(language)
                    if actual_language != expected_label:
                        yield {"_outcome": "rejected", "rejection": {
                            "source_id": repo_id, "revision": revision, "upstream_file": path,
                            "upstream_row_index": row_index, "reason": "upstream_language_mismatch",
                            "available_identifiers": _available_identifiers(row, source),
                        }, "_cursor": {"path": path, "row_index": row_index}}
                        continue
                script_field = source.get("script_field")
                if script_field and row.get(script_field) != source.get("script_labels", {}).get(language):
                    yield {"_outcome": "rejected", "rejection": {
                        "source_id": repo_id, "revision": revision, "upstream_file": path,
                        "upstream_row_index": row_index, "reason": "upstream_script_mismatch",
                        "available_identifiers": _available_identifiers(row, source),
                    }, "_cursor": {"path": path, "row_index": row_index}}
                    continue
                latin_fraction = None
                if source.get("language_validation_profile") == "english_source_latin_script_guard":
                    letter_count = 0
                    latin_count = 0
                    for char in text:
                        if char.isalpha():
                            letter_count += 1
                            latin_count += int(unicodedata.name(char, "").startswith("LATIN "))
                    latin_fraction = latin_count / letter_count if letter_count else 0.0
                    if not letter_count or latin_fraction < float(source["minimum_latin_script_fraction"]):
                        yield {"_outcome": "rejected", "rejection": {
                            "source_id": repo_id, "revision": revision, "upstream_file": path,
                            "upstream_row_index": row_index,
                            "reason": "english_source_non_latin_script_guard",
                            "available_identifiers": _available_identifiers(row, source),
                        }, "_cursor": {"path": path, "row_index": row_index}}
                        continue
                quality_field = source.get("quality_field")
                quality_score = row.get(quality_field) if quality_field else None
                minimum_score = source.get("minimum_quality_score")
                if minimum_score is not None and (
                    isinstance(quality_score, bool) or not isinstance(quality_score, (int, float))
                    or not math.isfinite(float(quality_score))
                ):
                    yield {"_outcome": "rejected", "rejection": {
                        "source_id": repo_id, "revision": revision, "upstream_file": path,
                        "upstream_row_index": row_index, "reason": "missing_or_invalid_quality_score",
                        "available_identifiers": _available_identifiers(row, source),
                    }, "_cursor": {"path": path, "row_index": row_index}}
                    continue
                if minimum_score is not None and float(quality_score) < float(minimum_score):
                    yield {"_outcome": "rejected", "rejection": {
                        "source_id": repo_id, "revision": revision, "upstream_file": path,
                        "upstream_row_index": row_index, "reason": "below_source_quality_threshold",
                        "available_identifiers": _available_identifiers(row, source),
                    }, "_cursor": {"path": path, "row_index": row_index}}
                    continue
                if id_field == "__source_file_row__":
                    external_id = f"{path}#{row_index}"
                else:
                    external_id = str(row[id_field])
                stable_id = hashlib.sha256(f"{repo_id}\0{revision}\0{subset}\0{external_id}".encode()).hexdigest()
                raw_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                url = row.get(source.get("url_field", "")) if source.get("url_field") else None
                if not url and source.get("url_template"):
                    url = source["url_template"].format(identifier=external_id)
                yield {"_outcome": "accepted", "record": {
                    "text": text,
                    "source_id": repo_id,
                    "revision": revision,
                    "subset": subset,
                    "document_id": stable_id,
                    "upstream_document_id": external_id,
                    "url_or_archive_id": str(url or external_id),
                    "title": str(row.get(source.get("title_field", ""), "")) if source.get("title_field") else "",
                    "license": source["license_claim"],
                    "rights_status": source["rights_review_status"],
                    "language": language,
                    "upstream_language_label": row.get(language_field) if language_field else language,
                    "upstream_script_label": row.get(script_field) if script_field else None,
                    "language_validation_profile": source.get("language_validation_profile"),
                    "latin_script_fraction": latin_fraction,
                    "quality_score": float(quality_score) if quality_score is not None else None,
                    "quality_method": source.get("quality_method"),
                    "raw_sha256": raw_sha,
                    "upstream_file": path,
                    "upstream_file_sha256": file_sha,
                    "upstream_row_index": row_index,
                    "source_record": f"{repo_id}@{revision}/{path}#{row_index}",
                }, "_cursor": {"path": path, "row_index": row_index}}

    return file_provenance, rows(), candidate_file_count


def acquire_source(source_id: str, output_dir: str | Path, *, config_path: str | Path = CONFIG_PATH,
                   admission: str = "audit", max_documents: int | None = None,
                   max_target_tokens: int | None = None, tokenizer_path: str | Path | None = None,
                   resume: bool = False, token: str | None = None,
                   minimum_free_gb: float | None = None) -> dict[str, Any]:
    """Stream a pinned source into a resumable, immutable raw JSONL snapshot.

    Audit acquisition is permitted for candidate sources. Production acquisition requires both
    independent rights and provenance approvals in the single YAML configuration.
    """
    if admission not in {"audit", "production"}:
        raise SourceAcquisitionError("admission must be audit or production")
    if max_documents is not None and max_documents <= 0:
        raise SourceAcquisitionError("max_documents must be positive")
    if max_target_tokens is not None and max_target_tokens <= 0:
        raise SourceAcquisitionError("max_target_tokens must be positive")
    if max_documents is not None and max_target_tokens is not None:
        raise SourceAcquisitionError("choose one acquisition bound: accepted documents or tokenizer target tokens")
    if max_target_tokens is not None and tokenizer_path is None:
        raise SourceAcquisitionError("--tokenizer is required when bounding acquisition by SHINRA target tokens")
    config = load_config(config_path)
    reserve_gb = (float(config["runtime"]["minimum_free_disk_gb"])
                  if minimum_free_gb is None else float(minimum_free_gb))
    if reserve_gb < 0:
        raise SourceAcquisitionError("minimum_free_gb cannot be negative")
    source = next((item for item in config["data"]["sources"] if item["id"] == source_id), None)
    if source is None:
        raise SourceAcquisitionError(f"source is not in the configured exact-revision allowlist: {source_id}")
    destination = Path(output_dir)
    if destination.exists():
        raise SourceAcquisitionError(f"refusing to overwrite existing source snapshot: {destination}")
    if admission == "production" and not (
        source.get("status") == "approved"
        and source.get("rights_review_status") == "approved"
        and source.get("provenance_review_status") == "approved"
    ):
        raise SourceAcquisitionError(f"{source_id}: production admission is blocked pending rights/provenance approval")
    tokenizer = None
    tokenizer_bundle_sha = None
    forbidden_token_ids: set[int] = set()
    if max_target_tokens is not None:
        from tokenizer import load_shinra_tokenizer
        from tokenizer.contract import special_ids
        tokenizer = load_shinra_tokenizer(tokenizer_path, config, require_frozen_bundle=False)
        tokenizer_bundle_sha = tokenizer.shinra_bundle_sha256
        forbidden_token_ids = set(special_ids(config).values())
    files, rows, candidate_file_count = _source_rows(source, token, int(config["data"]["parquet_batch_rows"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.partial"
    state_path = stage / "acquisition_state.json"
    identity = {"source_id": source_id, "revision": source["revision"], "files": files,
                "candidate_file_count": candidate_file_count,
                "max_documents": max_documents, "max_target_tokens": max_target_tokens,
                "tokenizer_bundle_sha256": tokenizer_bundle_sha}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if stage.exists() and not resume:
        raise SourceAcquisitionError(f"partial acquisition exists; pass --resume to verify and continue: {stage}")
    if resume and (not stage.is_dir() or not state_path.is_file()):
        raise SourceAcquisitionError(f"no resumable acquisition state exists at {stage}")
    if not stage.exists():
        if resume:
            raise SourceAcquisitionError(f"no partial acquisition is available to resume: {stage}")
        stage.mkdir()
    records_path = stage / "records.jsonl"
    rejections_path = stage / "rejections.jsonl"
    if resume:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceAcquisitionError(f"invalid acquisition resume state: {exc}") from exc
        if state.get("identity_sha256") != identity_sha:
            raise SourceAcquisitionError("acquisition resume source files, revision, bounds or tokenizer differ")
        accepted, scanned, rejected = (int(state[key]) for key in ("accepted", "scanned", "rejected"))
        target_tokens = int(state["acquired_supervised_target_tokens"])
        rejected_by_reason = {str(k): int(v) for k, v in state["rejected_by_reason"].items()}
        cursor = state.get("committed_cursor")
        for path, offset in ((records_path, int(state["records_bytes"])),
                             (rejections_path, int(state["rejections_bytes"]))):
            if not path.is_file() or path.stat().st_size < offset:
                raise SourceAcquisitionError(f"resume journal points beyond an acquisition stream: {path}")
            with path.open("r+b") as stream:
                stream.truncate(offset)
    else:
        accepted = scanned = rejected = target_tokens = 0
        rejected_by_reason = {}
        cursor = None
        records_path.touch()
        rejections_path.touch()

    def persist_state(record_stream, rejection_stream, committed_cursor: dict[str, Any] | None) -> None:
        for stream in (record_stream, rejection_stream):
            stream.flush()
            os.fsync(stream.fileno())
        state = {"schema": "nullxes.shinra.source-acquisition-state.v1", "identity_sha256": identity_sha,
                 "accepted": accepted, "scanned": scanned, "rejected": rejected,
                 "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
                 "acquired_supervised_target_tokens": target_tokens,
                 "records_bytes": record_stream.tell(), "rejections_bytes": rejection_stream.tell(),
                 "committed_cursor": committed_cursor}
        temporary_state = stage / ".acquisition_state.json.tmp"
        temporary_state.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with temporary_state.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_state, state_path)

    def require_disk_headroom() -> None:
        if not reserve_gb:
            return
        available_gb = shutil.disk_usage(stage).free / (1024 ** 3)
        if available_gb < reserve_gb:
            raise SourceAcquisitionError(
                f"pausing resumable source acquisition: free={available_gb:.2f} GiB, "
                f"configured minimum={reserve_gb:.2f} GiB"
            )

    try:
        if not resume:
            with records_path.open("a", encoding="utf-8", newline="\n") as stream, \
                    rejections_path.open("a", encoding="utf-8", newline="\n") as rejected_stream:
                persist_state(stream, rejected_stream, None)
        with records_path.open("a", encoding="utf-8", newline="\n") as stream, \
                rejections_path.open("a", encoding="utf-8", newline="\n") as rejected_stream:
            pending = 0
            for outcome in rows:
                check_every = int(config["data"]["cleaning"]["disk_check_records"])
                if scanned % check_every == 0:
                    require_disk_headroom()
                current = outcome.get("_cursor")
                if not isinstance(current, dict) or not isinstance(current.get("path"), str) \
                        or type(current.get("row_index")) is not int:
                    raise SourceAcquisitionError("source adapter emitted an invalid resumable cursor")
                if cursor is not None and (current["path"], current["row_index"]) <= (
                        cursor["path"], int(cursor["row_index"])):
                    continue
                if outcome["_outcome"] == "rejected":
                    rejection = outcome["rejection"]
                    rejected_stream.write(json.dumps(rejection, ensure_ascii=False, sort_keys=True) + "\n")
                    rejected += 1
                    reason = rejection["reason"]
                    rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1
                else:
                    if outcome["_outcome"] != "accepted" or not isinstance(outcome.get("record"), dict):
                        raise SourceAcquisitionError("source adapter emitted an invalid acquisition outcome")
                    record = outcome["record"]
                    token_ids = None
                    if tokenizer is not None:
                        from tokenizer.contract import plain_text
                        try:
                            normalized_text = plain_text(record["text"], config)
                        except ValueError:
                            rejection = {"source_id": source_id, "revision": source["revision"],
                                         "upstream_file": record["upstream_file"],
                                         "upstream_row_index": record["upstream_row_index"],
                                         "reason": "reserved_control_or_unroundtrippable_text",
                                         "available_identifiers": {
                                             "upstream_document_id": record["upstream_document_id"],
                                             "url_or_archive_id": record["url_or_archive_id"],
                                         }}
                            rejected_stream.write(json.dumps(rejection, ensure_ascii=False, sort_keys=True) + "\n")
                            rejected += 1
                            scanned += 1
                            rejected_by_reason[rejection["reason"]] = rejected_by_reason.get(rejection["reason"], 0) + 1
                            pending += 1
                            current = outcome["_cursor"]
                            if pending >= int(config["data"]["parquet_batch_rows"]):
                                persist_state(stream, rejected_stream, current)
                                cursor = current
                                pending = 0
                            continue
                        token_ids = tokenizer.encode(normalized_text, add_special_tokens=False, truncation=False)
                        if any(int(token_id) in forbidden_token_ids for token_id in token_ids):
                            raise SourceAcquisitionError(
                                f"{record['source_record']}: ordinary text encoded to a reserved control ID"
                            )
                    stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    accepted += 1
                    if tokenizer is not None:
                        target_tokens += len(token_ids) + 1
                scanned += 1
                pending += 1
                reached_bound = (max_documents is not None and accepted >= max_documents) \
                    or (max_target_tokens is not None and target_tokens >= max_target_tokens)
                if pending >= int(config["data"]["parquet_batch_rows"]) or reached_bound:
                    persist_state(stream, rejected_stream, current)
                    cursor = current
                    pending = 0
                if reached_bound:
                    break
            if pending:
                require_disk_headroom()
                persist_state(stream, rejected_stream, current)
                cursor = current
            require_disk_headroom()
        if scanned == 0:
            raise SourceAcquisitionError(f"{source_id}@{source['revision']}: source returned no rows")
        manifest = {
            "schema": "nullxes.shinra.raw-source.v3",
            "purpose": "pretraining-candidate-source-audit",
            "admission": "production" if admission == "production" and not (max_documents or max_target_tokens) else "audit_only",
            "source_id": source_id,
            "revision": source["revision"],
            "parquet_batch_rows": config["data"]["parquet_batch_rows"],
            "stream_format": source.get("format", "parquet"),
            "identity_strategy": "immutable_upstream_file_row" if source["document_id_field"] == "__source_file_row__"
                                else "upstream_document_identifier",
            "license_claim": source["license_claim"],
            "rights_review_status": source.get("rights_review_status", "audit_only"),
            "provenance_review_status": source.get("provenance_review_status", "audit_only"),
            "files": files,
            "candidate_file_count": candidate_file_count,
            "selected_file_count": len(files),
            "file_selection": {"prefixes": source["parquet_prefixes"],
                               "max_source_files": source.get("max_source_files"),
                               "order": "lexicographic_path_within_pinned_revision"},
            "records": accepted,
            "accepted": accepted,
            "scanned": scanned,
            "rejected": rejected,
            "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
            "bounded_slice": max_documents is not None or max_target_tokens is not None,
            "max_accepted_documents": max_documents,
            "requested_max_target_tokens": max_target_tokens,
            "acquired_supervised_target_tokens": target_tokens if tokenizer is not None else None,
            "target_limit_reached": max_target_tokens is not None and target_tokens >= max_target_tokens,
            "tokenizer_bundle_sha256": tokenizer_bundle_sha,
            "target_token_count_semantics": "body-token targets plus one final EOD per accepted document; pre-clean estimate"
                                             if tokenizer is not None else None,
            "records_sha256": sha256_file(records_path),
            "rejections_file": "rejections.jsonl",
            "rejections_sha256": sha256_file(rejections_path),
        }
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        temporary_manifest = stage / ".manifest.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, stage / "manifest.json")
        os.replace(stage, destination)
        (destination / "acquisition_state.json").unlink(missing_ok=True)
        return manifest
    except Exception:
        # Keep only data committed by the checksummed resume journal. On resume, later bytes are truncated.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire an immutable SHINRA Hugging Face source snapshot")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--admission", choices=("audit", "production"), default="audit")
    parser.add_argument("--max-documents", type=int, default=None,
                        help="maximum accepted documents; rejected rows are still scanned and recorded")
    parser.add_argument("--max-target-tokens", type=int, default=None,
                        help="stop after at least this many Native tokenizer targets (whole documents; may overshoot)")
    parser.add_argument("--tokenizer", default=None,
                        help="verified Native tokenizer bundle; required for --max-target-tokens")
    parser.add_argument("--resume", action="store_true",
                        help="verify the immutable source, truncate to the last committed cursor, and continue")
    parser.add_argument("--minimum-free-gb", type=float, default=None,
                        help="pause while preserving the resumable stage below this free-space reserve")
    parser.add_argument("--token", default=None, help="optional Hugging Face token; prefer HF_TOKEN environment login")
    args = parser.parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    result = acquire_source(args.source_id, args.output_dir, config_path=args.config,
                            admission=args.admission, max_documents=args.max_documents,
                            max_target_tokens=args.max_target_tokens, tokenizer_path=args.tokenizer,
                            resume=args.resume, token=token, minimum_free_gb=args.minimum_free_gb)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
