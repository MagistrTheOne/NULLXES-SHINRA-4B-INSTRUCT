"""Pinned, streaming Hugging Face dataset adapters for SHINRA source records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
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
               batch_size: int, required_columns: tuple[str, ...]) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
    except ImportError as exc:
        raise SourceAcquisitionError(
            "source acquisition requires the preinstalled optional dependencies pyarrow and huggingface_hub"
        ) from exc

    fs = HfFileSystem(token=token)
    remote = f"datasets/{repo_id}@{revision}/{path}"
    try:
        with fs.open(remote, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            missing_columns = sorted(set(required_columns) - set(parquet.schema_arrow.names))
            if missing_columns:
                raise SourceAcquisitionError(
                    f"{repo_id}@{revision}/{path}: required Parquet columns are missing: {missing_columns}"
                )
            for batch in parquet.iter_batches(batch_size=batch_size):
                yield from batch.to_pylist()
    except SourceAcquisitionError:
        raise
    except Exception as exc:
        raise SourceAcquisitionError(f"failed streaming pinned Parquet file {repo_id}@{revision}/{path}: {exc}") from exc


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


def _rejection_reason(row: Any, text_field: str, id_field: str) -> str | None:
    if not isinstance(row, dict):
        return "malformed_row_shape"
    if text_field not in row:
        return "missing_text_field"
    text = row[text_field]
    if text is None:
        return "null_text"
    if not isinstance(text, str):
        return "non_string_text"
    if not text.strip():
        return "empty_text"
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
                 ) -> tuple[list[dict[str, str]], Iterator[dict[str, Any]]]:
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
    paths = sorted(path for path in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
                   if any(path.startswith(prefix) for prefix in source["parquet_prefixes"])
                   and path.endswith(".parquet"))
    if not paths:
        raise SourceAcquisitionError(f"{repo_id}@{revision}: no Parquet files match configured prefixes")
    infos = api.get_paths_info(repo_id, paths, repo_type="dataset", revision=revision)
    file_provenance = [{"path": info.path, "sha256": _content_sha(info)} for info in infos]
    if {item["path"] for item in file_provenance} != set(paths):
        raise SourceAcquisitionError(f"{repo_id}@{revision}: upstream metadata omitted one or more Parquet files")
    required_columns = (source["text_field"], source["document_id_field"])
    for path in paths:
        _validate_parquet_schema(repo_id, revision, path, token, required_columns)

    def rows() -> Iterator[dict[str, Any]]:
        for path in paths:
            file_sha = next(item["sha256"] for item in file_provenance if item["path"] == path)
            subset = source["subset_by_path"].get(next((prefix.rstrip("/") for prefix in source["parquet_prefixes"]
                                                        if path.startswith(prefix)), "default"), "default")
            for row_index, row in enumerate(_file_rows(
                repo_id, revision, path, token, batch_size,
                (source["text_field"], source["document_id_field"]),
            )):
                text_field = source["text_field"]
                id_field = source["document_id_field"]
                reason = _rejection_reason(row, text_field, id_field)
                if reason:
                    yield {"_outcome": "rejected", "rejection": {
                        "source_id": repo_id, "revision": revision, "upstream_file": path,
                        "upstream_row_index": row_index, "reason": reason,
                        "available_identifiers": _available_identifiers(row, source),
                    }}
                    continue
                text = row[text_field]
                upstream_id = row[id_field]
                language = source["language_by_subset"].get(subset)
                if language not in {"en", "ru"}:
                    raise SourceAcquisitionError(f"{repo_id}: no configured language for subset {subset!r}")
                external_id = str(upstream_id)
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
                    "raw_sha256": raw_sha,
                    "upstream_file": path,
                    "upstream_file_sha256": file_sha,
                    "upstream_row_index": row_index,
                    "source_record": f"{repo_id}@{revision}/{path}#{row_index}",
                }}

    return file_provenance, rows()


def acquire_source(source_id: str, output_dir: str | Path, *, config_path: str | Path = CONFIG_PATH,
                   admission: str = "audit", max_documents: int | None = None, token: str | None = None) -> dict[str, Any]:
    """Stream pinned Parquet rows to an immutable raw JSONL snapshot.

    Audit acquisition is permitted for candidate sources. Production acquisition requires both
    independent rights and provenance approvals in the single YAML configuration.
    """
    if admission not in {"audit", "production"}:
        raise SourceAcquisitionError("admission must be audit or production")
    if max_documents is not None and max_documents <= 0:
        raise SourceAcquisitionError("max_documents must be positive")
    config = load_config(config_path)
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
    files, rows = _source_rows(source, token, int(config["data"]["parquet_batch_rows"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    accepted = 0
    scanned = 0
    rejected = 0
    rejected_by_reason: dict[str, int] = {}
    try:
        records_path = stage / "records.jsonl"
        rejections_path = stage / "rejections.jsonl"
        with records_path.open("w", encoding="utf-8", newline="\n") as stream:
            with rejections_path.open("w", encoding="utf-8", newline="\n") as rejected_stream:
                for outcome in rows:
                    if outcome["_outcome"] == "rejected":
                        rejection = outcome["rejection"]
                        rejected_stream.write(json.dumps(rejection, ensure_ascii=False, sort_keys=True) + "\n")
                        scanned += 1
                        rejected += 1
                        reason = rejection["reason"]
                        rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1
                        continue
                    if outcome["_outcome"] != "accepted" or not isinstance(outcome.get("record"), dict):
                        raise SourceAcquisitionError("source adapter emitted an invalid acquisition outcome")
                    stream.write(json.dumps(outcome["record"], ensure_ascii=False, sort_keys=True) + "\n")
                    accepted += 1
                    scanned += 1
                    if max_documents is not None and accepted >= max_documents:
                        break
        if scanned == 0:
            raise SourceAcquisitionError(f"{source_id}@{source['revision']}: source returned no rows")
        manifest = {
            "schema": "nullxes.shinra.raw-source.v2",
            "purpose": "pretraining-candidate-source-audit",
            "admission": "production" if admission == "production" and max_documents is None else "audit_only",
            "source_id": source_id,
            "revision": source["revision"],
            "parquet_batch_rows": config["data"]["parquet_batch_rows"],
            "license_claim": source["license_claim"],
            "rights_review_status": source.get("rights_review_status", "audit_only"),
            "provenance_review_status": source.get("provenance_review_status", "audit_only"),
            "files": files,
            "records": accepted,
            "accepted": accepted,
            "scanned": scanned,
            "rejected": rejected,
            "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
            "bounded_slice": max_documents is not None,
            "max_accepted_documents": max_documents,
            "records_sha256": sha256_file(records_path),
            "rejections_file": "rejections.jsonl",
            "rejections_sha256": sha256_file(rejections_path),
        }
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        temporary_manifest = stage / ".manifest.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_manifest, stage / "manifest.json")
        os.replace(stage, destination)
        return manifest
    except Exception:
        import shutil

        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire an immutable SHINRA Hugging Face source snapshot")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--admission", choices=("audit", "production"), default="audit")
    parser.add_argument("--max-documents", type=int, default=None,
                        help="maximum accepted documents; rejected rows are still scanned and recorded")
    parser.add_argument("--token", default=None, help="optional Hugging Face token; prefer HF_TOKEN environment login")
    args = parser.parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    result = acquire_source(args.source_id, args.output_dir, config_path=args.config,
                            admission=args.admission, max_documents=args.max_documents, token=token)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
