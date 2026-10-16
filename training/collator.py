"""Lazy row-group Parquet dataset with packed-manifest and language-token accounting."""

from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from configs.loader import load_config
from data.hashing import sha256_file
from data.packed_data import ShardError


class PackedParquetDataset(Dataset):
    LANGUAGE_CODES = {"en": 0, "ru": 1}

    def __init__(self, data_dir: str | Path, config: dict | None = None, expected_split: str | None = None,
                 *, allow_smoke_test: bool = False, allow_audit_data: bool = False,
                 include_row_indices: bool = False) -> None:
        self.config = config or load_config()
        self.include_row_indices = include_row_indices
        configured_bundle_sha = self.config["tokenizer"].get("artifact_sha256")
        root = Path(data_dir)
        self.files = sorted(root.glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No parquet files in {root}")
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"{root}: packed manifest.json is required")
        packed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared_sha = packed_manifest.pop("manifest_sha256", None)
        actual_sha = hashlib.sha256(json.dumps(packed_manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if declared_sha != actual_sha:
            raise ValueError(f"{root}: packed manifest self-hash mismatch")
        self.packed_manifest_sha256 = declared_sha
        admission = packed_manifest.get("admission")
        bundle_sha = packed_manifest.get("tokenizer_bundle_sha256")
        self.admission = admission
        self.tokenizer_bundle_sha256 = bundle_sha
        if packed_manifest.get("schema") != "nullxes.shinra.packed.v3" \
                or admission not in {"production", "audit_only"} \
                or not isinstance(bundle_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", bundle_sha):
            raise ValueError(f"{root}: packed artifact schema/admission/tokenizer identity is invalid")
        if admission == "production" and (not configured_bundle_sha or configured_bundle_sha != bundle_sha):
            raise ValueError(f"{root}: production packed tokenizer identity differs from the frozen YAML SHA")
        if admission == "audit_only" and not allow_audit_data:
            raise ValueError(f"{root}: audit-only packed data requires explicit allow_audit_data=True")
        if admission == "audit_only" and configured_bundle_sha and configured_bundle_sha != bundle_sha:
            raise ValueError(f"{root}: audit packed tokenizer identity differs from the configured SHA")
        if expected_split is not None and packed_manifest.get("split") != expected_split:
            raise ValueError(f"{root}: expected {expected_split} split")
        if packed_manifest.get("smoke_test_only") and not allow_smoke_test:
            raise ValueError("smoke-test shards cannot be used for training or evaluation")
        declared = packed_manifest.get("shard_files")
        shard_hashes = packed_manifest.get("shard_sha256")
        if not isinstance(declared, list) or sorted(declared) != [path.name for path in self.files]:
            raise ShardError(f"{root}: packed shard set differs from manifest")
        if not isinstance(shard_hashes, dict) or set(shard_hashes) != set(declared):
            raise ShardError(f"{root}: packed shard hashes are incomplete")
        source_hashes: set[str] = set()
        self._row_groups: list[tuple[int, int, int, int]] = []
        total_rows = 0
        languages: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        self._row_group_starts: list[int] = []
        self._parquet_files = [pq.ParquetFile(path) for path in self.files]
        source_allowlist = {source["id"]: source for source in self.config["data"]["sources"]}
        previous_document: str | None = None
        previous_chunk_index = -1
        previous_chunk_count = 0
        previous_token_end = 0
        previous_identity: tuple[Any, ...] | None = None
        documents_seen = body_tokens_seen = eod_tokens_seen = target_tokens_seen = 0
        for file_idx, (path, parquet) in enumerate(zip(self.files, self._parquet_files)):
            if sha256_file(path) != shard_hashes[path.name]:
                raise ShardError(f"{path}: SHA256 differs from packed manifest")
            if "language" not in parquet.schema_arrow.names or "labels" not in parquet.schema_arrow.names:
                raise ShardError(f"{path}: per-row language and explicit loss labels are required")
            if not {"tokenizer_bundle_sha256", "source_manifest_sha256", "input_ids", "attention_mask"}.issubset(
                parquet.schema_arrow.names
            ):
                raise ShardError(f"{path}: required tokenizer, corpus or sequence fields are missing")
            row_metadata_fields = {"source_id", "revision", "document_id", "upstream_document_id", "source_record",
                "subset", "license", "url_or_archive_id", "window_semantics",
                "rights_status", "language",
                "chunk_index", "chunk_count",
                "token_start", "token_end", "document_token_count", "split", "raw_sha256", "clean_sha256", "upstream_file_sha256",
                "is_final_document_window"}
            if not row_metadata_fields.issubset(parquet.schema_arrow.names):
                raise ShardError(f"{path}: per-row source/document/chunk provenance is incomplete")
            for row_group_idx in range(parquet.metadata.num_row_groups):
                metadata = parquet.metadata.row_group(row_group_idx)
                row_count = metadata.num_rows
                self._row_group_starts.append(total_rows)
                columns = ["labels", "tokenizer_bundle_sha256", "source_manifest_sha256",
                           *sorted(row_metadata_fields)]
                table = parquet.read_row_group(row_group_idx, columns=columns)
                if set(table.column("tokenizer_bundle_sha256").to_pylist()) != {bundle_sha}:
                    raise ShardError(f"{path}: row tokenizer identity is missing or mismatched")
                group_sources = set(table.column("source_manifest_sha256").to_pylist())
                if len(group_sources) != 1:
                    raise ShardError(f"{path}: mixed corpus manifests in a row group")
                source_hashes.update(group_sources)
                language_values = table.column("language").to_pylist()
                unknown = sorted(set(language_values) - self.LANGUAGE_CODES.keys())
                if unknown:
                    raise ShardError(f"{path}: unsupported language labels: {unknown}")
                language_codes = np.fromiter((self.LANGUAGE_CODES[value] for value in language_values),
                                             dtype=np.uint8, count=row_count)
                label_array = table.column("labels").combine_chunks()
                offsets = label_array.offsets.to_numpy(zero_copy_only=False)
                values = label_array.values.to_numpy(zero_copy_only=False)
                counts = np.empty(row_count, dtype=np.uint16)
                for row_idx in range(row_count):
                    start, end = int(offsets[row_idx]), int(offsets[row_idx + 1])
                    counts[row_idx] = np.count_nonzero(values[start + 1:end] != -100)
                metadata_columns = {name: table.column(name).to_pylist() for name in row_metadata_fields}
                for row_idx in range(row_count):
                    source_id = metadata_columns["source_id"][row_idx]
                    revision = metadata_columns["revision"][row_idx]
                    source = source_allowlist.get(source_id)
                    if source is None or source.get("revision") != revision:
                        raise ShardError(f"{path}: row source/revision is outside the current YAML allowlist")
                    if admission == "production" and (
                        source.get("status") != "approved" or source.get("rights_review_status") != "approved"
                        or source.get("provenance_review_status") != "approved"
                    ):
                        raise ShardError(f"{path}: production row belongs to a source without current approval")
                    if metadata_columns["split"][row_idx] != packed_manifest.get("split"):
                        raise ShardError(f"{path}: row split differs from the packed split manifest")
                    if metadata_columns["license"][row_idx] != source.get("license_claim"):
                        raise ShardError(f"{path}: row license claim differs from YAML source metadata")
                    if metadata_columns["rights_status"][row_idx] != source.get("rights_review_status"):
                        raise ShardError(f"{path}: row rights status differs from the source YAML")
                    if admission == "production" and metadata_columns["rights_status"][row_idx] != "approved":
                        raise ShardError(f"{path}: row has no production rights approval")
                    if not metadata_columns["source_record"][row_idx] or not metadata_columns["subset"][row_idx]:
                        raise ShardError(f"{path}: upstream row and source subset provenance are missing")
                    if metadata_columns["window_semantics"][row_idx] != "isolated-continuation-windows.v1":
                        raise ShardError(f"{path}: unsupported continuation-window semantics")
                    if not metadata_columns["upstream_document_id"][row_idx] \
                            or not metadata_columns["url_or_archive_id"][row_idx]:
                        raise ShardError(f"{path}: upstream document URL/archive provenance is missing")
                    document_id = metadata_columns["document_id"][row_idx]
                    chunk_index = int(metadata_columns["chunk_index"][row_idx])
                    chunk_count = int(metadata_columns["chunk_count"][row_idx])
                    token_start = int(metadata_columns["token_start"][row_idx])
                    token_end = int(metadata_columns["token_end"][row_idx])
                    document_token_count = int(metadata_columns["document_token_count"][row_idx])
                    final_window = bool(metadata_columns["is_final_document_window"][row_idx])
                    if not document_id or chunk_count <= 0 or not 0 <= chunk_index < chunk_count \
                            or token_start < 0 or token_end <= token_start or document_token_count < token_end:
                        raise ShardError(f"{path}: invalid document chunk offsets or indices")
                    if final_window != (chunk_index == chunk_count - 1):
                        raise ShardError(f"{path}: EOD/final-window declaration differs from chunk count")
                    if final_window != (token_end == document_token_count):
                        raise ShardError(f"{path}: final chunk does not account for the complete source token tail")
                    if token_end - token_start != int(counts[row_idx]) - int(final_window):
                        raise ShardError(f"{path}: source token offsets differ from supervised body-token count")
                    identity = tuple(metadata_columns[name][row_idx] for name in
                                     ("source_id", "revision", "language", "raw_sha256", "clean_sha256",
                                      "document_token_count"))
                    for hash_field in ("raw_sha256", "clean_sha256", "upstream_file_sha256"):
                        if not re.fullmatch(r"[a-f0-9]{64}", str(metadata_columns[hash_field][row_idx])):
                            raise ShardError(f"{path}: malformed per-row {hash_field}")
                    if document_id != previous_document:
                        if previous_document is not None and previous_chunk_index != previous_chunk_count - 1:
                            raise ShardError(f"{path}: document {previous_document} is missing one or more chunks")
                        if chunk_index != 0 or token_start != 0:
                            raise ShardError(f"{path}: document does not begin at chunk 0/token offset 0")
                        documents_seen += 1
                    else:
                        if chunk_index != previous_chunk_index + 1 or token_start != previous_token_end \
                                or chunk_count != previous_chunk_count or identity != previous_identity:
                            raise ShardError(f"{path}: document chunk sequence has a gap, overlap or provenance change")
                    previous_document = document_id
                    previous_chunk_index = chunk_index
                    previous_chunk_count = chunk_count
                    previous_token_end = token_end
                    previous_identity = identity
                    body_tokens_seen += token_end - token_start
                    eod_tokens_seen += int(final_window)
                    target_tokens_seen += int(counts[row_idx])
                languages.append(language_codes)
                targets.append(counts)
                self._row_groups.append((total_rows, total_rows + row_count, file_idx, row_group_idx))
                total_rows += row_count
        if previous_document is not None and previous_chunk_index != previous_chunk_count - 1:
            raise ShardError(f"packed split ends before document {previous_document} is complete")
        if len(source_hashes) != 1:
            raise ShardError(f"{root}: Parquet shards have inconsistent source manifests")
        self.source_manifest_sha256 = next(iter(source_hashes))
        if self.source_manifest_sha256 != packed_manifest.get("source_manifest_sha256"):
            raise ShardError(f"{root}: Parquet shards do not match packed manifest")
        frozen_corpus = self.config["data"].get("manifest_sha256")
        if frozen_corpus is not None and frozen_corpus != self.source_manifest_sha256:
            raise ShardError(f"{root}: corpus differs from frozen YAML identity")
        self.language_codes = np.concatenate(languages) if languages else np.empty(0, dtype=np.uint8)
        self.target_tokens = np.concatenate(targets) if targets else np.empty(0, dtype=np.uint16)
        self.length = total_rows
        if self.length == 0 or np.any(self.target_tokens == 0):
            raise ShardError(f"{root}: empty split or row without a supervised next-token target")
        report = packed_manifest.get("report", {})
        expected_counts = {
            "documents": documents_seen,
            "chunks": total_rows,
            "body_tokens": body_tokens_seen,
            "bos_tokens": total_rows,
            "eod_tokens": eod_tokens_seen,
            "input_tokens": body_tokens_seen + total_rows + eod_tokens_seen,
            "supervised_target_tokens": target_tokens_seen,
            "padded_slots": total_rows * int(self.config["model"]["train_sequence_length"])
                            - (body_tokens_seen + total_rows + eod_tokens_seen),
            "shards": len(self.files),
        }
        mismatches = {key: (report.get(key), value) for key, value in expected_counts.items()
                      if report.get(key) != value}
        if mismatches:
            raise ShardError(f"{root}: manifest token/document accounting differs from Parquet rows: {mismatches}")
        if expected_split == "train" and set(self.language_codes.tolist()) != {0, 1}:
            raise ShardError("training split must contain both English and Russian rows for token-level mixing")
        self._cache_key: tuple[int, int] | None = None
        self._cache_table = None

    def __len__(self) -> int:
        return self.length

    def _locate(self, index: int) -> tuple[int, int, int]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        group_position = bisect_right(self._row_group_starts, index) - 1
        start, end, file_idx, row_group_idx = self._row_groups[group_position]
        if index >= end:
            raise IndexError(index)
        return file_idx, row_group_idx, index - start

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, row_group_idx, row = self._locate(index)
        key = (file_idx, row_group_idx)
        if key != self._cache_key:
            self._cache_table = self._parquet_files[file_idx].read_row_group(row_group_idx)
            self._cache_key = key
        item = {name: self._cache_table.column(name)[row].as_py() for name in self._cache_table.column_names}
        from data.pack import pack_token_stream
        tok = self.config["tokenizer"]
        ids = item["input_ids"]
        active = next((i for i, value in enumerate(ids) if value == tok["pad_id"]), len(ids))
        length = int(self.config["model"]["train_sequence_length"])
        if len(ids) != length or any(value != tok["pad_id"] for value in ids[active:]):
            raise ShardError("packed row length or right-padding contract is invalid")
        is_final = bool(item["is_final_document_window"])
        if (active > 0 and ids[active - 1] == tok["end_of_document_id"]) != is_final:
            raise ShardError("EOD must appear exactly on the declared final window of a source document")
        expected = pack_token_stream([ids[:active]], length, config=self.config)[0]
        if any(item.get(field) != expected[field] for field in ("input_ids", "attention_mask", "labels")):
            raise ShardError("packed row BOS/EOD/PAD or loss mask contract is invalid")
        result = {"input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                  "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                  "labels": torch.tensor(item["labels"], dtype=torch.long)}
        if self.include_row_indices:
            result["row_index"] = index
        return result


def collate_lm(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = {key: torch.stack([row[key] for row in batch], dim=0)
              for key in batch[0] if key != "row_index"}
    if "row_index" in batch[0]:
        result["row_indices"] = [int(row["row_index"]) for row in batch]
    return result


def load_packed_dataset(data_dir: str | Path, pad_id: int | None = None, *, config: dict | None = None,
                        expected_split: str | None = None, allow_smoke_test: bool = False,
                        allow_audit_data: bool = False, include_row_indices: bool = False):
    config = config or load_config()
    configured_pad = int(config["tokenizer"]["pad_id"])
    if pad_id is not None and pad_id != configured_pad:
        raise ShardError("loader pad ID differs from configs/shinra.yaml")
    return PackedParquetDataset(data_dir, config, expected_split, allow_smoke_test=allow_smoke_test,
                                allow_audit_data=allow_audit_data, include_row_indices=include_row_indices)
