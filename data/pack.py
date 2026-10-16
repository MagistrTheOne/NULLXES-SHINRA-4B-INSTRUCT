"""Document-isolated chunking and atomically published Parquet split artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from configs.loader import CONFIG_PATH, load_config
from data.hashing import sha256_file
from tokenizer import load_shinra_tokenizer
from tokenizer.contract import encode_text, plain_text

PACKER_VERSION = "shinra-pack.v3"
CHUNKING_POLICY_VERSION = "isolated-continuation-windows.v1"


class PackContractError(RuntimeError):
    pass


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def current_packed_forbidden_ids(config: dict | None = None) -> frozenset[int]:
    config = config or load_config()
    tok = config["tokenizer"]
    return frozenset({int(tok["end_of_turn_id"]), *(int(i) for i in tok["special_token_ids"].values()
                      if int(i) != int(tok["end_of_document_id"]) )})


def assert_pretrain_body_clean(body_ids: list[int], config: dict | None = None) -> None:
    tok = (config or load_config())["tokenizer"]
    forbidden = {int(tok["unk_id"]), int(tok["bos_id"]), int(tok["end_of_turn_id"]), int(tok["pad_id"]),
                 int(tok["end_of_document_id"]), *map(int, tok["special_token_ids"].values())}
    overlap = forbidden.intersection(body_ids)
    if overlap:
        raise PackContractError(f"special control IDs in document body: {sorted(overlap)}")
    if any(not 0 <= i < int(tok["vocab_size"]) for i in body_ids):
        raise PackContractError("document body contains an out-of-range token ID")


def assert_packed_ids_clean(ids: list[int], config: dict | None = None) -> None:
    overlap = current_packed_forbidden_ids(config).intersection(ids)
    if overlap:
        raise PackContractError(f"forbidden special IDs in packed sequence: {sorted(overlap)}")


def wrap_pretrain_window(body_ids: list[int], *, final_window: bool,
                         config: dict | None = None) -> list[int]:
    config = config or load_config()
    tok = config["tokenizer"]
    if not body_ids:
        raise PackContractError("document windows must contain at least one source token")
    if not config["data"]["packing"].get("isolate_document_attention"):
        raise PackContractError("document isolation is disabled; this packer only implements isolated rows")
    if config["data"]["packing"].get("loss_on_bos") is not False:
        raise PackContractError("isolated document windows require loss_on_bos=False")
    assert_pretrain_body_clean(body_ids, config)
    wrapped = [int(tok["bos_id"]), *body_ids]
    if final_window:
        if not config["data"]["packing"].get("loss_on_eod", True):
            raise PackContractError("excluding EOD from loss is not supported by the row contract")
        wrapped.append(int(tok["end_of_document_id"]))
    assert_packed_ids_clean(wrapped, config)
    return wrapped


def wrap_pretrain_document(body_ids: list[int], bos_id: int | None = None, end_id: int | None = None,
                           config: dict | None = None) -> list[int]:
    config = config or load_config()
    tok = config["tokenizer"]
    if bos_id is not None and bos_id != int(tok["bos_id"]):
        raise PackContractError("BOS ID differs from configs/shinra.yaml")
    if end_id is not None and end_id != int(tok["end_of_document_id"]):
        raise PackContractError("document end ID differs from configs/shinra.yaml")
    return wrap_pretrain_window(body_ids, final_window=True, config=config)


def _row(ids: list[int], pad_id: int, config: dict) -> dict:
    assert_packed_ids_clean(ids, config)
    tok = config["tokenizer"]
    mask = [int(token != pad_id) for token in ids]
    labels = [-100] + [token if keep else -100 for token, keep in list(zip(ids, mask))[1:]]
    for index, token in enumerate(ids):
        if token in {pad_id, int(tok["bos_id"])}:
            labels[index] = -100
        elif index > 0 and mask[index - 1] == 0:
            labels[index] = -100
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def chunk_token_ranges(token_count: int, sequence_length: int) -> list[tuple[int, int, bool]]:
    """Return non-overlapping half-open body ranges for independent continuation windows."""
    if sequence_length < 3:
        raise PackContractError("sequence_length must leave room for BOS, one source token and terminal EOD")
    if token_count <= 0:
        raise PackContractError("cannot chunk an empty document")
    final_capacity = sequence_length - 2
    continuation_capacity = sequence_length - 1
    if token_count <= final_capacity:
        return [(0, token_count, True)]
    chunk_count = (token_count - final_capacity + continuation_capacity - 1) // continuation_capacity + 1
    result: list[tuple[int, int, bool]] = []
    offset = 0
    remaining_chunks = chunk_count
    while remaining_chunks > 1:
        # Reserve at least one body token for each remaining window, including the terminal EOD window.
        size = min(continuation_capacity, token_count - offset - (remaining_chunks - 1))
        if size <= 0:
            raise PackContractError("chunk allocation produced an empty continuation window")
        end = offset + size
        result.append((offset, end, False))
        offset = end
        remaining_chunks -= 1
    end = token_count
    final_length = end - offset
    if not final_length or final_length > final_capacity:
        raise PackContractError("chunk allocation violated final-window capacity")
    result.append((offset, end, True))
    if result[0][0] != 0 or result[-1][1] != token_count \
            or any(result[index][1] != result[index + 1][0] for index in range(len(result) - 1)):
        raise PackContractError("chunking ranges lost, reordered or duplicated source token offsets")
    return result


def chunk_token_ids(token_ids: list[int], sequence_length: int) -> list[tuple[list[int], int, int, bool]]:
    return [(token_ids[start:end], start, end, final)
            for start, end, final in chunk_token_ranges(len(token_ids), sequence_length)]


def pack_token_stream(documents: list[list[int]], sequence_length: int, pad_id: int | None = None,
                      config: dict | None = None) -> list[dict]:
    config = config or load_config()
    if sequence_length < 3:
        raise PackContractError("sequence_length must be at least 3")
    configured_pad = int(config["tokenizer"]["pad_id"])
    pad_id = configured_pad if pad_id is None else pad_id
    if pad_id != configured_pad:
        raise PackContractError("pad ID differs from configs/shinra.yaml")
    if config["data"]["packing"]["policy"] != "isolated_documents":
        raise PackContractError("only isolated document packing is implemented")
    rows: list[dict] = []
    tok = config["tokenizer"]
    for document in documents:
        if len(document) < 2 or document[0] != tok["bos_id"]:
            raise PackContractError("wrapped window requires BOS and at least one body token")
        final = document[-1] == tok["end_of_document_id"]
        if final and len(document) < 3:
            raise PackContractError("terminal window requires BOS, body and EOD")
        body = document[1:-1] if final else document[1:]
        if not body:
            raise PackContractError("wrapped window requires at least one source body token")
        assert_pretrain_body_clean(body, config)
        if len(document) > sequence_length:
            raise PackContractError("window exceeds sequence_length; hidden truncation is forbidden")
        assert_packed_ids_clean(document, config)
        padded = list(document) + [pad_id] * (sequence_length - len(document))
        rows.append(_row(padded, pad_id, config))
    return rows


def iter_clean_records(input_dir: Path, split: str, manifest: dict,
                       config: dict | None = None) -> Iterator[dict[str, Any]]:
    config = config or load_config()
    source_allowlist = {source["id"]: source for source in config["data"]["sources"]}
    path = input_dir / f"{split}.jsonl"
    expected_hash = manifest.get("files", {}).get(f"{split}.jsonl")
    if not path.is_file() or not expected_hash:
        raise PackContractError(f"cleaned {split}.jsonl or its manifest hash is missing")
    if sha256_file(path) != expected_hash:
        raise PackContractError(f"{path}: SHA256 differs from corpus manifest")
    required = ("text", "source_id", "revision", "license", "language", "split", "document_id", "subset",
                "source_record", "rights_status", "url_or_archive_id", "raw_sha256", "clean_sha256",
                "upstream_file_sha256")
    from data.numeric_integrity import numeric_spans

    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PackContractError(f"{path}:{line_no}: malformed JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise PackContractError(f"{path}:{line_no}: record must be an object")
            missing = [key for key in required if not isinstance(row.get(key), str) or not row[key].strip()]
            if missing:
                raise PackContractError(f"{path}:{line_no}: missing provenance/text fields {missing}")
            if row["split"] != split:
                raise PackContractError(f"{path}:{line_no}: expected split {split}, got {row['split']}")
            source = source_allowlist.get(row["source_id"])
            if source is None or source.get("revision") != row["revision"]:
                raise PackContractError(f"{path}:{line_no}: source revision is outside the configured allowlist")
            if row["license"] != source.get("license_claim"):
                raise PackContractError(f"{path}:{line_no}: source license metadata differs from configs/shinra.yaml")
            if manifest.get("admission") == "production" and (
                source.get("status") != "approved" or source.get("rights_review_status") != "approved"
                or source.get("provenance_review_status") != "approved"
            ):
                raise PackContractError(f"{path}:{line_no}: source lacks production rights/provenance approval")
            if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row["clean_sha256"]:
                raise PackContractError(f"{path}:{line_no}: clean text hash mismatch")
            if numeric_spans(row["text"]) != row.get("numeric_spans"):
                raise PackContractError(f"{path}:{line_no}: numeric span audit mismatch")
            yield row


def resolve_pretrain_special_ids(tokenizer, config: dict | None = None) -> tuple[int, int, int]:
    config = config or load_config()
    tok = config["tokenizer"]
    ids_by_string = {tok["special_tokens"][key]: int(value) for key, value in tok["special_token_ids"].items()}
    ids_by_string.update({tok["special_tokens"]["bos"]: int(tok["bos_id"]),
                          tok["special_tokens"]["end_of_turn"]: int(tok["end_of_turn_id"]),
                          tok["special_tokens"]["pad"]: int(tok["pad_id"]),
                          tok["special_tokens"]["unk"]: int(tok["unk_id"]),
                          tok["special_tokens"]["end_of_document"]: int(tok["end_of_document_id"])})
    actual = {token: tokenizer.convert_tokens_to_ids(token) for token in ids_by_string}
    if actual != ids_by_string:
        raise PackContractError(f"tokenizer special ID mismatch: {actual} != {ids_by_string}")
    if tokenizer.eos_token_id != int(tok["end_of_turn_id"]):
        raise PackContractError("tokenizer eos_token_id must be EOT")
    return int(tok["bos_id"]), int(tok["end_of_document_id"]), int(tok["pad_id"])


def _write_journal(stage: Path, body: dict[str, Any]) -> None:
    journal = {**body}
    journal["journal_sha256"] = _json_sha(journal)
    temporary = stage / ".journal.json.tmp"
    temporary.write_text(json.dumps(journal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, stage / "journal.json")


def _load_journal(stage: Path, identity: dict[str, Any]) -> dict[str, Any]:
    path = stage / "journal.json"
    try:
        journal = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackContractError(f"partial split lacks a readable recovery journal: {exc}") from exc
    digest = journal.pop("journal_sha256", None)
    if digest != _json_sha(journal):
        raise PackContractError("partial split journal self-hash is invalid")
    if journal.get("identity") != identity:
        raise PackContractError("partial split identity differs from the current corpus/tokenizer/config")
    names = journal.get("shard_files", [])
    hashes = journal.get("shard_sha256", {})
    if not isinstance(names, list) or names != [f"packed-{index:05d}.parquet" for index in range(len(names))]:
        raise PackContractError("partial split journal shard names are not a contiguous ordered sequence")
    if not isinstance(hashes, dict) or set(names) != set(hashes):
        raise PackContractError("partial split journal has incomplete shard checksums")
    counter_fields = ("next_document_index", "documents", "chunks", "body_tokens", "bos_tokens",
                      "eod_tokens", "target_tokens", "padded_slots")
    if any(type(journal.get(field)) is not int or journal[field] < 0 for field in counter_fields):
        raise PackContractError("partial split journal contains invalid counters")
    if journal["next_document_index"] != journal["documents"]:
        raise PackContractError("partial split journal source cursor differs from its committed document count")
    if journal["chunks"] != journal["bos_tokens"] or journal["eod_tokens"] > journal["chunks"]:
        raise PackContractError("partial split journal token/window counters are inconsistent")
    if journal["target_tokens"] != journal["body_tokens"] + journal["eod_tokens"]:
        raise PackContractError("partial split journal target-token accounting is inconsistent")
    if journal["padded_slots"] != journal["chunks"] * int(identity["sequence_length"]) - (
        journal["body_tokens"] + journal["bos_tokens"] + journal["eod_tokens"]
    ):
        raise PackContractError("partial split journal padding accounting is inconsistent")
    actual_files = sorted(path.name for path in stage.glob("packed-*.parquet"))
    if not set(names).issubset(actual_files):
        raise PackContractError("partial split shard set is missing journaled data")
    for name in names:
        if sha256_file(stage / name) != hashes[name]:
            raise PackContractError(f"partial split shard checksum mismatch: {name}")
    orphaned = sorted(set(actual_files) - set(names))
    expected_orphan = f"packed-{len(names):05d}.parquet"
    if orphaned:
        if orphaned != [expected_orphan]:
            raise PackContractError(f"partial split contains unjournaled shard files: {orphaned}")
        try:
            pq.ParquetFile(stage / expected_orphan).metadata.num_rows
        except Exception as exc:
            raise PackContractError(f"uncommitted shard is not a readable Parquet file: {expected_orphan}") from exc
        # The preceding journal is the commit point; replay its source documents after removing this orphan.
        (stage / expected_orphan).unlink()
    for temporary in stage.glob(".packed-*.parquet.tmp"):
        temporary.unlink()
    return journal


def pack_documents(input_dir: Path, tokenizer_path: Path, output_dir: Path, sequence_length: int,
                   shard_tokens: int, *, config_path: Path, required_split: str,
                   config: dict | None = None, smoke_test_documents: int | None = None,
                   resume: bool = False) -> dict:
    config = config or load_config(config_path)
    if smoke_test_documents is not None and smoke_test_documents <= 0:
        raise PackContractError("smoke_test_documents must be positive")
    if config["data"]["packing"]["policy"] != "isolated_documents":
        raise PackContractError("only isolated document packing is implemented")
    if sequence_length != int(config["model"]["train_sequence_length"]):
        raise PackContractError("sequence_length must match model.train_sequence_length in configs/shinra.yaml")
    if shard_tokens != int(config["data"]["packed_shard_tokens"]):
        raise PackContractError("shard_tokens must match data.packed_shard_tokens in configs/shinra.yaml")
    corpus_path = input_dir / "manifest.json"
    if not corpus_path.is_file():
        raise PackContractError("corpus manifest.json is required")
    corpus_manifest = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus_manifest_sha = corpus_manifest.get("manifest_sha256")
    if corpus_manifest_sha != _json_sha({k: v for k, v in corpus_manifest.items() if k != "manifest_sha256"}):
        raise PackContractError("corpus manifest self-hash is invalid")
    corpus_admission = corpus_manifest.get("admission")
    if corpus_manifest.get("purpose") != "pretraining" or corpus_admission not in {"production", "audit_only"}:
        raise PackContractError("packing requires a pretraining corpus manifest with an explicit admission status")
    if config["data"].get("manifest_sha256") is not None and config["data"]["manifest_sha256"] != corpus_manifest_sha:
        raise PackContractError("corpus manifest SHA256 does not match configs/shinra.yaml")
    if corpus_manifest.get("split_hashes", {}).get(required_split) is None:
        raise PackContractError(f"corpus manifest has no frozen {required_split} split hash")
    source_allowlist = {source["id"]: source for source in config["data"]["sources"]}
    source_revisions = corpus_manifest.get("source_revisions")
    if not isinstance(source_revisions, list) or not source_revisions:
        raise PackContractError("corpus manifest lacks source revision and review provenance")
    for provenance in source_revisions:
        source = source_allowlist.get(provenance.get("source_id"))
        if source is None or source.get("revision") != provenance.get("revision"):
            raise PackContractError("corpus manifest contains a source revision outside the YAML allowlist")
        if corpus_manifest.get("admission") == "production" and (
            source.get("status") != "approved" or source.get("rights_review_status") != "approved"
            or source.get("provenance_review_status") != "approved"
            or provenance.get("rights_review_status") != "approved"
            or provenance.get("provenance_review_status") != "approved"
        ):
            raise PackContractError("production corpus manifest contains a source without both approvals")
    tokenizer = load_shinra_tokenizer(
        tokenizer_path, config, require_frozen_bundle=(corpus_admission == "production")
    )
    bundle_sha = tokenizer.shinra_bundle_sha256
    _, _, pad_id = resolve_pretrain_special_ids(tokenizer, config)
    output_dir = Path(output_dir)
    output_root = output_dir.parent
    if output_dir.exists():
        raise PackContractError(f"refusing to overwrite published packed split directory: {output_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    root_manifest = output_root / "manifest.json"
    if root_manifest.exists():
        existing_root = json.loads(root_manifest.read_text(encoding="utf-8"))
        if existing_root.get("manifest_sha256") != corpus_manifest_sha:
            raise PackContractError("packed output root belongs to a different corpus manifest")
    else:
        temporary_root_manifest = output_root / ".manifest.json.tmp"
        temporary_root_manifest.write_text(json.dumps(corpus_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_root_manifest, root_manifest)
    stage = output_root / f".{output_dir.name}.partial"
    identity = {"source_manifest_sha256": corpus_manifest_sha,
                "source_split_sha256": corpus_manifest["files"][f"{required_split}.jsonl"],
                "tokenizer_bundle_sha256": bundle_sha, "config_sha256": _json_sha(config),
                "split": required_split, "sequence_length": sequence_length,
                "shard_tokens": shard_tokens, "packer": PACKER_VERSION,
                "chunking_policy": CHUNKING_POLICY_VERSION}
    if stage.exists():
        if not resume:
            raise PackContractError(f"partial output exists; pass --resume to verify and continue: {stage}")
        journal = _load_journal(stage, identity)
    else:
        if resume:
            raise PackContractError(f"no partial split is available to resume: {stage}")
        stage.mkdir(parents=True)
        journal = {"identity": identity, "next_document_index": 0, "documents": 0, "chunks": 0,
                   "body_tokens": 0, "bos_tokens": 0, "eod_tokens": 0, "target_tokens": 0,
                   "padded_slots": 0, "shard_files": [], "shard_sha256": {}}
        _write_journal(stage, journal)
    row_group_rows: list[dict] = []
    row_group_size = int(config["data"]["packing"]["parquet_row_group_rows"])
    writer = None
    temporary_shard: Path | None = None
    shard_active_slots = 0
    shard_target_tokens = 0
    shard_index_row_count = 0
    shard_files = list(journal["shard_files"])
    shard_hashes = dict(journal["shard_sha256"])
    documents = int(journal["documents"])
    chunks = int(journal["chunks"])
    body_tokens = int(journal["body_tokens"])
    bos_tokens = int(journal["bos_tokens"])
    eod_tokens = int(journal["eod_tokens"])
    target_tokens = int(journal["target_tokens"])
    padded_slots = int(journal["padded_slots"])
    next_document_index = int(journal["next_document_index"])
    shard_index = len(shard_files)

    def flush_row_group() -> None:
        nonlocal writer, temporary_shard, row_group_rows
        if not row_group_rows:
            return
        table = pa.Table.from_pylist(row_group_rows)
        if writer is None:
            temporary_shard = stage / f".packed-{shard_index:05d}.parquet.tmp"
            writer = pq.ParquetWriter(temporary_shard, table.schema)
        writer.write_table(table, row_group_size=row_group_size)
        row_group_rows = []

    def append_row(row: dict) -> None:
        nonlocal shard_active_slots, shard_target_tokens, shard_index_row_count
        row_group_rows.append(row)
        shard_active_slots += sum(row["attention_mask"])
        shard_target_tokens += sum(value != -100 for value in row["labels"][1:])
        shard_index_row_count += 1
        if len(row_group_rows) >= row_group_size:
            flush_row_group()

    def commit_shard(next_index: int) -> None:
        nonlocal writer, temporary_shard, shard_index, target_tokens, padded_slots
        nonlocal shard_active_slots, shard_target_tokens, shard_index_row_count
        flush_row_group()
        if writer is None or temporary_shard is None:
            return
        writer.close()
        writer = None
        name = f"packed-{shard_index:05d}.parquet"
        os.replace(temporary_shard, stage / name)
        temporary_shard = None
        digest = sha256_file(stage / name)
        shard_files.append(name)
        shard_hashes[name] = digest
        target_tokens += shard_target_tokens
        padded_slots += shard_index_row_count * sequence_length - shard_active_slots
        shard_index += 1
        shard_active_slots = 0
        shard_target_tokens = 0
        shard_index_row_count = 0
        _write_journal(stage, {"identity": identity, "next_document_index": next_index,
            "documents": documents, "chunks": chunks, "body_tokens": body_tokens, "bos_tokens": bos_tokens,
            "eod_tokens": eod_tokens, "target_tokens": target_tokens, "padded_slots": padded_slots,
            "shard_files": shard_files, "shard_sha256": shard_hashes})

    try:
        for document_index, record in enumerate(tqdm(
                iter_clean_records(input_dir, required_split, corpus_manifest, config), desc=f"pack:{required_split}")):
            if document_index < next_document_index:
                continue
            if plain_text(record["text"], config) != record["text"]:
                raise PackContractError("cleaned corpus differs from configured NFC normalization")
            ids = encode_text(tokenizer, record["text"], config)
            if not ids:
                raise PackContractError(f"{record['document_id']}: tokenizer produced an empty sequence")
            pieces = chunk_token_ranges(len(ids), sequence_length)
            count = len(pieces)
            for chunk_index, (token_start, token_end, final_window) in enumerate(pieces):
                body = ids[token_start:token_end]
                wrapped = wrap_pretrain_window(body, final_window=final_window, config=config)
                packed_row = pack_token_stream([wrapped], sequence_length, pad_id, config)[0]
                packed_row.update({
                    "source_manifest_sha256": corpus_manifest_sha,
                    "tokenizer_bundle_sha256": bundle_sha,
                    "source_id": record["source_id"],
                    "revision": record["revision"],
                    "document_id": record["document_id"],
                    "upstream_document_id": record.get("upstream_document_id", record["document_id"]),
                    "source_record": record.get("source_record", record["document_id"]),
                    "subset": record["subset"],
                    "url_or_archive_id": record["url_or_archive_id"],
                    "license": record["license"],
                    "rights_status": record["rights_status"],
                    "language": record["language"],
                    "split": record["split"],
                    "raw_sha256": record["raw_sha256"],
                    "clean_sha256": record["clean_sha256"],
                    "upstream_file_sha256": record["upstream_file_sha256"],
                    "chunk_index": chunk_index,
                    "chunk_count": count,
                    "token_start": token_start,
                    "token_end": token_end,
                    "document_token_count": len(ids),
                    "window_semantics": CHUNKING_POLICY_VERSION,
                    "is_final_document_window": final_window,
                })
                append_row(packed_row)
                chunks += 1
                body_tokens += len(body)
                bos_tokens += 1
                eod_tokens += int(final_window)
            documents += 1
            if smoke_test_documents is not None and documents >= smoke_test_documents:
                next_document_index = document_index + 1
                break
            next_document_index = document_index + 1
            if shard_active_slots >= shard_tokens:
                commit_shard(next_document_index)
        if writer is not None or row_group_rows:
            commit_shard(next_document_index)
        if not documents:
            raise PackContractError("cannot publish an empty packed split")
        split_path = input_dir / f"{required_split}.jsonl"
        if sha256_file(split_path) != corpus_manifest["files"][f"{required_split}.jsonl"]:
            raise PackContractError("corpus split changed during packing")
        report = {
            "packer": PACKER_VERSION, "chunking_policy": CHUNKING_POLICY_VERSION,
            "window_semantics": "independent continuation windows; no overlap or cross-document packing",
            "split": required_split, "documents": documents, "chunks": chunks,
            "smoke_test_only": smoke_test_documents is not None,
            "body_tokens": body_tokens, "bos_tokens": bos_tokens, "eod_tokens": eod_tokens,
            "input_tokens": body_tokens + bos_tokens + eod_tokens,
            "supervised_target_tokens": target_tokens,
            "sequence_length": sequence_length, "shards": len(shard_files),
            "shard_files": shard_files, "shard_sha256": shard_hashes,
            "padded_slots": padded_slots,
            "packing_efficiency": (body_tokens + bos_tokens + eod_tokens) /
                                  max((body_tokens + bos_tokens + eod_tokens) + padded_slots, 1),
        }
        (stage / "pack_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        packed_manifest = {
            "schema": "nullxes.shinra.packed.v3",
            "admission": "audit_only" if corpus_admission != "production" or smoke_test_documents else "production",
            "source_manifest_sha256": corpus_manifest_sha, "tokenizer_bundle_sha256": bundle_sha,
            "smoke_test_only": smoke_test_documents is not None, "split": required_split,
            "shard_files": shard_files, "shard_sha256": shard_hashes, "report": report,
        }
        packed_manifest["manifest_sha256"] = _json_sha(packed_manifest)
        (stage / "manifest.json").write_text(json.dumps(packed_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
        return report
    except Exception:
        # Keep the checksummed journal and completed shards for an explicit --resume attempt.
        if writer is not None:
            writer.close()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack an admitted SHINRA corpus split into isolated Parquet windows")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--shard-tokens", type=int, default=None)
    parser.add_argument("--split", choices=("train", "validation", "blind", "all"), required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--resume", action="store_true", help="verify the partial-split journal and resume safely")
    args = parser.parse_args()
    config = load_config(args.config)
    sequence_length = args.sequence_length or int(config["model"]["train_sequence_length"])
    shard_tokens = args.shard_tokens or int(config["data"]["packed_shard_tokens"])
    input_dir = Path(args.input_dir)
    tokenizer_path = Path(args.tokenizer)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    if args.split == "all":
        if args.resume:
            parser.error("resume one interrupted split at a time with --split <name> --resume")
        results = {}
        for split in ("train", "validation", "blind"):
            results[split] = pack_documents(
                input_dir, tokenizer_path, output_dir / split, sequence_length, shard_tokens,
                config_path=config_path, required_split=split, config=config,
            )
        print(json.dumps(results, indent=2, sort_keys=True))
        return
    result = pack_documents(input_dir, tokenizer_path, output_dir, sequence_length,
                            shard_tokens, config_path=config_path, required_split=args.split, resume=args.resume,
                            config=config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
