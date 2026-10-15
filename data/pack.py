"""Token packing with strict source, split, tokenizer and control ID contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from configs.loader import load_config

PACKER_VERSION = "shinra-pack.v2"


class PackContractError(RuntimeError):
    pass


def current_packed_forbidden_ids(config: dict | None = None) -> frozenset[int]:
    config = config or load_config()
    tok = config["tokenizer"]
    return frozenset({int(tok["end_of_turn_id"]), *(int(i) for i in tok["special_token_ids"].values()
                      if int(i) != int(tok["end_of_document_id"]) )})


def assert_pretrain_body_clean(body_ids: list[int], config: dict | None = None) -> None:
    tok = (config or load_config())["tokenizer"]
    forbidden = {int(tok["bos_id"]), int(tok["end_of_turn_id"]), int(tok["pad_id"]),
                 int(tok["end_of_document_id"]), *map(int, tok["special_token_ids"].values())}
    overlap = forbidden.intersection(body_ids)
    if overlap:
        raise PackContractError(f"special control IDs in document body: {sorted(overlap)}")


def assert_packed_ids_clean(ids: list[int], config: dict | None = None) -> None:
    overlap = current_packed_forbidden_ids(config).intersection(ids)
    if overlap:
        raise PackContractError(f"forbidden special IDs in packed sequence: {sorted(overlap)}")


def wrap_pretrain_document(body_ids: list[int], bos_id: int | None = None, end_id: int | None = None,
                           config: dict | None = None) -> list[int]:
    config = config or load_config()
    tok = config["tokenizer"]
    bos_id = int(tok["bos_id"]) if bos_id is None else bos_id
    end_id = int(tok["end_of_document_id"]) if end_id is None else end_id
    if not body_ids:
        raise PackContractError("empty document")
    if bos_id != int(tok["bos_id"]) or end_id != int(tok["end_of_document_id"]):
        raise PackContractError("BOS or document end ID differs from configs/shinra.yaml")
    assert_pretrain_body_clean(body_ids, config)
    wrapped = [bos_id, *body_ids, end_id]
    assert_packed_ids_clean(wrapped, config)
    return wrapped


def _row(ids: list[int], pad_id: int, config: dict) -> dict:
    assert_packed_ids_clean(ids, config)
    mask = [int(token != pad_id) for token in ids]
    labels = [token if keep else -100 for token, keep in zip(ids, mask)]
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def pack_token_stream(documents: list[list[int]], sequence_length: int, pad_id: int | None = None,
                      config: dict | None = None) -> list[dict]:
    config = config or load_config()
    if sequence_length < 2:
        raise PackContractError("sequence_length must be at least 2")
    configured_pad = int(config["tokenizer"]["pad_id"])
    pad_id = configured_pad if pad_id is None else pad_id
    if pad_id != configured_pad:
        raise PackContractError("pad ID differs from configs/shinra.yaml")
    buffer: list[int] = []
    rows: list[dict] = []
    for document in documents:
        if not document:
            raise PackContractError("empty wrapped document")
        assert_packed_ids_clean(document, config)
        buffer.extend(document)
        while len(buffer) >= sequence_length:
            rows.append(_row(buffer[:sequence_length], pad_id, config))
            buffer = buffer[sequence_length:]
    if buffer:
        buffer.extend([pad_id] * (sequence_length - len(buffer)))
        rows.append(_row(buffer, pad_id, config))
    return rows


def iter_parquet_records(input_dir: Path):
    files = sorted(Path(input_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards in {input_dir}")
    required = {"text", "source_id", "revision", "license", "split", "raw_sha256", "clean_sha256"}
    for file in files:
        table = pq.read_table(file)
        missing = required - set(table.column_names)
        if missing:
            raise PackContractError(f"{file}: missing provenance columns {sorted(missing)}")
        for row in table.to_pylist():
            if not isinstance(row.get("text"), str) or not row["text"]:
                raise PackContractError(f"{file}: malformed record text")
            if row["split"] not in {"train", "validation", "blind"}:
                raise PackContractError(f"{file}: unsupported split {row['split']!r}")
            if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row["clean_sha256"]:
                raise PackContractError(f"{file}: clean text hash mismatch")
            if row.get("numeric_spans") is not None:
                from data.numeric_integrity import numeric_spans

                if numeric_spans(row["text"]) != row["numeric_spans"]:
                    raise PackContractError(f"{file}: numeric span audit mismatch")
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("source_id", "revision", "license", "raw_sha256")):
                raise PackContractError(f"{file}: required provenance missing")
            yield row


def resolve_pretrain_special_ids(tokenizer, config: dict | None = None) -> tuple[int, int, int]:
    config = config or load_config()
    tok = config["tokenizer"]
    ids_by_string = {tok["special_tokens"][key]: int(value) for key, value in tok["special_token_ids"].items()}
    ids_by_string.update({tok["special_tokens"]["bos"]: int(tok["bos_id"]),
                          tok["special_tokens"]["end_of_turn"]: int(tok["end_of_turn_id"]),
                          tok["special_tokens"]["pad"]: int(tok["pad_id"])})
    actual = {token: tokenizer.convert_tokens_to_ids(token) for token in ids_by_string}
    if actual != ids_by_string:
        raise PackContractError(f"tokenizer special ID mismatch: {actual} != {ids_by_string}")
    if tokenizer.eos_token_id != int(tok["end_of_turn_id"]):
        raise PackContractError("tokenizer eos_token_id must be EOT")
    return int(tok["bos_id"]), int(tok["end_of_document_id"]), int(tok["pad_id"])


def pack_documents(input_dir: Path, tokenizer_path: Path, output_dir: Path, sequence_length: int,
                   shard_tokens: int, *, config_path: Path, required_split: str) -> dict:
    config = load_config(config_path)
    data_manifest = input_dir / "manifest.json"
    if not data_manifest.is_file():
        raise PackContractError("corpus manifest.json is required")
    corpus_manifest = json.loads(data_manifest.read_text(encoding="utf-8"))
    corpus_manifest_sha = corpus_manifest.get("manifest_sha256")
    if not corpus_manifest_sha:
        raise PackContractError("corpus manifest has no self-hash")
    if config["data"].get("manifest_sha256") is not None and config["data"]["manifest_sha256"] != corpus_manifest_sha:
        raise PackContractError("corpus manifest SHA256 does not match configs/shinra.yaml")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    bos_id, end_id, pad_id = resolve_pretrain_special_ids(tokenizer, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer: list[int] = []
    rows: list[dict] = []
    shard_index = documents = tokens_in = tokens_out = 0
    shard_paths = []
    for record in tqdm(iter_parquet_records(input_dir), desc="pack"):
        if record["split"] != required_split:
            raise PackContractError(f"expected {required_split} records, got {record['split']}")
        ids = tokenizer.encode(record["text"], add_special_tokens=False, truncation=False)
        if not ids:
            raise PackContractError("tokenizer produced an empty sequence")
        documents += 1
        tokens_in += len(ids)
        buffer.extend(wrap_pretrain_document(ids, bos_id, end_id, config))
        while len(buffer) >= sequence_length:
            rows.append(_row(buffer[:sequence_length], pad_id, config))
            buffer = buffer[sequence_length:]
            if len(rows) * sequence_length >= shard_tokens:
                shard_paths.append(_write_parquet_shard(rows, output_dir, shard_index,
                                                        source_manifest_sha256=corpus_manifest_sha))
                tokens_out += len(rows) * sequence_length
                shard_index += 1
                rows = []
    if buffer:
        buffer.extend([pad_id] * (sequence_length - len(buffer)))
        rows.append(_row(buffer, pad_id, config))
    if rows:
        shard_paths.append(_write_parquet_shard(rows, output_dir, shard_index,
                                                source_manifest_sha256=corpus_manifest_sha))
        tokens_out += len(rows) * sequence_length
        shard_index += 1
    report = {"packer": PACKER_VERSION, "split": required_split, "documents": documents,
              "tokens_in": tokens_in, "tokens_out": tokens_out, "sequence_length": sequence_length,
              "shards": shard_index, "shard_files": shard_paths,
              "packing_efficiency": tokens_out / max(tokens_in, 1)}
    (output_dir / "pack_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    packed_manifest_path = output_dir / "manifest.json"
    packed_manifest = {"schema": "nullxes.shinra.packed.v1", "source_manifest_sha256": corpus_manifest_sha,
                       "split": required_split, "shard_files": shard_paths,
                       "shard_sha256": {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                                         for name in shard_paths}, "report": report}
    packed_manifest["manifest_sha256"] = hashlib.sha256(json.dumps(packed_manifest, sort_keys=True).encode()).hexdigest()
    packed_manifest_path.write_text(json.dumps(packed_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _write_parquet_shard(rows: list[dict], output_dir: Path, index: int, *, source_manifest_sha256: str) -> str:
    name = f"packed-{index:05d}.parquet"
    enriched = [{**row, "source_manifest_sha256": source_manifest_sha256} for row in rows]
    Dataset.from_list(enriched).to_parquet(str(output_dir / name))
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack audited SHINRA corpus Parquet into model sequences")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--shard-tokens", type=int, required=True)
    parser.add_argument("--split", choices=("train", "validation", "blind"), required=True)
    parser.add_argument("--config", default="configs/shinra.yaml")
    args = parser.parse_args()
    result = pack_documents(Path(args.input_dir), Path(args.tokenizer), Path(args.output_dir), args.sequence_length,
                            args.shard_tokens, config_path=Path(args.config), required_split=args.split)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
