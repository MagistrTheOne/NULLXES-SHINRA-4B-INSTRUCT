"""Token packing with strict source, split, tokenizer and control ID contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
from configs.loader import load_config
from tokenizer import load_shinra_tokenizer
from tokenizer.contract import encode_text, plain_text
from tokenizer.bundle import file_sha256

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


def wrap_pretrain_document(body_ids: list[int], bos_id: int | None = None, end_id: int | None = None,
                           config: dict | None = None) -> list[int]:
    config = config or load_config()
    tok = config["tokenizer"]
    bos_id = int(tok["bos_id"]) if bos_id is None else bos_id
    end_id = int(tok["end_of_document_id"]) if end_id is None else end_id
    if not body_ids:
        raise PackContractError("empty document")
    if not config["data"]["packing"].get("isolate_document_attention"):
        raise PackContractError("document isolation is disabled; this packer only implements isolated rows")
    if int(tok["pad_id"]) in body_ids:
        raise PackContractError("PAD token is forbidden in source document bodies")
    if bos_id != int(tok["bos_id"]) or end_id != int(tok["end_of_document_id"]):
        raise PackContractError("BOS or document end ID differs from configs/shinra.yaml")
    assert_pretrain_body_clean(body_ids, config)
    if config["data"]["packing"].get("loss_on_bos") is not False:
        raise PackContractError("isolated document packing requires loss_on_bos=False")
    if not config["data"]["packing"].get("loss_on_eod", True):
        raise PackContractError("excluding EOD from loss is not supported by the current row contract")
    wrapped = [bos_id, *body_ids, end_id]
    assert_packed_ids_clean(wrapped, config)
    return wrapped


def _row(ids: list[int], pad_id: int, config: dict) -> dict:
    assert_packed_ids_clean(ids, config)
    tok = config["tokenizer"]
    mask = [int(token != pad_id) for token in ids]
    labels = [-100] + [token if keep else -100 for token, keep in list(zip(ids, mask))[1:]]
    for index, token in enumerate(ids):
        if token in {pad_id, int(tok["bos_id"])}:
            labels[index] = -100
        elif index > 0 and mask[index - 1] == 0:
            # Do not train a target across a padded row boundary.
            labels[index] = -100
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def _write_parquet_shard(rows: list[dict], output_dir: Path, index: int, *, source_manifest_sha256: str) -> str:
    name = f"packed-{index:05d}.parquet"
    enriched = [{**row, "source_manifest_sha256": source_manifest_sha256} for row in rows]
    pq.write_table(pa.Table.from_pylist(enriched), output_dir / name)
    return name


def _flush_rows(rows: list[dict], output_dir: Path, shard_index: int, source_manifest_sha256: str,
                sequence_length: int) -> tuple[str, int, int]:
    shard_name = _write_parquet_shard(rows, output_dir, shard_index,
                                      source_manifest_sha256=source_manifest_sha256)
    tokens = sum(sum(row["attention_mask"]) for row in rows)
    return shard_name, tokens, len(rows) * sequence_length


def pack_token_stream(documents: list[list[int]], sequence_length: int, pad_id: int | None = None,
                      config: dict | None = None) -> list[dict]:
    config = config or load_config()
    if sequence_length < 2:
        raise PackContractError("sequence_length must be at least 2")
    configured_pad = int(config["tokenizer"]["pad_id"])
    pad_id = configured_pad if pad_id is None else pad_id
    if pad_id != configured_pad:
        raise PackContractError("pad ID differs from configs/shinra.yaml")
    if config["data"]["packing"]["policy"] != "isolated_documents":
        raise PackContractError("only isolated document packing is implemented")
    rows: list[dict] = []
    for document in documents:
        if len(document) < 3:
            raise PackContractError("wrapped document requires BOS, nonempty body and EOD")
        tok = config["tokenizer"]
        if document[0] != tok["bos_id"] or document[-1] != tok["end_of_document_id"]:
            raise PackContractError("wrapped document must begin with BOS and end with EOD")
        assert_pretrain_body_clean(document[1:-1], config)
        if len(document) > sequence_length:
            raise PackContractError("document exceeds sequence_length; explicit chunking policy is required")
        assert_packed_ids_clean(document, config)
        padded = list(document) + [pad_id] * (sequence_length - len(document))
        rows.append(_row(padded, pad_id, config))
    return rows


def iter_clean_records(input_dir: Path, split: str, manifest: dict):
    path = input_dir / f"{split}.jsonl"
    expected_hash = manifest.get("files", {}).get(f"{split}.jsonl")
    if not path.is_file() or not expected_hash:
        raise PackContractError(f"cleaned {split}.jsonl or its manifest hash is missing")
    if file_sha256(path) != expected_hash:
        raise PackContractError(f"{path}: SHA256 differs from corpus manifest")
    required = ("text", "source_id", "revision", "license", "split", "raw_sha256", "clean_sha256")
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


def pack_documents(input_dir: Path, tokenizer_path: Path, output_dir: Path, sequence_length: int,
                   shard_tokens: int, *, config_path: Path, required_split: str,
                   config: dict | None = None, smoke_test_documents: int | None = None) -> dict:
    config = config or load_config(config_path)
    if smoke_test_documents is not None and smoke_test_documents <= 0:
        raise PackContractError("smoke_test_documents must be positive")
    if config["data"]["packing"]["policy"] != "isolated_documents":
        raise PackContractError("only isolated document packing is implemented")
    if sequence_length != int(config["model"]["train_sequence_length"]):
        raise PackContractError("sequence_length must match model.train_sequence_length in configs/shinra.yaml")
    if shard_tokens != int(config["data"]["packed_shard_tokens"]):
        raise PackContractError("shard_tokens must match data.packed_shard_tokens in configs/shinra.yaml")
    data_manifest = input_dir / "manifest.json"
    if not data_manifest.is_file():
        raise PackContractError("corpus manifest.json is required")
    corpus_manifest = json.loads(data_manifest.read_text(encoding="utf-8"))
    corpus_manifest_sha = corpus_manifest.get("manifest_sha256")
    if not corpus_manifest_sha:
        raise PackContractError("corpus manifest has no self-hash")
    manifest_body = dict(corpus_manifest)
    manifest_body.pop("manifest_sha256", None)
    actual_manifest_sha = hashlib.sha256(json.dumps(manifest_body, sort_keys=True).encode()).hexdigest()
    if actual_manifest_sha != corpus_manifest_sha:
        raise PackContractError("corpus manifest self-hash is invalid")
    if config["data"].get("manifest_sha256") is not None and config["data"]["manifest_sha256"] != corpus_manifest_sha:
        raise PackContractError("corpus manifest SHA256 does not match configs/shinra.yaml")
    output_root = output_dir.parent
    root_manifest = output_root / "manifest.json"
    if root_manifest.exists():
        existing = json.loads(root_manifest.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") != corpus_manifest_sha:
            raise PackContractError("packed output root already belongs to a different corpus manifest")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PackContractError(f"refusing to overwrite non-empty packed split directory: {output_dir}")
    tokenizer = load_shinra_tokenizer(tokenizer_path, config)
    bundle_sha = tokenizer.shinra_bundle_sha256
    bos_id, end_id, pad_id = resolve_pretrain_special_ids(tokenizer, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    shard_index = documents = tokens_in = tokens_out = padded_slots = 0
    shard_paths = []
    for record in tqdm(iter_clean_records(input_dir, required_split, corpus_manifest), desc="pack"):
        if plain_text(record["text"], config) != record["text"]:
            raise PackContractError("cleaned corpus is not NFC-normalized")
        ids = encode_text(tokenizer, record["text"], config)
        if not ids:
            raise PackContractError("tokenizer produced an empty sequence")
        documents += 1
        tokens_in += len(ids)
        document_rows = pack_token_stream(
            [wrap_pretrain_document(ids, bos_id, end_id, config)], sequence_length, pad_id, config
        )
        for row in document_rows:
            row["tokenizer_bundle_sha256"] = bundle_sha
        rows.extend(document_rows)
        if smoke_test_documents is not None and documents >= smoke_test_documents:
            break
        if len(rows) * sequence_length >= shard_tokens:
            shard_name, shard_tokens_actual, shard_slots = _flush_rows(
                rows, output_dir, shard_index, corpus_manifest_sha, sequence_length
            )
            shard_paths.append(shard_name)
            tokens_out += shard_tokens_actual
            padded_slots += shard_slots
            shard_index += 1
            rows = []
    if rows:
        shard_name, shard_tokens_actual, shard_slots = _flush_rows(
            rows, output_dir, shard_index, corpus_manifest_sha, sequence_length
        )
        shard_paths.append(shard_name)
        tokens_out += shard_tokens_actual
        padded_slots += shard_slots
        shard_index += 1
    if not documents:
        raise PackContractError("cannot publish an empty packed split")
    if file_sha256(input_dir / f"{required_split}.jsonl") != corpus_manifest["files"][f"{required_split}.jsonl"]:
        raise PackContractError("corpus split changed during packing")
    report = {"packer": PACKER_VERSION, "split": required_split, "documents": documents,
              "smoke_test_only": smoke_test_documents is not None,
              "tokens_in": tokens_in, "tokens_out": tokens_out, "sequence_length": sequence_length,
              "shards": shard_index, "shard_files": shard_paths, "padded_slots": padded_slots,
              "packing_efficiency": tokens_out / max(padded_slots, 1)}
    (output_dir / "pack_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    packed_manifest_path = output_dir / "manifest.json"
    packed_manifest = {"schema": "nullxes.shinra.packed.v2", "source_manifest_sha256": corpus_manifest_sha,
                       "tokenizer_bundle_sha256": bundle_sha,
                       "smoke_test_only": smoke_test_documents is not None,
                       "split": required_split, "shard_files": shard_paths,
                       "shard_sha256": {name: file_sha256(output_dir / name)
                                         for name in shard_paths}, "report": report}
    packed_manifest["manifest_sha256"] = hashlib.sha256(json.dumps(packed_manifest, sort_keys=True).encode()).hexdigest()
    packed_manifest_path.write_text(json.dumps(packed_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not root_manifest.exists():
        temporary = output_root / ".manifest.json.tmp"
        temporary.write_text(json.dumps(corpus_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(root_manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack audited SHINRA corpus Parquet into model sequences")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--shard-tokens", type=int, default=None)
    parser.add_argument("--split", choices=("train", "validation", "blind"), required=True)
    parser.add_argument("--config", default="configs/shinra.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    sequence_length = args.sequence_length or int(config["model"]["train_sequence_length"])
    shard_tokens = args.shard_tokens or int(config["data"]["packed_shard_tokens"])
    result = pack_documents(Path(args.input_dir), Path(args.tokenizer), Path(args.output_dir), sequence_length,
                            shard_tokens, config_path=Path(args.config), required_split=args.split)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
