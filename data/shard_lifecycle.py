"""Generate → pack → SHA → optional delete scratch. Disk ceiling abort."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

from data.ledger import RunLedger
from data.pack import pack_token_stream, resolve_pretrain_special_ids, wrap_pretrain_document
from data.shards import write_shard_bin
from data.synth.stream import iter_stream_records
from tokenizer.special_tokens import PRETRAIN_PAD_ID


class DiskCeilingError(RuntimeError):
    pass


EncodeFn = Callable[[str], list[int]]


def dir_usage_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def assert_under_ceiling(path: Path, ceiling_gb: float) -> None:
    used = dir_usage_bytes(path) / (1024**3)
    if used > ceiling_gb:
        raise DiskCeilingError(f"{path} uses {used:.2f} GB > ceiling {ceiling_gb} GB")


def encode_with_tokenizer(tokenizer, text: str) -> list[int]:
    # Pretrain DNA: never TemplateProcessing / chat_template / add_special_tokens=True.
    return tokenizer.encode(text, add_special_tokens=False, truncation=False)


def build_packed_shard(
    *,
    stage: str,
    shard_index: int,
    seed: int,
    record_count: int,
    encode: EncodeFn,
    output_dir: Path,
    sequence_length: int,
    start_index: int = 0,
    bos_id: int = 1,
    end_id: int = 18,
    pad_id: int = PRETRAIN_PAD_ID,
    scratch_dir: Path | None = None,
    ceiling_root: Path | None = None,
    ceiling_gb: float | None = None,
    ledger: RunLedger | None = None,
) -> dict:
    if end_id == 2 or end_id != 18:
        raise ValueError("S0–S2 document stop must be END_OF_TEXT=18, never eos/EOT=2")
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_id = f"shard-{shard_index:05d}"
    scratch = scratch_dir or (output_dir / "_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    jsonl_path = scratch / f"{shard_id}.jsonl"
    wrapped: list[list[int]] = []
    n_docs = 0
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in iter_stream_records(stage, seed, record_count, start_index=start_index):
            ids = encode(record["text"])
            if not ids:
                continue
            wrapped.append(wrap_pretrain_document(ids, bos_id=bos_id, end_id=end_id))
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            n_docs += 1
    rows = pack_token_stream(wrapped, sequence_length=sequence_length, pad_id=pad_id)
    bin_path = output_dir / f"{shard_id}.bin"
    meta = write_shard_bin(
        rows,
        bin_path,
        stage=stage,
        shard_id=shard_id,
        pad_id=pad_id,
        extra_meta={"seed": seed, "start_index": start_index, "records": n_docs, "status": "pending"},
    )
    jsonl_path.unlink(missing_ok=True)
    if scratch.exists() and not any(scratch.iterdir()):
        scratch.rmdir()
    if ceiling_root is not None and ceiling_gb is not None:
        assert_under_ceiling(ceiling_root, ceiling_gb)
    if ledger is not None:
        ledger.append({"event": "shard_ready", "shard_id": shard_id, "produced_tokens": meta["produced_tokens"], "sha256": meta["sha256"]})
    return meta


def mark_consumed(bin_path: Path, ledger: RunLedger | None = None, delete_bin: bool = False) -> dict:
    meta_path = bin_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["status"] = "deleted" if delete_bin else "consumed"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if delete_bin:
        bin_path.unlink(missing_ok=True)
    if ledger is not None:
        ledger.append({"event": "shard_consumed", "shard_id": meta["shard_id"], "status": meta["status"]})
    return meta


def resolve_ids(tokenizer) -> tuple[int, int, int]:
    return resolve_pretrain_special_ids(tokenizer)


def rmtree_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
