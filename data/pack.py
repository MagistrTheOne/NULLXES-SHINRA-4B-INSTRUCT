"""Pack tokenized documents into fixed-length SHINRA v2 pretrain sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from tokenizer.special_tokens import (
    BOS,
    END_OF_TEXT,
    PAD,
    PRETRAIN_BODY_FORBIDDEN_IDS,
    PRETRAIN_BOS_ID,
    PRETRAIN_END_OF_TEXT_ID,
    PRETRAIN_EOT_ID,
    PRETRAIN_FORBIDDEN_IDS,
    PRETRAIN_PAD_ID,
)

PACKER_VERSION = "shinra-v2-pack.v1"


class PackContractError(RuntimeError):
    pass


def iter_parquet_texts(input_dir: Path):
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards in {input_dir}")
    for file in files:
        table = pq.read_table(file, columns=["text"])
        for text in table.column("text").to_pylist():
            if isinstance(text, str) and text:
                yield text


def resolve_pretrain_special_ids(tokenizer) -> tuple[int, int, int]:
    unk_id = tokenizer.unk_token_id
    bos_id = tokenizer.convert_tokens_to_ids(BOS)
    end_id = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    pad_id = tokenizer.convert_tokens_to_ids(PAD)
    eos_id = tokenizer.eos_token_id
    if bos_id in (None, unk_id) or bos_id != PRETRAIN_BOS_ID:
        raise PackContractError(f"BOS id {bos_id} is not {PRETRAIN_BOS_ID}")
    if pad_id in (None, unk_id) or pad_id != PRETRAIN_PAD_ID:
        raise PackContractError(f"PAD id {pad_id} is not {PRETRAIN_PAD_ID}")
    if end_id in (None, unk_id) or end_id != PRETRAIN_END_OF_TEXT_ID:
        raise PackContractError(f"END_OF_TEXT id {end_id} is not {PRETRAIN_END_OF_TEXT_ID}")
    if end_id == eos_id or end_id == PRETRAIN_EOT_ID:
        raise PackContractError("END_OF_TEXT must not alias eos/EOT")
    return bos_id, end_id, pad_id


def assert_pretrain_body_clean(body_ids: list[int]) -> None:
    overlap = PRETRAIN_BODY_FORBIDDEN_IDS.intersection(body_ids)
    if overlap:
        raise PackContractError(f"forbidden special ids in document body: {sorted(overlap)}")


def assert_packed_ids_clean(ids: list[int]) -> None:
    overlap = PRETRAIN_FORBIDDEN_IDS.intersection(ids)
    if overlap:
        raise PackContractError(f"forbidden special ids in packed sequence: {sorted(overlap)}")


def wrap_pretrain_document(body_ids: list[int], bos_id: int = PRETRAIN_BOS_ID, end_id: int = PRETRAIN_END_OF_TEXT_ID) -> list[int]:
    if not body_ids:
        raise PackContractError("empty document")
    assert_pretrain_body_clean(body_ids)
    wrapped = [bos_id, *body_ids, end_id]
    assert_packed_ids_clean(wrapped)
    return wrapped


def _row(seq: list[int], pad_id: int, padded: bool) -> dict:
    assert_packed_ids_clean(seq)
    if padded:
        mask = [0 if tok == pad_id else 1 for tok in seq]
        labels = [tok if m else -100 for tok, m in zip(seq, mask)]
    else:
        mask = [1] * len(seq)
        labels = list(seq)
    return {"input_ids": seq, "labels": labels, "attention_mask": mask}


def pack_token_stream(
    wrapped_docs: list[list[int]],
    sequence_length: int,
    pad_id: int = PRETRAIN_PAD_ID,
) -> list[dict]:
    if sequence_length < 2:
        raise PackContractError("sequence_length must be at least 2")
    buf: list[int] = []
    rows: list[dict] = []
    for piece in wrapped_docs:
        if not piece:
            raise PackContractError("empty wrapped document")
        assert_packed_ids_clean(piece)
        buf.extend(piece)
        while len(buf) >= sequence_length:
            seq = buf[:sequence_length]
            buf = buf[sequence_length:]
            rows.append(_row(seq, pad_id, padded=False))
    if buf:
        while len(buf) < sequence_length:
            buf.append(pad_id)
        rows.append(_row(buf[:sequence_length], pad_id, padded=True))
    return rows


def pack_documents(
    input_dir: Path,
    tokenizer_path: Path,
    output_dir: Path,
    sequence_length: int,
    shard_tokens: int = 50_000_000,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    bos_id, end_id, pad_id = resolve_pretrain_special_ids(tokenizer)

    output_dir.mkdir(parents=True, exist_ok=True)
    buf: list[int] = []
    packed_rows: list[dict] = []
    shard_idx = 0
    n_docs = 0
    n_tokens_in = 0
    n_tokens_out = 0

    def flush_shard(force: bool = False) -> None:
        nonlocal packed_rows, shard_idx, n_tokens_out
        if not packed_rows:
            return
        if not force and len(packed_rows) * sequence_length < shard_tokens:
            return
        ds = Dataset.from_list(packed_rows)
        path = output_dir / f"packed-{shard_idx:05d}.parquet"
        ds.to_parquet(str(path))
        n_tokens_out += len(packed_rows) * sequence_length
        packed_rows = []
        shard_idx += 1

    for text in tqdm(iter_parquet_texts(input_dir), desc="pack"):
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not ids:
            continue
        n_docs += 1
        n_tokens_in += len(ids)
        piece = wrap_pretrain_document(ids, bos_id=bos_id, end_id=end_id)
        buf.extend(piece)
        while len(buf) >= sequence_length:
            seq = buf[:sequence_length]
            buf = buf[sequence_length:]
            packed_rows.append(_row(seq, pad_id, padded=False))
            flush_shard(force=False)

    if buf:
        while len(buf) < sequence_length:
            buf.append(pad_id)
        packed_rows.append(_row(buf[:sequence_length], pad_id, padded=True))
    flush_shard(force=True)

    report = {
        "packer": PACKER_VERSION,
        "documents": n_docs,
        "tokens_in": n_tokens_in,
        "tokens_out": n_tokens_out,
        "sequence_length": sequence_length,
        "shards": shard_idx,
        "packing_efficiency": n_tokens_out / max(n_tokens_in, 1),
        "wrap": "bos+body+end_of_text",
    }
    (output_dir / "pack_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack SHINRA v2 documents into training sequences")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--shard-tokens", type=int, default=50_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = pack_documents(
        input_dir=Path(args.input_dir),
        tokenizer_path=Path(args.tokenizer),
        output_dir=Path(args.output_dir),
        sequence_length=args.sequence_length,
        shard_tokens=args.shard_tokens,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
