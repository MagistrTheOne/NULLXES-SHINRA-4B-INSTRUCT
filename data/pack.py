"""Pack tokenized documents into fixed-length training sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from tokenizer.special_tokens import BOS, DOCUMENT, END_OF_TEXT


def iter_parquet_texts(input_dir: Path):
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet shards in {input_dir}")
    for file in files:
        table = pq.read_table(file, columns=["text"])
        for text in table.column("text").to_pylist():
            if isinstance(text, str) and text:
                yield text


def pack_documents(
    input_dir: Path,
    tokenizer_path: Path,
    output_dir: Path,
    sequence_length: int,
    eos_between: bool = True,
    shard_tokens: int = 50_000_000,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    bos_id = tokenizer.convert_tokens_to_ids(BOS)
    document_id = tokenizer.convert_tokens_to_ids(DOCUMENT)
    end_id = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    if bos_id is None or bos_id == tokenizer.unk_token_id:
        bos_id = tokenizer.bos_token_id
    if document_id is None or document_id == tokenizer.unk_token_id:
        document_id = bos_id
    if end_id is None or end_id == tokenizer.unk_token_id:
        end_id = tokenizer.eos_token_id

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
        piece = [bos_id, document_id] + ids
        if eos_between:
            piece.append(end_id)
        buf.extend(piece)
        while len(buf) >= sequence_length:
            seq = buf[:sequence_length]
            buf = buf[sequence_length:]
            packed_rows.append(
                {
                    "input_ids": seq,
                    "labels": seq.copy(),
                    "attention_mask": [1] * sequence_length,
                }
            )
            flush_shard(force=False)

    if buf:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end_id
        while len(buf) < sequence_length:
            buf.append(pad_id)
        labels = buf[:sequence_length]
        mask = [1 if tok != pad_id else 0 for tok in labels]
        label_out = [tok if m else -100 for tok, m in zip(labels, mask)]
        packed_rows.append({"input_ids": labels, "labels": label_out, "attention_mask": mask})
    flush_shard(force=True)

    report = {
        "documents": n_docs,
        "tokens_in": n_tokens_in,
        "tokens_out": n_tokens_out,
        "sequence_length": sequence_length,
        "shards": shard_idx,
        "packing_efficiency": n_tokens_out / max(n_tokens_in, 1),
    }
    (output_dir / "pack_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack SHINRA documents into training sequences")
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
