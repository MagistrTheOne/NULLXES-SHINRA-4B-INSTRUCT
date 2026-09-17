"""Build packed SFT conversations with assistant-only loss masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from tokenizer.special_tokens import ASSISTANT, EOT
from .sources import SFT_MIX


def _as_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    if isinstance(row.get("messages"), list):
        messages = []
        for msg in row["messages"]:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or msg.get("from")
            content = msg.get("content") or msg.get("value") or ""
            if role in {"system", "user", "assistant", "tool", "code"} and content:
                messages.append({"role": str(role), "content": str(content)})
        return messages or None
    if "instruction" in row and "output" in row:
        messages = []
        if row.get("system"):
            messages.append({"role": "system", "content": str(row["system"])})
        user = str(row["instruction"])
        if row.get("input"):
            user = user + "\n" + str(row["input"])
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": str(row["output"])})
        return messages
    if "prompt" in row and "response" in row:
        return [
            {"role": "user", "content": str(row["prompt"])},
            {"role": "assistant", "content": str(row["response"])},
        ]
    if "conversations" in row and isinstance(row["conversations"], list):
        mapping = {"human": "user", "gpt": "assistant", "bot": "assistant"}
        messages = []
        for msg in row["conversations"]:
            role = mapping.get(str(msg.get("from", "")).lower(), msg.get("from"))
            content = msg.get("value") or msg.get("content") or ""
            if role and content:
                messages.append({"role": str(role), "content": str(content)})
        return messages or None
    return None


def mask_assistant_labels(tokenizer, input_ids: list[int]) -> list[int]:
    assistant_id = tokenizer.convert_tokens_to_ids(ASSISTANT)
    end_id = tokenizer.convert_tokens_to_ids(EOT)
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i] == assistant_id:
            j = i + 1
            while j < len(input_ids) and input_ids[j] != end_id:
                labels[j] = input_ids[j]
                j += 1
            if j < len(input_ids) and input_ids[j] == end_id:
                labels[j] = end_id
                i = j + 1
                continue
            i = j
            continue
        i += 1
    return labels


def build_sft(
    tokenizer_path: Path,
    output_dir: Path,
    sequence_length: int,
    max_docs_per_source: int | None,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    stats = {"kept": 0, "dropped": 0, "sources": {}}
    for name, spec in SFT_MIX.items():
        load_kwargs = {"path": spec["hf_id"], "split": spec.get("split", "train"), "streaming": True}
        dataset = load_dataset(**load_kwargs)
        kept = 0
        for i, row in enumerate(tqdm(dataset, desc=name)):
            if max_docs_per_source is not None and i >= max_docs_per_source:
                break
            messages = _as_messages(row)
            if not messages:
                stats["dropped"] += 1
                continue
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=sequence_length)
            if len(ids) < 16:
                stats["dropped"] += 1
                continue
            labels = mask_assistant_labels(tokenizer, ids)
            if all(x == -100 for x in labels):
                stats["dropped"] += 1
                continue
            pad_len = sequence_length - len(ids)
            attn = [1] * len(ids) + [0] * pad_len
            ids = ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            rows.append({"input_ids": ids, "labels": labels, "attention_mask": attn, "source": name})
            kept += 1
            stats["kept"] += 1
            if len(rows) >= 4096:
                _flush(rows, output_dir)
                rows = []
        stats["sources"][name] = kept
    if rows:
        _flush(rows, output_dir)
    (output_dir / "sft_report.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


_shard = 0


def _flush(rows: list[dict], output_dir: Path) -> None:
    global _shard
    Dataset.from_list(rows).to_parquet(str(output_dir / f"sft-{_shard:05d}.parquet"))
    _shard += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SHINRA SFT shards")
    parser.add_argument("--tokenizer", default="tokenizer/artifacts")
    parser.add_argument("--output-dir", default="data/packed/sft")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    args = parser.parse_args()
    report = build_sft(
        Path(args.tokenizer),
        Path(args.output_dir),
        args.sequence_length,
        args.max_docs_per_source,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
