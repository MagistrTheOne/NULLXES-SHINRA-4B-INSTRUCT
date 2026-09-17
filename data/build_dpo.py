"""Build DPO preference pairs tokenized with the SHINRA chat template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from .sources import DPO_MIX


def _prompt_chosen_rejected(row: dict[str, Any]) -> tuple[str, str, str] | None:
    if row.get("chosen") and row.get("rejected") and (row.get("prompt") or row.get("question")):
        return str(row.get("prompt") or row.get("question") or ""), str(row["chosen"]), str(row["rejected"])
    if row.get("accepted") and row.get("rejected"):
        return str(row.get("prompt") or row.get("question") or ""), str(row["accepted"]), str(row["rejected"])
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if isinstance(chosen, list) and isinstance(rejected, list):
        prompt_msgs = [m for m in chosen[:-1] if isinstance(m, dict)]
        chosen_msg = chosen[-1]["content"] if chosen and isinstance(chosen[-1], dict) else None
        rejected_msg = rejected[-1]["content"] if rejected and isinstance(rejected[-1], dict) else None
        if chosen_msg and rejected_msg:
            prompt = "\n".join(str(m.get("content", "")) for m in prompt_msgs)
            return prompt, str(chosen_msg), str(rejected_msg)
    if row.get("chosen_response") and row.get("rejected_response"):
        return str(row.get("prompt") or row.get("question") or ""), str(row["chosen_response"]), str(row["rejected_response"])
    if row.get("response_j") and row.get("response_k"):
        return str(row.get("prompt") or ""), str(row["response_j"]), str(row["response_k"])
    return None


def build_dpo(tokenizer_path: Path, output_dir: Path, max_length: int, max_docs: int | None) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    stats = {"kept": 0, "dropped": 0}
    shard = 0

    def flush() -> None:
        nonlocal rows, shard
        if not rows:
            return
        Dataset.from_list(rows).to_parquet(str(output_dir / f"dpo-{shard:05d}.parquet"))
        shard += 1
        rows = []

    for name, spec in DPO_MIX.items():
        dataset = load_dataset(spec["hf_id"], split=spec.get("split", "train"), streaming=True)
        for i, row in enumerate(tqdm(dataset, desc=name)):
            if max_docs is not None and i >= max_docs:
                break
            parsed = _prompt_chosen_rejected(row)
            if parsed is None:
                stats["dropped"] += 1
                continue
            prompt, chosen, rejected = parsed
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            chosen_text = prompt_text + chosen + (tokenizer.eos_token or "")
            rejected_text = prompt_text + rejected + (tokenizer.eos_token or "")
            if len(tokenizer.encode(chosen_text, add_special_tokens=False)) > max_length:
                stats["dropped"] += 1
                continue
            rows.append(
                {
                    "prompt": prompt_text,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "source": name,
                }
            )
            stats["kept"] += 1
            if len(rows) >= 4096:
                flush()
    flush()
    (output_dir / "dpo_report.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SHINRA DPO shards")
    parser.add_argument("--tokenizer", default="tokenizer/artifacts")
    parser.add_argument("--output-dir", default="data/packed/dpo")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(build_dpo(Path(args.tokenizer), Path(args.output_dir), args.max_length, args.max_docs_per_source), indent=2))


if __name__ == "__main__":
    main()
