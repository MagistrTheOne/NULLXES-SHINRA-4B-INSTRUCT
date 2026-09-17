"""Download + clean + pack the SHINRA pretrain mix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clean import run_clean
from .pack import pack_documents
from .sources import PRETRAIN_MIX
from .stats import write_corpus_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SHINRA pretrain shards")
    parser.add_argument("--work-dir", default="data")
    parser.add_argument("--tokenizer", default="tokenizer/artifacts")
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument("--skip-pack", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work = Path(args.work_dir)
    clean_dir = work / "clean" / "pretrain"
    packed_dir = work / "packed" / "pretrain"
    report = run_clean(clean_dir, sources=PRETRAIN_MIX, max_docs_per_source=args.max_docs_per_source)
    stats = write_corpus_stats(clean_dir, clean_dir / "corpus_stats.json")
    result = {"clean": report, "stats": stats}
    if not args.skip_pack:
        result["pack"] = pack_documents(
            input_dir=clean_dir,
            tokenizer_path=Path(args.tokenizer),
            output_dir=packed_dir,
            sequence_length=args.sequence_length,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
