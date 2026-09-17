"""SHINRA tokenizer statistics: DNA + mix plan + corpus friendship + trained fertility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .corpus_friendship import analyze_friendship, write_friendship_report
from .train_tokenizer import _iter_text_files, evaluate_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="SHINRA tokenizer statistics report")
    parser.add_argument("--tokenizer-dir", default="tokenizer/artifacts")
    parser.add_argument("--corpus", nargs="+", default=None)
    parser.add_argument("--parquet-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    tok_dir = Path(args.tokenizer_dir)
    corpus_path = None
    if args.corpus:
        first = Path(args.corpus[0])
        corpus_path = first if first.is_file() else next(
            (p for p in first.rglob("*.txt") if p.is_file()), first
        )
    parquet_dir = Path(args.parquet_dir) if args.parquet_dir else None
    out = Path(args.output) if args.output else Path("tokenizer/tokenizer_friendship.json")
    report = analyze_friendship(
        corpus_path=corpus_path,
        parquet_dir=parquet_dir,
        tokenizer_dir=tok_dir if tok_dir.exists() else None,
    )
    if args.corpus and (tok_dir / "tokenizer.json").exists():
        files = _iter_text_files([Path(p) for p in args.corpus])
        report["trained_holdout"] = evaluate_tokenizer(tok_dir, files, tok_dir / "tokenizer_stats.json")
    write_friendship_report(report, out)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
