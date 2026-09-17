"""Print tokenizer_stats.json and re-run holdout compression metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train_tokenizer import _iter_text_files, evaluate_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="SHINRA tokenizer statistics report")
    parser.add_argument("--tokenizer-dir", default="tokenizer/artifacts")
    parser.add_argument("--corpus", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    tok_dir = Path(args.tokenizer_dir)
    existing = tok_dir / "tokenizer_stats.json"
    if args.corpus:
        files = _iter_text_files([Path(p) for p in args.corpus])
        out = Path(args.output) if args.output else tok_dir / "tokenizer_stats.json"
        stats = evaluate_tokenizer(tok_dir, files, out)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return
    if existing.exists():
        print(existing.read_text(encoding="utf-8"))
        return
    raise SystemExit("No stats found. Pass --corpus to evaluate.")


if __name__ == "__main__":
    main()
