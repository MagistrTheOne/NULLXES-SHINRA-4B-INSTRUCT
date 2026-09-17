"""Convert a trained SentencePiece model to Hugging Face tokenizer.json."""

from __future__ import annotations

import argparse
from pathlib import Path

from .train_tokenizer import convert_spm_to_hf, evaluate_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SHINRA tokenizer.model → tokenizer.json")
    parser.add_argument("--model", required=True, help="Path to tokenizer.model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-corpus", nargs="*", default=[])
    args = parser.parse_args()
    model = Path(args.model).resolve()
    output = Path(args.output_dir).resolve()
    convert_spm_to_hf(model, output)
    if args.eval_corpus:
        from pathlib import Path as P

        evaluate_tokenizer(output, [P(p) for p in args.eval_corpus], output / "tokenizer_stats.json")
    print(f"Wrote Hugging Face tokenizer bundle to {output}")


if __name__ == "__main__":
    main()
