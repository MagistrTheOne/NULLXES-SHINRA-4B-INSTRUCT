"""lm-evaluation-harness wrapper for SHINRA checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_TASKS = [
    "arc_challenge",
    "hellaswag",
    "mmlu",
    "gsm8k",
    "humaneval",
    "mbpp",
    "winogrande",
    "truthfulqa_mc2",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lm-eval on a SHINRA checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--output", default="evaluation/results/harness.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        batch_size=args.batch_size,
        max_length=8192,
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        limit=args.limit,
        log_samples=False,
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "results": results.get("results"),
        "n-shot": results.get("n-shot"),
        "config": {k: str(v) for k, v in (results.get("config") or {}).items()},
    }
    path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(serializable["results"], indent=2))


if __name__ == "__main__":
    main()
