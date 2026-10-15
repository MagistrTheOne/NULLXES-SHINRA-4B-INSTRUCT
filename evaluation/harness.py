"""lm-evaluation-harness wrapper for SHINRA checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from configs.loader import load_config


def main() -> None:
    approved = load_config()
    eval_config = approved["evaluation"]
    default_tasks = [task for group in eval_config["benchmark_tasks"].values() for task in group]
    parser = argparse.ArgumentParser(description="Run lm-eval on a SHINRA checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", nargs="+", default=default_tasks)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--output", default=f"{eval_config['report_dir']}/harness.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        raise FileNotFoundError("evaluation requires a local SHINRA checkpoint directory")

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=args.model,
        dtype=approved["precision"]["preferred"],
        trust_remote_code=False,
        batch_size=args.batch_size,
        max_length=int(eval_config["benchmark_max_length"]),
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
