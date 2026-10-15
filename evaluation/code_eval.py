"""Code eval slice: HumanEval + MBPP via the shared harness."""

from __future__ import annotations

from evaluation.harness import main as harness_main
from configs.loader import load_config

CODE_TASKS = load_config()["evaluation"]["benchmark_tasks"]["code"]


def main() -> None:
    import sys

    if "--tasks" not in sys.argv:
        sys.argv.extend(["--tasks", *CODE_TASKS])
    harness_main()


if __name__ == "__main__":
    main()
