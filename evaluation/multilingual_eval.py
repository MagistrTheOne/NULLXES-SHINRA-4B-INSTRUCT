"""Multilingual eval slice. Expand task list as SHINRA checkpoints land."""

from __future__ import annotations

from evaluation.harness import main as harness_main

MULTILINGUAL_TASKS = ["mmlu", "hellaswag"]


def main() -> None:
    import sys

    if "--tasks" not in sys.argv:
        sys.argv.extend(["--tasks", *MULTILINGUAL_TASKS])
    harness_main()


if __name__ == "__main__":
    main()
