"""Stage 1 — pretrain NULLXES SHINRA-4B-BASE."""

from __future__ import annotations

from .arguments import add_common_args, build_train_config
from .trainer import run_lm_training


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain NULLXES SHINRA-4B-BASE")
    add_common_args(parser)
    parser.set_defaults(train_config="configs/pretrain_a100.yaml")
    args = parser.parse_args()
    cfg = build_train_config("pretrain", args)
    run_lm_training(cfg)


if __name__ == "__main__":
    main()
