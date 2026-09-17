"""Stage 2 — instruction-tune NULLXES SHINRA-4B-INSTRUCT from BASE."""

from __future__ import annotations

from .arguments import add_common_args, build_train_config
from .trainer import run_lm_training


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SFT NULLXES SHINRA-4B-INSTRUCT")
    add_common_args(parser)
    parser.add_argument("--base-from", required=True, help="Path to SHINRA-4B-BASE checkpoint")
    parser.set_defaults(train_config="configs/sft_a100.yaml")
    args = parser.parse_args()
    cfg = build_train_config("sft", args)
    run_lm_training(cfg)


if __name__ == "__main__":
    main()
