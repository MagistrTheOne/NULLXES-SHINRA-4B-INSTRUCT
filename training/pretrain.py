"""Stage 1 — pretrain NULLXES SHINRA-4B-BASE. ARCHIVE (A100 chain, not the V2 drum)."""

from __future__ import annotations

from .arguments import add_common_args, build_train_config
from .trainer import run_lm_training


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Pretrain NULLXES SHINRA-4B-BASE (ARCHIVE)")
    add_common_args(parser)
    parser.set_defaults(train_config=None)
    args = parser.parse_args()
    if not args.train_config:
        raise SystemExit("ARCHIVE module: pass --train-config explicitly (A100 recipes removed from active tree)")
    cfg = build_train_config("pretrain", args)
    run_lm_training(cfg)


if __name__ == "__main__":
    main()
