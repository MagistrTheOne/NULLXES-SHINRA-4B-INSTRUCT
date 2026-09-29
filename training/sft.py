"""Stage 2 — instruction-tune NULLXES SHINRA-4B-INSTRUCT from BASE. ARCHIVE (not the V2 drum)."""

from __future__ import annotations

from .arguments import add_common_args, build_train_config
from .trainer import run_lm_training


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SFT NULLXES SHINRA-4B-INSTRUCT (ARCHIVE)")
    add_common_args(parser)
    parser.add_argument("--base-from", required=True, help="Path to SHINRA-4B-BASE checkpoint")
    parser.set_defaults(train_config=None)
    args = parser.parse_args()
    if not args.train_config:
        raise SystemExit("ARCHIVE module: pass --train-config explicitly (A100 recipes removed from active tree)")
    cfg = build_train_config("sft", args)
    run_lm_training(cfg)


if __name__ == "__main__":
    main()
