"""S0–S2 rolling drum CLI: produce N → SHA → train once → consume/delete → N+1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def unhide_cuda() -> None:
    val = os.environ.get("CUDA_VISIBLE_DEVICES")
    if val is None or str(val).strip() == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


unhide_cuda()

from data.rolling_drum import run_rolling_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="SHINRA v2 S0–S2 rolling shard drum")
    parser.add_argument("--config", default="configs/shinra_4b.yaml")
    parser.add_argument("--stage-config", required=True)
    parser.add_argument("--storage-config", default="configs/storage_g4.yaml")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="tokenizer/artifacts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--records-per-shard", type=int, default=4096)
    parser.add_argument("--max-shards", type=int, default=1024)
    parser.add_argument("--fake-tokenizer", action="store_true")
    parser.add_argument("--heldout-records", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--attention-implementation", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stop-after-steps", type=int, default=None)
    parser.add_argument("--record-consumed-trace", action="store_true")
    parser.add_argument("--keep-consumed-shards", action="store_true")
    args = parser.parse_args()
    args.delete_consumed = not args.keep_consumed_shards
    result = run_rolling_stage(args)
    print(json.dumps({k: v for k, v in result.items() if k != "consumed_trace"}, indent=2))


if __name__ == "__main__":
    main()
