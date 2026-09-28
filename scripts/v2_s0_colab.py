"""Colab G4 S0 entry: tokenizer-only fetch, 20M-scale controller, status.json."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


def assert_g4() -> str:
    if not torch.cuda.is_available():
        raise SystemExit("S0 train needs CUDA G4")
    name = torch.cuda.get_device_name(0)
    if "2080" in name:
        raise SystemExit("RTX 2080 is forbidden")
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="SHINRA v2 S0 on Colab G4")
    parser.add_argument("--fetch-tokenizer", action="store_true")
    parser.add_argument("--tokenizer", default="tokenizer/artifacts")
    parser.add_argument("--corpus-dir", default="/content/shinra_scratch/corpus")
    parser.add_argument("--output-dir", default="/content/shinra_scratch/s0")
    parser.add_argument("--records-per-shard", type=int, default=8192)
    parser.add_argument("--max-shards", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--fake-tokenizer", action="store_true")
    parser.add_argument("--dry-hardware", action="store_true")
    args = parser.parse_args()

    hw = yaml.safe_load(Path("configs/colab.yaml").read_text(encoding="utf-8"))
    if hw["gpu_class"] != "G4":
        raise SystemExit("configs/colab.yaml is not G4")
    if hw["model"]["attention_implementation"] != "sdpa":
        raise SystemExit("G4 attention must be sdpa")
    if int(hw["training"]["max_tokens"]) != 0:
        raise SystemExit("colab.yaml must keep max_tokens: 0 (hardware profile)")

    if args.dry_hardware:
        print(json.dumps({"gpu_class": "G4", "disk_ceiling_gb": hw["storage"]["disk_ceiling_gb"], "attn": "sdpa"}))
        return

    name = assert_g4()
    print(json.dumps({"cuda_device": name, "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3}))

    if args.fetch_tokenizer and not args.fake_tokenizer:
        subprocess.check_call([sys.executable, str(Path("scripts/fetch_tokenizer.py")), "--dest", args.tokenizer])

    cmd = [
        sys.executable,
        str(Path("scripts/v2_stage_run.py")),
        "--config",
        "configs/shinra_4b.yaml",
        "--stage-config",
        "configs/stages/s0_bringup.yaml",
        "--storage-config",
        "configs/storage_g4.yaml",
        "--corpus-dir",
        args.corpus_dir,
        "--tokenizer",
        args.tokenizer,
        "--output-dir",
        args.output_dir,
        "--records-per-shard",
        str(args.records_per_shard),
        "--max-shards",
        str(args.max_shards),
        "--attention-implementation",
        "sdpa",
    ]
    if args.max_tokens is not None:
        cmd.extend(["--max-tokens", str(args.max_tokens)])
    if args.fake_tokenizer:
        cmd.append("--fake-tokenizer")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(cmd, env=env)
    status = Path(args.output_dir) / "run" / "status.json"
    if status.exists():
        print(status.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
