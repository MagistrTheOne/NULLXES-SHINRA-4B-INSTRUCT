"""Colab G4 Phase C: pack FineWeb-Edu EN, resume S0 step-00001358, 8M honest tokens."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from data.data_v1.phase_c import (
    CEILING_HONEST_TOKENS,
    RESUME_RELATIVE,
    TARGET_HONEST_TOKENS,
    pack_from_tokenizer,
)

DRIVE = Path("/content/drive/MyDrive/NULLXES/SHINRA-v2")
SCRATCH = Path("/content/shinra_scratch")


def assert_g4() -> str:
    if not torch.cuda.is_available():
        raise SystemExit("Phase C train needs CUDA G4")
    name = torch.cuda.get_device_name(0)
    if "2080" in name:
        raise SystemExit("RTX 2080 is forbidden")
    return name


def _copy_tree(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"copy {src} -> {dest}", flush=True)
    shutil.copytree(src, dest)


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        return
    print(f"copy {src} -> {dest}", flush=True)
    shutil.copy2(src, dest)


def find_jsonl(roots: list[Path]) -> Path:
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        hits.extend(sorted(p for p in root.rglob("*.jsonl") if p.is_file() and not p.name.endswith(".tmp")))
    named = [p for p in hits if "fineweb-edu-en" in p.name]
    pool = named or hits
    if len(pool) == 1:
        return pool[0]
    if len(pool) > 1:
        return pool[0]
    raise SystemExit(f"FineWeb-Edu JSONL not found under {roots}")


def ensure_tokenizer(dest: Path) -> Path:
    if (dest / "tokenizer.json").is_file():
        return dest
    drive = DRIVE / "tokenizer_artifacts"
    if (drive / "tokenizer.json").is_file():
        dest.mkdir(parents=True, exist_ok=True)
        for item in drive.iterdir():
            if item.is_file():
                _copy_file(item, dest / item.name)
        return dest
    raise SystemExit("tokenizer.json missing on scratch and Drive")


def ensure_s0(dest: Path) -> Path:
    marker = dest / "trainer_state.pt"
    if marker.is_file():
        return dest
    if dest.exists() and not (dest / "trainer_state.pt").is_file():
        raise SystemExit(f"incomplete S0 ckpt at {dest}")
    drive = DRIVE / RESUME_RELATIVE
    if (drive / "trainer_state.pt").is_file():
        _copy_tree(drive, dest)
        return dest
    raise SystemExit(f"S0 checkpoint missing: {dest} and {drive}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SHINRA v2 Phase C on Colab G4")
    parser.add_argument("--tokenizer", default="tokenizer/artifacts")
    parser.add_argument("--corpus-dir", default=str(SCRATCH / "corpus"))
    parser.add_argument("--output-dir", default=str(SCRATCH / "c"))
    parser.add_argument("--s0-ckpt", default=str(SCRATCH / RESUME_RELATIVE))
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--max-tokens", type=int, default=TARGET_HONEST_TOKENS)
    parser.add_argument("--max-steps", type=int, default=650)
    parser.add_argument("--skip-pack", action="store_true")
    args = parser.parse_args()

    hw = yaml.safe_load(Path("configs/colab.yaml").read_text(encoding="utf-8"))
    if hw["gpu_class"] != "G4":
        raise SystemExit("configs/colab.yaml is not G4")
    if hw["model"]["attention_implementation"] != "sdpa":
        raise SystemExit("G4 attention must be sdpa")

    name = assert_g4()
    print(json.dumps({"cuda_device": name, "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3, "phase": "c"}))

    tok = ensure_tokenizer(Path(args.tokenizer))
    s0 = ensure_s0(Path(args.s0_ckpt))
    jsonl = Path(args.jsonl) if args.jsonl else find_jsonl([SCRATCH / "data_v1", DRIVE / "data_v1"])
    if DRIVE in jsonl.parents and not str(jsonl).startswith(str(SCRATCH)):
        local = SCRATCH / "data_v1" / jsonl.name
        _copy_file(jsonl, local)
        jsonl = local

    pack_dir = Path(args.corpus_dir) / "c"
    packed = pack_dir / "shard-00000.bin"
    if args.skip_pack and packed.is_file():
        print("skip pack: shard-00000.bin exists")
    elif packed.is_file() and (pack_dir / "phase_c.pack.json").is_file():
        print("skip pack: phase_c.pack.json exists")
    else:
        report = pack_from_tokenizer(
            input_path=jsonl,
            output_dir=pack_dir,
            tokenizer_path=tok,
            target_honest_tokens=TARGET_HONEST_TOKENS,
            ceiling_honest_tokens=CEILING_HONEST_TOKENS,
        )
        print(json.dumps({"pack": report["status"], "honest_tokens": report["honest_tokens"], "n_sequences": report["n_sequences"]}))

    cmd = [
        sys.executable,
        str(Path("scripts/v2_stage_run.py")),
        "--config",
        "configs/shinra_4b.yaml",
        "--stage-config",
        "configs/stages/c_edu_en_pilot.yaml",
        "--storage-config",
        "configs/storage_g4.yaml",
        "--corpus-dir",
        args.corpus_dir,
        "--tokenizer",
        str(tok),
        "--output-dir",
        args.output_dir,
        "--resume-from",
        str(s0),
        "--heldout-records",
        "0",
        "--max-tokens",
        str(args.max_tokens),
        "--max-steps",
        str(args.max_steps),
        "--attention-implementation",
        "sdpa",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(cmd, env=env)
    status = Path(args.output_dir) / "run" / "status.json"
    if status.exists():
        print(status.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
