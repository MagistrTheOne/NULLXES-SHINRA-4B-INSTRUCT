"""Copy SHINRA modeling sources into a Hugging Face checkpoint directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MODEL_FILES = [
    "configuration_shinra.py",
    "modeling_shinra.py",
    "modeling_attn.py",
    "modeling_mlp.py",
    "modeling_norm.py",
]


def export_code(checkpoint_dir: Path, repo_root: Path | None = None) -> None:
    root = repo_root or Path(__file__).resolve().parents[1]
    src = root / "model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        shutil.copy2(src / name, checkpoint_dir / name)
    card = root / "docs" / "MODEL_CARD.md"
    if card.exists():
        shutil.copy2(card, checkpoint_dir / "README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle SHINRA Python sources into a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    export_code(Path(args.checkpoint))
    print(f"Copied SHINRA modeling files into {args.checkpoint}")


if __name__ == "__main__":
    main()
