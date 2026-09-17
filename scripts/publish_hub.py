"""Publish a SHINRA checkpoint to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

from .export_hf_bundle import export_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload NULLXES SHINRA to the Hub")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-id", required=True, help="e.g. NULLXES/SHINRA-4B-INSTRUCT")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--commit-message", default="Publish NULLXES SHINRA-4B-INSTRUCT")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    export_code(checkpoint)
    private = not args.public
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    create_repo(args.repo_id, private=private, exist_ok=True, token=token, repo_type="model")
    api.upload_folder(
        folder_path=str(checkpoint),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        ignore_patterns=["*.pt", "optimizer.bin", "rng_state*", "scheduler.bin"],
    )
    print(f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
