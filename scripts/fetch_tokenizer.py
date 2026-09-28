"""Download SHINRA tokenizer files only. Never safetensors / S0.4.1 weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "MagistrTheOne/NULLXES-SHINRA-4B-INSTRUCT"
ALLOW = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer.vocab",
)


def fetch_tokenizer(dest: Path, repo: str = DEFAULT_REPO) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        allow_patterns=list(ALLOW),
        ignore_patterns=["*.safetensors", "*.bin", "exp/**", "model.safetensors"],
    )
    forbidden = list(dest.rglob("*.safetensors"))
    if forbidden:
        raise RuntimeError(f"refusing tokenizer fetch that pulled weights: {forbidden}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch tokenizer only from Hub INSTRUCT (no weights)")
    parser.add_argument("--dest", type=Path, default=Path("tokenizer/artifacts"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args()
    path = fetch_tokenizer(args.dest, args.repo)
    print(path)


if __name__ == "__main__":
    main()
