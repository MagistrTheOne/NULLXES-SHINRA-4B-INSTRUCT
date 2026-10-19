"""Disabled legacy external-corpus acquisition entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.loader import CONFIG_PATH


class SourceAcquisitionError(RuntimeError):
    """External source acquisition is not admitted in SHINRA Native."""


def acquire_source(source_id: str, output_dir: str | Path, *, config_path: str | Path = CONFIG_PATH,
                   admission: str = "audit", max_documents: int | None = None,
                   max_target_tokens: int | None = None, tokenizer_path: str | Path | None = None,
                   resume: bool = False, token: str | None = None,
                   minimum_free_gb: float | None = None) -> dict[str, Any]:
    raise SourceAcquisitionError(
        "UNAVAILABLE: external corpus acquisition is disabled by SHINRA Native policy; "
        "use only NULLXES Synthetic Engine records; no network request was made"
    )


def main() -> None:
    raise SystemExit(
        "UNAVAILABLE: external corpus acquisition is disabled by SHINRA Native policy; "
        "no Hugging Face or other network access is permitted"
    )


if __name__ == "__main__":
    main()
