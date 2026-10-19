"""Disabled legacy corpus builder that depended on external source acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from configs.loader import CONFIG_PATH


class CorpusBuildError(RuntimeError):
    """The legacy source-based corpus build is not admitted."""


def build_corpus(tokenizer_path: str | Path, output_dir: str | Path, *,
                 config_path: str | Path = CONFIG_PATH, resume: bool = False,
                 minimum_free_gb: float | None = None) -> dict[str, Any]:
    raise CorpusBuildError(
        "UNAVAILABLE: legacy source-based corpus build is disabled; "
        "use data.build_synthetic_corpus with NULLXES Synthetic Engine records only; no network request was made"
    )


def main() -> None:
    raise SystemExit(
        "UNAVAILABLE: legacy source-based corpus build is disabled; "
        "only the NULLXES Synthetic Engine pipeline is admitted; no network access is permitted"
    )


if __name__ == "__main__":
    main()
