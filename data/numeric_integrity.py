"""Reusable numeric-preservation checks; legacy external-source cleaning is disabled."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from tokenizer.contract import normalize_text


class DataIntegrityError(ValueError):
    """Raised when text normalization changes a numeric value or policy is violated."""


NUMERIC_SPAN = re.compile(
    r"(?<![\w])[-+]?(?:\d{1,3}(?:[ ,\u00a0]\d{3})+|\d+)(?:[.,]\d+)?"
    r"(?:[eE][-+]?\d+)?(?:\s?(?:%|‰|°[CFК]?|[A-Za-zА-Яа-яЁёµμΩ]+(?:\s*/\s*[A-Za-zА-Яа-яЁёµμΩ]+)?))?"
    r"(?!\w)"
)


def numeric_spans(text: str) -> list[str]:
    """Return numeric spans in source order, preserving their exact spelling."""
    return [match.group(0) for match in NUMERIC_SPAN.finditer(text)]


def clean_text(text: str, *, normalization: str = "NFC") -> str:
    """Normalize text only when its ordered numeric spans remain byte-for-byte identical."""
    if not isinstance(text, str):
        raise DataIntegrityError("text must be a string")
    try:
        cleaned = normalize_text(text, normalization)
    except ValueError as exc:
        raise DataIntegrityError(str(exc)) from exc
    before, after = numeric_spans(text), numeric_spans(cleaned)
    if before != after:
        raise DataIntegrityError(f"numeric spans changed during normalization: before={before!r}, after={after!r}")
    return cleaned


def prepare_local_corpus(inputs: Iterable[str | Path], output_dir: str | Path, **_options) -> dict:
    """Refuse the retired cleaner before examining source paths or opening files."""
    raise DataIntegrityError(
        "UNAVAILABLE: legacy source-snapshot corpus preparation is disabled; "
        "use data.build_synthetic_corpus with the NULLXES Synthetic Engine"
    )


def main() -> None:
    raise SystemExit(
        "UNAVAILABLE: python -m data.numeric_integrity is disabled for legacy source snapshots; "
        "use python -m data.build_synthetic_corpus for the NULLXES Synthetic Engine"
    )


if __name__ == "__main__":
    main()
