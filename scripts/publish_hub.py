"""Disabled publication entrypoint for the SHINRA Native offline policy."""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "UNAVAILABLE: SHINRA Hub publication is disabled by project policy; "
        "no credentials were read and no network request was made"
    )


if __name__ == "__main__":
    main()
