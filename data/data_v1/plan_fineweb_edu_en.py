"""Frozen FineWeb-Edu EN acquisition plan. No network. No moving HEAD."""

from __future__ import annotations

from typing import Any

SOURCE_ID = "fineweb-edu-en"
REPOSITORY = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
UPSTREAM_VERSION = "v1.4.0"
UPSTREAM_PATH = "sample/10BT/013_00000.parquet"
FILENAME = "013_00000.parquet"
RAW_FORMAT = "parquet"
STRATEGY = "single_frozen_file"
BOUND_BYTES = 4_294_967_296
EXPECTED_RAW_SHA256 = "sha256:b393f51fefab26cd6f4c8f65707c1924f6666c4961a0ebebe04bb57f7ec832de"
DOMAIN = "general"


def frozen_plan() -> dict[str, Any]:
    return {
        "source_id": SOURCE_ID,
        "repository": REPOSITORY,
        "revision": REVISION,
        "upstream_version": UPSTREAM_VERSION,
        "path": UPSTREAM_PATH,
        "filename": FILENAME,
        "format": RAW_FORMAT,
        "strategy": STRATEGY,
        "bound_bytes": BOUND_BYTES,
        "expected_raw_sha256": EXPECTED_RAW_SHA256,
        "domain": DOMAIN,
    }
