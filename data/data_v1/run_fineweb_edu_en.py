"""FineWeb-Edu EN frozen shard: acquire → adapt → materialize. No training. No S1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data.data_v1.acquire_hf import acquire_fineweb_edu_en
from data.data_v1.adapt_parquet import adapt_parquet_to_jsonl
from data.data_v1.materialize import PRODUCTION_SCRATCH, materialize
from data.data_v1.plan_fineweb_edu_en import REVISION, SOURCE_ID
from data.data_v1.acquisition import PRODUCTION_ACQUISITION_ROOT
from data.data_v1.sources import load_allowlist


def run_fineweb_edu_en_slice(
    *,
    acquisition_root: str | Path | None = None,
    materialize_root: str | Path | None = None,
    fetch_file: Any | None = None,
    free_bytes: int | None = None,
    extra_s0: tuple[Path, ...] = (),
    extra_materialize: tuple[Path, ...] = (),
) -> dict[str, Any]:
    allowlist = load_allowlist()
    acq_root = Path(acquisition_root) if acquisition_root is not None else PRODUCTION_ACQUISITION_ROOT
    mat_root = Path(materialize_root) if materialize_root is not None else PRODUCTION_SCRATCH
    acquired = acquire_fineweb_edu_en(
        acquisition_root=acq_root,
        allowlist=allowlist,
        fetch_file=fetch_file,
        free_bytes=free_bytes,
        extra_s0=extra_s0,
        extra_materialize=extra_materialize,
    )
    adapter_jsonl = mat_root / SOURCE_ID / "from-parquet.adapter.jsonl"
    adapted = adapt_parquet_to_jsonl(
        acquired["path"],
        adapter_jsonl,
        source_id=SOURCE_ID,
        snapshot=REVISION,
    )
    materialized = materialize(
        adapted["path"],
        SOURCE_ID,
        scratch_root=mat_root,
        allowlist=allowlist,
        free_bytes=free_bytes,
        extra_forbidden=extra_s0,
    )
    adapter_jsonl.unlink(missing_ok=True)
    return {
        "acquired": {k: (str(v) if isinstance(v, Path) else v) for k, v in acquired.items() if k != "receipt"},
        "adapted_records": adapted["records"],
        "materialized": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in materialized.items()
            if k != "receipt"
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire+adapt+materialize FineWeb-Edu EN frozen shard")
    parser.add_argument("--acquisition-root", type=Path, default=None)
    parser.add_argument("--materialize-root", type=Path, default=None)
    args = parser.parse_args()
    body = run_fineweb_edu_en_slice(
        acquisition_root=args.acquisition_root,
        materialize_root=args.materialize_root,
    )
    print(json.dumps(body, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
