"""Selection and provenance helpers for the real-corpus packing smoke run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable


def select_packable_validation_records(
    records: Iterable[dict], encode: Callable[[str], list[int]], sequence_length: int, limit: int
) -> tuple[list[tuple[dict, list[int]]], dict[str, int]]:
    if sequence_length < 3 or limit <= 0:
        raise ValueError("sequence_length must be at least 3 and limit must be positive")
    selected: list[tuple[dict, list[int]]] = []
    stats = {
        "documents_examined": 0,
        "documents_packable": 0,
        "documents_accepted": 0,
        "documents_skipped_overlength": 0,
    }
    for record in records:
        stats["documents_examined"] += 1
        token_ids = [int(token_id) for token_id in encode(record["text"])]
        if len(token_ids) + 2 > sequence_length:
            stats["documents_skipped_overlength"] += 1
            continue
        stats["documents_packable"] += 1
        if len(selected) < limit:
            selected.append((dict(record), token_ids))
    stats["documents_accepted"] = len(selected)
    if len(selected) != limit:
        raise RuntimeError(
            f"packing smoke requires {limit} eligible validation documents; "
            f"found {len(selected)} after examining {stats['documents_examined']}"
        )
    return selected, stats


def write_selected_validation_corpus(
    output_dir: Path, source_manifest: dict, source_manifest_sha256: str,
    selected_records: list[tuple[dict, list[int]]], selection_stats: dict[str, int],
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite smoke corpus directory: {output_dir}")
    if not selected_records or selection_stats.get("documents_accepted") != len(selected_records):
        raise ValueError("smoke corpus requires exactly the accepted selected records")
    output_dir.mkdir(parents=True)
    rows_path = output_dir / "validation.jsonl"
    with rows_path.open("x", encoding="utf-8", newline="\n") as stream:
        for record, _ in selected_records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = dict(source_manifest)
    manifest["files"] = dict(source_manifest["files"])
    import hashlib

    digest = hashlib.sha256()
    with rows_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    manifest["files"]["validation.jsonl"] = digest.hexdigest()
    manifest["smoke_subset"] = {
        "source_manifest_sha256": source_manifest_sha256,
        "selection_policy": "first eligible records in validation order",
        **selection_stats,
    }
    manifest_body = dict(manifest)
    manifest_body.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest_body, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
