"""Build small SHINRA v2 synthetic pretrain fixtures. Does not emit 50M tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from random import Random

from data.synth.contamination import assert_clean_pretrain_text
from data.synth.layers import GENERATOR_IDS, GENERATORS, LAYERS
from data.synth.schema import (
    CORPUS_ID,
    LANGUAGES,
    LICENSE,
    MAX_RECORDS,
    validate_manifest,
    validate_record,
    whitespace_tokens,
)


def mix_seed(seed: int, layer: str, language: str, index: int) -> int:
    payload = f"{seed}:{layer}:{language}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def generate_record(seed: int, layer: str, language: str, index: int) -> dict:
    local = mix_seed(seed, layer, language, index)
    text, semantics = GENERATORS[layer](Random(local), language)
    record = {
        "text": text,
        "layer": layer,
        "language": language,
        "generator": GENERATOR_IDS[layer],
        "seed": local,
        "semantics": semantics,
    }
    validate_record(record)
    assert_clean_pretrain_text(record["text"])
    return record


def iter_records(seed: int, per_layer: int) -> list[dict]:
    if per_layer < 2 or per_layer % 2:
        raise ValueError("per_layer must be an even integer >= 2")
    total = per_layer * len(LAYERS)
    if total > MAX_RECORDS:
        raise ValueError(f"refusing {total} records; cap is {MAX_RECORDS} this cycle")
    out: list[dict] = []
    half = per_layer // 2
    for layer in LAYERS:
        for language in LANGUAGES:
            for index in range(half):
                out.append(generate_record(seed, layer, language, index))
    return out


def records_to_jsonl(records: list[dict]) -> bytes:
    lines = [json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for rec in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def shard_manifest(records: list[dict], corpus_seed: int, blob: bytes) -> dict:
    if not records:
        raise ValueError("empty shard")
    layer = records[0]["layer"]
    language = records[0]["language"]
    if any(r["layer"] != layer or r["language"] != language for r in records):
        raise ValueError("shard must be homogeneous")
    text_chars = sum(len(r["text"]) for r in records)
    tokens = sum(whitespace_tokens(r["text"]) for r in records)
    manifest = {
        "corpus_id": CORPUS_ID,
        "family": layer,
        "language": language,
        "seed": corpus_seed,
        "records": len(records),
        "chars": text_chars,
        "tokens": tokens,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "license": LICENSE,
        "generator": GENERATOR_IDS[layer],
    }
    validate_manifest(manifest)
    return manifest


def write_corpus(output_dir: Path, seed: int, per_layer: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = iter_records(seed, per_layer)
    shards = []
    for layer in LAYERS:
        for language in LANGUAGES:
            subset = [r for r in records if r["layer"] == layer and r["language"] == language]
            blob = records_to_jsonl(subset)
            rel = Path(layer) / f"{language}.jsonl"
            path = output_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            man = shard_manifest(subset, seed, blob)
            man_path = output_dir / layer / f"{language}.manifest.json"
            man_path.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            shards.append({"path": rel.as_posix(), "sha256": man["sha256"], **man})
    digest_src = "".join(s["sha256"] for s in shards).encode()
    index = {
        "corpus_id": CORPUS_ID,
        "license": LICENSE,
        "seed": seed,
        "per_layer": per_layer,
        "records": len(records),
        "sha256": hashlib.sha256(digest_src).hexdigest(),
        "shards": [{"path": s["path"], "family": s["family"], "language": s["language"], "sha256": s["sha256"]} for s in shards],
    }
    (output_dir / "corpus.manifest.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHINRA v2 synthetic pretrain fixtures (not 50M tokens)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-layer", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = write_corpus(args.output_dir, args.seed, args.per_layer)
    print(json.dumps({"corpus_id": index["corpus_id"], "records": index["records"], "sha256": index["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
