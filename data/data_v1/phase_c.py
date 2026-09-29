"""Phase C learning-pilot packer. CPU contract only. Train is closed.

Resume identity is S0 `s0/final/step-00001358`. Source is FineWeb-Edu EN JSONL.
Pack DNA is `[BOS=1] + body + [END_OF_TEXT=18]`. Honest tokens are labels != -100.
This module does not train, does not load Hub weights, and does not touch S1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from data.data_v1 import PhaseBError
from data.data_v1.ingest import ensure_local_path
from tokenizer.special_tokens import (
    PRETRAIN_BOS_ID,
    PRETRAIN_BODY_FORBIDDEN_IDS,
    PRETRAIN_END_OF_TEXT_ID,
    PRETRAIN_EOT_ID,
    PRETRAIN_FORBIDDEN_IDS,
    PRETRAIN_PAD_ID,
)

EncodeFn = Callable[[str], list[int]]

STAGE = "c"
SOURCE_ID = "fineweb-edu-en"
PACKER_VERSION = "shinra-v2-pack.v1"
TARGET_HONEST_TOKENS = 8_000_000
CEILING_HONEST_TOKENS = 10_000_000
SEQUENCE_LENGTH = 2048
RESUME_RELATIVE = "s0/final/step-00001358"
RESUME_STEP = 1358
TRAIN_AUTHORIZED = False
S1_STATUS = "closed"
HUB_STATUS = "closed"


class PhaseCError(ValueError):
    """Phase C pack / authorization failure."""


def assert_train_closed() -> None:
    raise PhaseCError("Phase C train is not authorized")


def resolve_corpus_jsonl(path: str | Path) -> Path:
    try:
        root = ensure_local_path(path)
    except PhaseBError as exc:
        raise PhaseCError(str(exc).replace("Phase B", "Phase C")) from exc
    if root.is_file():
        if root.suffix.lower() != ".jsonl":
            raise PhaseCError(f"Phase C corpus must be jsonl, got {root}")
        return root
    jsonls = sorted(
        p for p in root.glob("*.jsonl") if p.is_file() and not p.name.endswith(".tmp")
    )
    if len(jsonls) != 1:
        raise PhaseCError("Phase C expects exactly one materialized jsonl")
    return jsonls[0]


def iter_jsonl_texts(jsonl_path: Path) -> Iterator[str]:
    for line_no, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PhaseCError(f"{jsonl_path}:{line_no}: invalid json") from exc
        if not isinstance(obj, dict):
            raise PhaseCError(f"{jsonl_path}:{line_no}: record must be an object")
        text = obj.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        yield text


def load_tokenizer(path: str | Path):
    root = Path(path)
    json_file = root / "tokenizer.json" if root.is_dir() else root
    if not json_file.is_file():
        raise PhaseCError("tokenizer.json not found on local path")
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(json_file))


def encode_with_tokenizer(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False).ids)


def assert_pretrain_body_clean(body_ids: list[int]) -> None:
    overlap = PRETRAIN_BODY_FORBIDDEN_IDS.intersection(body_ids)
    if overlap:
        raise PhaseCError(f"forbidden special ids in document body: {sorted(overlap)}")


def assert_packed_ids_clean(ids: list[int]) -> None:
    overlap = PRETRAIN_FORBIDDEN_IDS.intersection(ids)
    if overlap:
        raise PhaseCError(f"forbidden special ids in packed sequence: {sorted(overlap)}")


def wrap_pretrain_document(
    body_ids: list[int],
    bos_id: int = PRETRAIN_BOS_ID,
    end_id: int = PRETRAIN_END_OF_TEXT_ID,
) -> list[int]:
    if not body_ids:
        raise PhaseCError("empty document")
    if end_id == PRETRAIN_EOT_ID or end_id == bos_id:
        raise PhaseCError("document terminator must be END_OF_TEXT, not EOT/BOS/eos")
    if end_id != PRETRAIN_END_OF_TEXT_ID:
        raise PhaseCError(f"document terminator id {end_id} is not {PRETRAIN_END_OF_TEXT_ID}")
    assert_pretrain_body_clean(body_ids)
    wrapped = [bos_id, *body_ids, end_id]
    assert_packed_ids_clean(wrapped)
    return wrapped


def count_honest(row: dict[str, Any]) -> int:
    return sum(1 for tok in row["labels"] if tok != -100)


def _packed_row(seq: list[int], pad_id: int, padded: bool) -> dict[str, Any]:
    assert_packed_ids_clean(seq)
    if padded:
        mask = [0 if tok == pad_id else 1 for tok in seq]
        labels = [tok if m else -100 for tok, m in zip(seq, mask)]
    else:
        mask = [1] * len(seq)
        labels = list(seq)
    return {"input_ids": list(seq), "labels": labels, "attention_mask": mask}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_phase_c_shard(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    pad_id: int,
    extra_meta: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise PhaseCError("no packed rows")
    seq_len = len(rows[0]["input_ids"])
    arr = np.zeros((len(rows), seq_len), dtype=np.uint32)
    for i, row in enumerate(rows):
        ids = row["input_ids"]
        if len(ids) != seq_len:
            raise PhaseCError(f"row length {len(ids)} != {seq_len}")
        assert_packed_ids_clean(ids)
        arr[i] = np.asarray(ids, dtype=np.uint32)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    produced = int((arr != pad_id).sum())
    meta = {
        "stage": STAGE,
        "shard_id": "shard-00000",
        "path": path.name,
        "dtype": "uint32",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "n_sequences": int(arr.shape[0]),
        "sequence_length": int(arr.shape[1]),
        "pad_id": int(pad_id),
        "produced_tokens": produced,
        "bytes": int(arr.nbytes),
        "sha256": _sha256_file(path),
        "packer": PACKER_VERSION,
        "status": "pending",
    }
    meta.update(extra_meta)
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def pack_corpus(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    encode: EncodeFn,
    sequence_length: int = SEQUENCE_LENGTH,
    target_honest_tokens: int = TARGET_HONEST_TOKENS,
    ceiling_honest_tokens: int = CEILING_HONEST_TOKENS,
    pad_id: int = PRETRAIN_PAD_ID,
    bos_id: int = PRETRAIN_BOS_ID,
    end_id: int = PRETRAIN_END_OF_TEXT_ID,
    source_id: str = SOURCE_ID,
) -> dict[str, Any]:
    if end_id == PRETRAIN_EOT_ID or end_id != PRETRAIN_END_OF_TEXT_ID:
        raise PhaseCError("document stop must be END_OF_TEXT=18")
    if bos_id != PRETRAIN_BOS_ID:
        raise PhaseCError("document start must be BOS=1")
    if sequence_length < 2:
        raise PhaseCError("sequence_length must be at least 2")
    if target_honest_tokens <= 0 or ceiling_honest_tokens < target_honest_tokens:
        raise PhaseCError("ceiling must be >= target and both must be positive")

    jsonl_path = resolve_corpus_jsonl(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    buf: list[int] = []
    rows: list[dict[str, Any]] = []
    honest = 0
    documents_seen = 0
    documents_packed = 0
    documents_skipped = 0
    hit_target = False
    hit_ceiling = False

    def maybe_take(seq: list[int], padded: bool) -> bool:
        nonlocal honest, hit_ceiling
        row = _packed_row(seq, pad_id, padded=padded)
        add = count_honest(row)
        if add <= 0:
            return False
        if honest + add > ceiling_honest_tokens:
            hit_ceiling = True
            return False
        rows.append(row)
        honest += add
        return True

    for text in iter_jsonl_texts(jsonl_path):
        documents_seen += 1
        ids = encode(text)
        if not ids:
            documents_skipped += 1
            continue
        try:
            wrapped = wrap_pretrain_document(ids, bos_id=bos_id, end_id=end_id)
        except PhaseCError:
            documents_skipped += 1
            continue
        documents_packed += 1
        buf.extend(wrapped)
        while len(buf) >= sequence_length:
            seq = buf[:sequence_length]
            if not maybe_take(seq, padded=False):
                buf = []
                break
            buf = buf[sequence_length:]
            if honest >= target_honest_tokens:
                hit_target = True
                buf = []
                break
        if hit_target or hit_ceiling:
            break

    if not hit_target and not hit_ceiling and buf:
        padded = list(buf)
        while len(padded) < sequence_length:
            padded.append(pad_id)
        maybe_take(padded[:sequence_length], padded=True)
        if honest >= target_honest_tokens:
            hit_target = True

    status = "packed" if honest >= target_honest_tokens and honest <= ceiling_honest_tokens else "fail"
    if not rows:
        status = "fail"

    shard_meta = None
    if rows:
        shard_meta = write_phase_c_shard(
            rows,
            out / "shard-00000.bin",
            pad_id=pad_id,
            extra_meta={
                "honest_tokens": honest,
                "source_id": source_id,
                "resume_from": RESUME_RELATIVE,
                "resume_step": RESUME_STEP,
                "wrap": "[BOS=1] + body + [END_OF_TEXT=18]",
            },
        )

    report = {
        "stage": STAGE,
        "status": status,
        "source_id": source_id,
        "input": str(jsonl_path),
        "output_dir": str(out),
        "packer": PACKER_VERSION,
        "wrap": "[BOS=1] + body + [END_OF_TEXT=18]",
        "sequence_length": sequence_length,
        "target_honest_tokens": target_honest_tokens,
        "ceiling_honest_tokens": ceiling_honest_tokens,
        "honest_tokens": honest,
        "n_sequences": len(rows),
        "documents_seen": documents_seen,
        "documents_packed": documents_packed,
        "documents_skipped": documents_skipped,
        "hit_target": hit_target,
        "hit_ceiling": hit_ceiling,
        "resume_from": RESUME_RELATIVE,
        "resume_step": RESUME_STEP,
        "train_authorized": TRAIN_AUTHORIZED,
        "s1": S1_STATUS,
        "hub": HUB_STATUS,
        "shard": None if shard_meta is None else {
            "path": shard_meta["path"],
            "sha256": shard_meta["sha256"],
            "produced_tokens": shard_meta["produced_tokens"],
            "n_sequences": shard_meta["n_sequences"],
        },
    }
    (out / "phase_c.pack.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if status != "packed":
        raise PhaseCError(
            f"Phase C pack failed: honest={honest} target={target_honest_tokens} ceiling={ceiling_honest_tokens}"
        )
    return report


def pack_from_tokenizer(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    tokenizer_path: str | Path,
    sequence_length: int = SEQUENCE_LENGTH,
    target_honest_tokens: int = TARGET_HONEST_TOKENS,
    ceiling_honest_tokens: int = CEILING_HONEST_TOKENS,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path)
    return pack_corpus(
        input_path=input_path,
        output_dir=output_dir,
        encode=lambda text: encode_with_tokenizer(tokenizer, text),
        sequence_length=sequence_length,
        target_honest_tokens=target_honest_tokens,
        ceiling_honest_tokens=ceiling_honest_tokens,
    )


def run_train(*_args: Any, **_kwargs: Any) -> None:
    assert_train_closed()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SHINRA DATA V1 Phase C packer (train closed)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    pack_p = sub.add_parser("pack", help="Deterministic rolling pack from local FineWeb-Edu JSONL")
    pack_p.add_argument("--input", required=True)
    pack_p.add_argument("--output", required=True)
    pack_p.add_argument("--tokenizer", required=True)
    pack_p.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    pack_p.add_argument("--target", type=int, default=TARGET_HONEST_TOKENS)
    pack_p.add_argument("--ceiling", type=int, default=CEILING_HONEST_TOKENS)
    sub.add_parser("train", help="Refuses. Phase C train is not authorized.")
    args = parser.parse_args(argv)
    if args.cmd == "train":
        print("Phase C train is not authorized", file=sys.stderr)
        return 2
    try:
        report = pack_from_tokenizer(
            input_path=args.input,
            output_dir=args.output,
            tokenizer_path=args.tokenizer,
            sequence_length=args.sequence_length,
            target_honest_tokens=args.target,
            ceiling_honest_tokens=args.ceiling,
        )
    except PhaseCError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({k: report[k] for k in ("status", "honest_tokens", "n_sequences", "resume_from")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
