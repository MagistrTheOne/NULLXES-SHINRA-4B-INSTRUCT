"""Checksummed fixed-width packed token datasets and shard identity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from configs.loader import load_config
from data.pack import PACKER_VERSION, PackContractError, current_packed_forbidden_ids

COUNTER_VERSION = "post_shift_targets.v1"


class ShardError(RuntimeError):
    pass


class ShardIdentityError(ShardError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_shard_bin(rows: list[dict], path: Path, *, stage: str, shard_id: str,
                    pad_id: int | None = None, extra_meta: dict | None = None) -> dict:
    if not rows:
        raise ShardError("refusing to write empty shard")
    configured_pad_id = int(load_config()["tokenizer"]["pad_id"])
    pad_id = configured_pad_id if pad_id is None else pad_id
    if pad_id != configured_pad_id:
        raise ShardError(f"pad ID {pad_id} does not match config {configured_pad_id}")
    seq_len = len(rows[0]["input_ids"])
    array = np.asarray([row["input_ids"] for row in rows], dtype=np.uint32)
    if array.ndim != 2 or array.shape[1] != seq_len or any(len(row["input_ids"]) != seq_len for row in rows):
        raise ShardError("inconsistent sequence lengths")
    forbidden_ids = current_packed_forbidden_ids()
    for row in rows:
        overlap = forbidden_ids.intersection(row["input_ids"])
        if overlap:
            raise PackContractError(f"forbidden control IDs in shard row: {sorted(overlap)}")
    if Path(path).exists():
        raise FileExistsError(f"refusing to overwrite shard file: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    array.tofile(temp)
    temp.replace(path)
    meta = {"stage": stage, "shard_id": shard_id, "path": path.name, "dtype": "uint32",
            "shape": list(map(int, array.shape)), "n_sequences": int(array.shape[0]),
            "sequence_length": seq_len, "pad_id": int(pad_id), "produced_tokens": int((array != pad_id).sum()),
            "bytes": int(array.nbytes), "sha256": sha256_file(path), "packer": PACKER_VERSION,
            "status": "pending", **(extra_meta or {})}
    if not meta.get("source_manifest_sha256"):
        raise ShardError("source manifest SHA256 is required to write training shards")
    meta_path = path.with_suffix(".meta.json")
    temp_meta = meta_path.with_suffix(".json.tmp")
    temp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_meta.replace(meta_path)
    return meta


def verify_shard(bin_path: Path) -> dict:
    meta = json.loads(Path(bin_path).with_suffix(".meta.json").read_text(encoding="utf-8"))
    if sha256_file(bin_path) != meta.get("sha256"):
        raise ShardError(f"shard SHA256 mismatch: {bin_path}")
    if Path(bin_path).stat().st_size != int(meta["shape"][0]) * int(meta["shape"][1]) * 4:
        raise ShardError(f"shard byte length mismatch: {bin_path}")
    if meta.get("packer") != PACKER_VERSION or not meta.get("source_manifest_sha256"):
        raise ShardError("shard is missing packer provenance or corpus manifest SHA256")
    return meta


def verify_shard_identity(bin_path: Path, shard_id: str, sha256: str) -> dict:
    meta = verify_shard(bin_path)
    if meta["shard_id"] != shard_id or meta["sha256"] != sha256:
        raise ShardIdentityError("active shard identity mismatch")
    return meta


def update_shard_status(bin_path: Path, status: str, *, extra: dict | None = None, delete_bin: bool = False) -> dict:
    meta_path = Path(bin_path).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["status"] = "deleted" if delete_bin else status
    if extra:
        meta.update(extra)
    temp_meta = meta_path.with_suffix(".json.tmp")
    temp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_meta.replace(meta_path)
    if delete_bin:
        Path(bin_path).unlink()
    return meta


class PackedBinDataset(Dataset):
    def __init__(self, data_dir: str | Path, pad_id: int | None = None) -> None:
        self.files = sorted(Path(data_dir).glob("*.bin"))
        if not self.files:
            raise FileNotFoundError(f"No packed shards in {data_dir}")
        manifest = Path(data_dir) / "manifest.json"
        if not manifest.is_file():
            raise ShardError("packed dataset manifest.json is required")
        self.source_manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        configured_pad_id = int(load_config()["tokenizer"]["pad_id"])
        pad_id = configured_pad_id if pad_id is None else pad_id
        if pad_id != configured_pad_id:
            raise ShardError("loader pad ID differs from configs/shinra.yaml")
        self.pad_id = pad_id
        self._maps = []
        self._index: list[tuple[int, int]] = []
        aggregate = hashlib.sha256()
        self.total_produced_tokens = 0
        for file_index, path in enumerate(self.files):
            meta = verify_shard(path)
            if meta.get("source_manifest_sha256") != self.source_manifest_sha256:
                raise ShardError("shard source manifest does not match packed dataset manifest")
            self.total_produced_tokens += int(meta["produced_tokens"])
            aggregate.update(meta["sha256"].encode("ascii"))
            rows, length = map(int, meta["shape"])
            if int(meta["pad_id"]) != pad_id:
                raise ShardError("shard pad ID differs from approved config")
            mapped = np.memmap(path, dtype=np.uint32, mode="r", shape=(rows, length))
            self._maps.append(mapped)
            self._index.extend((file_index, row) for row in range(rows))
        self.sequence_length = int(self._maps[0].shape[1])
        self.shard_manifest_sha256 = aggregate.hexdigest()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_index, row_index = self._index[index]
        ids = np.asarray(self._maps[file_index][row_index], dtype=np.int64)
        mask = (ids != self.pad_id).astype(np.int64)
        overlap = current_packed_forbidden_ids().intersection(ids.tolist())
        if overlap:
            raise PackContractError(f"forbidden control IDs in training shard: {sorted(overlap)}")
        return {"input_ids": torch.from_numpy(ids.copy()), "attention_mask": torch.from_numpy(mask.copy()),
                "labels": torch.from_numpy(np.where(mask, ids, -100).copy())}


def count_label_tokens(labels: torch.Tensor) -> int:
    return int((labels != -100).sum().item())


def count_target_tokens_post_shift(labels: torch.Tensor) -> int:
    return int((labels[..., 1:] != -100).sum().item())


def list_shard_bins(data_dir: Path) -> list[Path]:
    return sorted(Path(data_dir).glob("*.bin"))


def assert_single_shard(data_dir: Path, shard_id: str | None = None) -> Path:
    bins = list_shard_bins(data_dir)
    if len(bins) != 1:
        raise ShardIdentityError(f"expected exactly one shard in {data_dir}, found {len(bins)}")
    if shard_id is not None and bins[0].stem != shard_id:
        raise ShardIdentityError(f"shard ID {bins[0].stem} != expected {shard_id}")
    return bins[0]
