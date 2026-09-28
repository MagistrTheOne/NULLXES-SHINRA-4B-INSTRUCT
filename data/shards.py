"""uint32 packed shards. 1 token = 4 bytes. Labels reconstructed from pad_id."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.pack import PACKER_VERSION, PackContractError, assert_packed_ids_clean
from tokenizer.special_tokens import PRETRAIN_FORBIDDEN_IDS, PRETRAIN_PAD_ID


class ShardError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def rows_to_array(rows: list[dict], sequence_length: int) -> np.ndarray:
    if not rows:
        raise ShardError("no packed rows")
    arr = np.zeros((len(rows), sequence_length), dtype=np.uint32)
    for i, row in enumerate(rows):
        ids = row["input_ids"]
        if len(ids) != sequence_length:
            raise ShardError(f"row length {len(ids)} != {sequence_length}")
        assert_packed_ids_clean(ids)
        arr[i] = np.asarray(ids, dtype=np.uint32)
    return arr


def write_shard_bin(
    rows: list[dict],
    path: Path,
    *,
    stage: str,
    shard_id: str,
    pad_id: int = PRETRAIN_PAD_ID,
    extra_meta: dict | None = None,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = rows_to_array(rows, len(rows[0]["input_ids"]))
    arr.tofile(path)
    produced = int((arr != pad_id).sum())
    meta = {
        "stage": stage,
        "shard_id": shard_id,
        "path": path.name,
        "dtype": "uint32",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "n_sequences": int(arr.shape[0]),
        "sequence_length": int(arr.shape[1]),
        "pad_id": int(pad_id),
        "produced_tokens": produced,
        "bytes": int(arr.nbytes),
        "sha256": sha256_file(path),
        "packer": PACKER_VERSION,
        "status": "pending",
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def load_shard_meta(path: Path) -> dict:
    meta_path = Path(path).with_suffix(".meta.json") if Path(path).suffix == ".bin" else Path(path)
    if meta_path.suffix != ".json":
        meta_path = Path(path).with_suffix(".meta.json")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def verify_shard(bin_path: Path) -> dict:
    meta = json.loads(bin_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    actual = sha256_file(bin_path)
    if actual != meta["sha256"]:
        raise ShardError(f"sha mismatch {bin_path.name}: {actual} != {meta['sha256']}")
    n_seq, seq_len = meta["shape"]
    expected = n_seq * seq_len * 4
    if bin_path.stat().st_size != expected:
        raise ShardError(f"size mismatch {bin_path.name}")
    return meta


class PackedBinDataset(Dataset):
    """Memmap uint32 sequences. labels = id if not pad else -100."""

    def __init__(self, data_dir: str | Path, pad_id: int = PRETRAIN_PAD_ID) -> None:
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.bin"))
        if not self.files:
            raise FileNotFoundError(f"No .bin shards in {self.data_dir}")
        self.pad_id = pad_id
        self._index: list[tuple[int, int]] = []
        self._maps: list[np.ndarray] = []
        for file_idx, path in enumerate(self.files):
            meta = verify_shard(path)
            n_seq, seq_len = meta["shape"]
            mm = np.memmap(path, dtype=np.uint32, mode="r", shape=(n_seq, seq_len))
            self._maps.append(mm)
            for row in range(n_seq):
                self._index.append((file_idx, row))
            if int(meta.get("pad_id", pad_id)) != pad_id:
                self.pad_id = int(meta["pad_id"])
        self.sequence_length = int(self._maps[0].shape[1])

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, row = self._index[index]
        ids = np.asarray(self._maps[file_idx][row], dtype=np.int64)
        mask = (ids != self.pad_id).astype(np.int64)
        labels = np.where(mask, ids, np.int64(-100))
        overlap = PRETRAIN_FORBIDDEN_IDS.intersection(ids.tolist())
        if overlap:
            raise PackContractError(f"forbidden ids in shard row: {sorted(overlap)}")
        return {
            "input_ids": torch.from_numpy(ids.copy()),
            "attention_mask": torch.from_numpy(mask.copy()),
            "labels": torch.from_numpy(labels.copy()),
        }


def count_label_tokens(labels: torch.Tensor) -> int:
    return int((labels != -100).sum().item())
