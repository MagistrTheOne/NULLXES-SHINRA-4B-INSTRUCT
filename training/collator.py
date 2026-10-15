"""Map-style packed parquet dataset + default collate for SHINRA."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from data.packed_data import ShardError
from configs.loader import load_config


class PackedParquetDataset(Dataset):
    def __init__(self, data_dir: str | Path) -> None:
        self.files = sorted(Path(data_dir).glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No parquet files in {data_dir}")
        import hashlib

        packed_manifest_path = Path(data_dir) / "manifest.json"
        if not packed_manifest_path.is_file():
            raise ValueError(f"{data_dir}: packed manifest.json is required")
        packed_manifest = __import__("json").loads(packed_manifest_path.read_text(encoding="utf-8"))
        packed_manifest_sha = packed_manifest.pop("manifest_sha256", None)
        expected_packed_sha = hashlib.sha256(__import__("json").dumps(packed_manifest, sort_keys=True).encode()).hexdigest()
        if packed_manifest_sha != expected_packed_sha:
            raise ValueError(f"{data_dir}: packed manifest self-hash mismatch")
        declared = packed_manifest.get("shard_files")
        shard_hashes = packed_manifest.get("shard_sha256")
        if not isinstance(declared, list) or sorted(declared) != [p.name for p in self.files]:
            raise ShardError(f"{data_dir}: packed shard set differs from manifest")
        if not isinstance(shard_hashes, dict) or set(shard_hashes) != set(declared):
            raise ShardError(f"{data_dir}: packed shard hashes are incomplete")
        for file in self.files:
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if shard_hashes[file.name] != digest:
                raise ShardError(f"{file}: SHA256 differs from packed manifest")
        source_manifest_hashes = set()
        for file in self.files:
            table = pq.read_table(file)
            if "source_manifest_sha256" not in table.column_names:
                raise ValueError(f"{file}: corpus provenance missing")
            hashes = set(table.column("source_manifest_sha256").to_pylist())
            if len(hashes) != 1:
                raise ValueError(f"{file}: mixed corpus manifests")
            source_manifest_hashes.update(hashes)
        if len(source_manifest_hashes) != 1:
            raise ValueError(f"{data_dir}: parquet shards have inconsistent corpus manifests")
        self.source_manifest_sha256 = next(iter(source_manifest_hashes))
        if self.source_manifest_sha256 != packed_manifest.get("source_manifest_sha256"):
            raise ValueError(f"{data_dir}: parquet shards do not match corpus manifest")
        self._offsets: list[tuple[int, int, int]] = []
        total = 0
        for file_idx, file in enumerate(self.files):
            n = pq.ParquetFile(file).metadata.num_rows
            self._offsets.append((total, total + n, file_idx))
            total += n
        self.length = total
        self._cache_idx = -1
        self._cache_table = None

    def __len__(self) -> int:
        return self.length

    def _locate(self, index: int) -> tuple[int, int]:
        for start, end, file_idx in self._offsets:
            if start <= index < end:
                return file_idx, index - start
        raise IndexError(index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, row = self._locate(index)
        if file_idx != self._cache_idx:
            self._cache_table = pq.read_table(self.files[file_idx])
            self._cache_idx = file_idx
        table = self._cache_table
        item = {name: table.column(name)[row].as_py() for name in table.column_names}
        out = {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }
        if "labels" in item and item["labels"] is not None:
            out["labels"] = torch.tensor(item["labels"], dtype=torch.long)
        else:
            out["labels"] = out["input_ids"].clone()
        return out


def collate_lm(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([row[key] for row in batch], dim=0) for key in keys}


def load_packed_dataset(data_dir: str | Path, pad_id: int | None = None):
    configured_pad = int(load_config()["tokenizer"]["pad_id"])
    if pad_id is not None and pad_id != configured_pad:
        raise ShardError("loader pad ID differs from configs/shinra.yaml")
    return PackedParquetDataset(data_dir)
