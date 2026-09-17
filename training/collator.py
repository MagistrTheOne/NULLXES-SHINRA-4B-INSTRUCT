"""Map-style packed parquet dataset + default collate for SHINRA."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class PackedParquetDataset(Dataset):
    def __init__(self, data_dir: str | Path) -> None:
        self.files = sorted(Path(data_dir).glob("*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No parquet files in {data_dir}")
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
        return out


def collate_lm(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([row[key] for row in batch], dim=0) for key in keys}
