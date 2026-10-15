"""Deterministic resumable sampler without worker prefetch ambiguity."""

from __future__ import annotations

from torch.utils.data import Sampler

import torch


class StatefulRandomSampler(Sampler[int]):
    def __init__(self, data_source, seed: int) -> None:
        self.data_source = data_source
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation: torch.Tensor | None = None
        self.position = 0
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.permutation is None or self.position >= len(self.permutation):
            self.permutation = torch.randperm(len(self.data_source), generator=self.generator)
            self.position = 0
            self.epoch += 1
        while self.position < len(self.permutation):
            index = int(self.permutation[self.position])
            self.position += 1
            yield index

    def state_dict(self) -> dict:
        return {
            "dataset_length": len(self.data_source),
            "generator_state": self.generator.get_state(),
            "permutation": self.permutation,
            "position": self.position,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["dataset_length"]) != len(self.data_source):
            raise ValueError("saved sampler dataset length differs from the current training dataset")
        permutation = state["permutation"]
        if permutation is not None:
            if permutation.ndim != 1 or len(permutation) != len(self.data_source):
                raise ValueError("saved sampler permutation is malformed")
            if sorted(permutation.tolist()) != list(range(len(self.data_source))):
                raise ValueError("saved sampler permutation is not a full dataset permutation")
        position = int(state["position"])
        if position < 0 or (permutation is not None and position > len(permutation)):
            raise ValueError("saved sampler cursor is invalid")
        self.generator.set_state(state["generator_state"])
        self.permutation = permutation
        self.position = position
        self.epoch = int(state["epoch"])
