"""Deterministic resumable sampler without worker prefetch ambiguity."""

from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch.utils.data import Sampler


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
            if not torch.equal(torch.sort(permutation).values,
                               torch.arange(len(self.data_source), dtype=permutation.dtype)):
                raise ValueError("saved sampler permutation is not a full dataset permutation")
        position = int(state["position"])
        if position < 0 or (permutation is not None and position > len(permutation)):
            raise ValueError("saved sampler cursor is invalid")
        self.generator.set_state(state["generator_state"])
        self.permutation = permutation
        self.position = position
        self.epoch = int(state["epoch"])


class StatefulTokenMixSampler(Sampler[int]):
    """Deterministically schedule rows by accumulated supervised EN/RU target tokens."""

    LANGUAGE_CODES = {"en": 0, "ru": 1}

    def __init__(self, data_source, seed: int, weights: dict[str, float]) -> None:
        self.data_source = data_source
        self.weights = {self.LANGUAGE_CODES[key]: float(weights[key]) for key in ("en", "ru")}
        if any(value <= 0 for value in self.weights.values()) or abs(sum(self.weights.values()) - 1.0) > 1e-12:
            raise ValueError("token-mix weights must be positive and sum to 1")
        languages = np.asarray(data_source.language_codes, dtype=np.uint8)
        targets = np.asarray(data_source.target_tokens, dtype=np.uint16)
        if len(languages) != len(data_source) or len(targets) != len(data_source):
            raise ValueError("packed dataset token-mix metadata length is inconsistent")
        self.indices = {code: np.flatnonzero(languages == code).astype(np.int64) for code in self.weights}
        if any(len(indices) == 0 for indices in self.indices.values()):
            raise ValueError("both EN and RU must have rows for the configured token-level mix")
        self.target_counts = targets
        self.generator = torch.Generator().manual_seed(seed)
        self.permutations: dict[int, torch.Tensor | None] = {code: None for code in self.weights}
        self.positions = {code: 0 for code in self.weights}
        self.epochs_by_language = {code: 0 for code in self.weights}
        self.tokens_by_language = {code: 0 for code in self.weights}
        self.position = 0
        self.epoch = 0
        self.dataset_identity = hashlib.sha256(languages.tobytes() + targets.tobytes()).hexdigest()

    def __len__(self) -> int:
        return len(self.data_source)

    def _new_permutation(self, code: int) -> None:
        self.permutations[code] = torch.randperm(len(self.indices[code]), generator=self.generator)
        self.positions[code] = 0
        self.epochs_by_language[code] += 1

    def _next_index(self, code: int) -> int:
        permutation = self.permutations[code]
        if permutation is None or self.positions[code] >= len(permutation):
            self._new_permutation(code)
            permutation = self.permutations[code]
        local_position = self.positions[code]
        self.positions[code] = local_position + 1
        dataset_index = int(self.indices[code][int(permutation[local_position])])
        self.tokens_by_language[code] += int(self.target_counts[dataset_index])
        return dataset_index

    def __iter__(self):
        if self.position >= len(self.data_source):
            self.position = 0
            self.epoch += 1
        while self.position < len(self.data_source):
            en_progress = self.tokens_by_language[0] / self.weights[0]
            ru_progress = self.tokens_by_language[1] / self.weights[1]
            code = 0 if en_progress <= ru_progress else 1
            index = self._next_index(code)
            self.position += 1
            yield index

    def mix_report(self) -> dict[str, int | float]:
        en, ru = self.tokens_by_language[0], self.tokens_by_language[1]
        total = en + ru
        return {"en_target_tokens": en, "ru_target_tokens": ru,
                "en_fraction": en / total if total else 0.0,
                "ru_fraction": ru / total if total else 0.0,
                "absolute_token_imbalance": abs(en - ru), "sampled_target_tokens": total}

    def state_dict(self) -> dict:
        return {"dataset_length": len(self.data_source), "dataset_identity": self.dataset_identity,
                "weights": self.weights, "generator_state": self.generator.get_state(),
                "permutations": self.permutations, "positions": self.positions,
                "epochs_by_language": self.epochs_by_language, "tokens_by_language": self.tokens_by_language,
                "position": self.position, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        if int(state["dataset_length"]) != len(self.data_source) or state["dataset_identity"] != self.dataset_identity:
            raise ValueError("saved token-mix sampler dataset identity differs from the current training split")
        saved_weights = {int(key): float(value) for key, value in state["weights"].items()}
        if saved_weights != self.weights:
            raise ValueError("saved token-mix weights differ from the current YAML")
        permutations = state["permutations"]
        for code in self.weights:
            value = permutations.get(code, permutations.get(str(code)))
            if value is not None and (value.ndim != 1 or len(value) != len(self.indices[code])
                    or not torch.equal(torch.sort(value).values,
                                       torch.arange(len(self.indices[code]), dtype=value.dtype))):
                raise ValueError("saved language-specific row permutation is malformed")
            self.permutations[code] = value
        self.generator.set_state(state["generator_state"])
        self.positions = {int(key): int(value) for key, value in state["positions"].items()}
        self.epochs_by_language = {int(key): int(value) for key, value in state["epochs_by_language"].items()}
        self.tokens_by_language = {int(key): int(value) for key, value in state["tokens_by_language"].items()}
        self.position = int(state["position"])
        self.epoch = int(state["epoch"])
        if not 0 <= self.position <= len(self.data_source):
            raise ValueError("saved token-mix sampler cursor is invalid")
        if any(not 0 <= self.positions[code] <= len(self.permutations[code])
               for code in self.weights if self.permutations[code] is not None):
            raise ValueError("saved per-language sampler cursor is invalid")
