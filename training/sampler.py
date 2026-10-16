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
    """Schedule rows by target tokens and separately account rows consumed by training."""

    LANGUAGE_CODES = {"en": 0, "ru": 1}

    def __init__(self, data_source, seed: int, weights: dict[str, float]) -> None:
        self.data_source = data_source
        self.seed = int(seed)
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
        self.generator = torch.Generator().manual_seed(self.seed)
        self.permutations: dict[int, torch.Tensor | None] = {code: None for code in self.weights}
        self.positions = {code: 0 for code in self.weights}
        self.epochs_by_language = {code: 0 for code in self.weights}
        self.scheduled_tokens_by_language = {code: 0 for code in self.weights}
        self.tokens_by_language = {code: 0 for code in self.weights}
        self.consumed_rows_by_language = {code: 0 for code in self.weights}
        self.pending_indices: list[int] = []
        self.position = 0
        self.epoch = 0
        packed_identity = str(getattr(data_source, "packed_manifest_sha256", ""))
        self.dataset_identity = hashlib.sha256(
            languages.tobytes() + targets.tobytes() + packed_identity.encode("ascii")
        ).hexdigest()

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
        self.scheduled_tokens_by_language[code] += int(self.target_counts[dataset_index])
        self.pending_indices.append(dataset_index)
        return dataset_index

    def __iter__(self):
        # DataLoader implementations may request indices ahead of the batch being
        # trained. On resume, replay those scheduled rows before advancing the cursor.
        for index in list(self.pending_indices):
            yield index
        if self.position >= len(self.data_source) and not self.pending_indices:
            self.position = 0
            self.epoch += 1
        while self.position < len(self.data_source):
            en_progress = self.scheduled_tokens_by_language[0] / self.weights[0]
            ru_progress = self.scheduled_tokens_by_language[1] / self.weights[1]
            code = 0 if en_progress <= ru_progress else 1
            index = self._next_index(code)
            self.position += 1
            yield index

    def mark_consumed(self, indices: list[int]) -> None:
        """Commit indices in delivery order after the training loop receives their batch."""
        for raw_index in indices:
            index = int(raw_index)
            if not self.pending_indices or self.pending_indices[0] != index:
                expected = self.pending_indices[0] if self.pending_indices else None
                raise ValueError(f"training row order differs from sampler delivery: got {index}, expected {expected}")
            self.pending_indices.pop(0)
            language_code = int(self.data_source.language_codes[index])
            if language_code not in self.weights:
                raise ValueError(f"training row {index} has unsupported language code {language_code}")
            self.tokens_by_language[language_code] += int(self.target_counts[index])
            self.consumed_rows_by_language[language_code] += 1

    def mix_report(self) -> dict[str, int | float]:
        en, ru = self.tokens_by_language[0], self.tokens_by_language[1]
        total = en + ru
        row_metrics: dict[str, int] = {}
        for code, name in ((0, "en"), (1, "ru")):
            pool_size = len(self.indices[code])
            presentations = self.consumed_rows_by_language[code]
            unique_rows = min(presentations, pool_size)
            row_metrics[f"{name}_row_presentations"] = presentations
            row_metrics[f"{name}_unique_rows_presented"] = unique_rows
            row_metrics[f"{name}_reused_row_presentations"] = presentations - unique_rows
        return {"en_target_tokens": en, "ru_target_tokens": ru,
                "en_fraction": en / total if total else 0.0,
                "ru_fraction": ru / total if total else 0.0,
                "absolute_token_imbalance": abs(en - ru), "sampled_target_tokens": total,
                **row_metrics,
                "reused_row_presentations": row_metrics["en_reused_row_presentations"]
                + row_metrics["ru_reused_row_presentations"]}

    def state_dict(self) -> dict:
        return {"state_version": 2, "dataset_length": len(self.data_source),
                "dataset_identity": self.dataset_identity,
                "seed": self.seed,
                "weights": self.weights, "generator_state": self.generator.get_state(),
                "permutations": self.permutations, "positions": self.positions,
                "epochs_by_language": self.epochs_by_language,
                "scheduled_tokens_by_language": self.scheduled_tokens_by_language,
                "tokens_by_language": self.tokens_by_language,
                "consumed_rows_by_language": self.consumed_rows_by_language,
                "pending_indices": list(self.pending_indices),
                "position": self.position, "epoch": self.epoch}

    def _replay_legacy_schedule(self, state: dict, scheduled_rows: int) -> list[int]:
        """Rebuild the v1 schedule from its seed and prove it matches saved cursors."""
        generator = torch.Generator().manual_seed(self.seed)
        permutations: dict[int, torch.Tensor | None] = {code: None for code in self.weights}
        positions = {code: 0 for code in self.weights}
        epochs = {code: 0 for code in self.weights}
        tokens = {code: 0 for code in self.weights}
        global_position = 0
        global_epoch = 0
        rows: list[int] = []

        while len(rows) < scheduled_rows:
            if global_position >= len(self.data_source):
                global_position = 0
                global_epoch += 1
            code = 0 if tokens[0] / self.weights[0] <= tokens[1] / self.weights[1] else 1
            permutation = permutations[code]
            if permutation is None or positions[code] >= len(permutation):
                permutation = torch.randperm(len(self.indices[code]), generator=generator)
                permutations[code] = permutation
                positions[code] = 0
                epochs[code] += 1
            local_position = positions[code]
            index = int(self.indices[code][int(permutation[local_position])])
            positions[code] = local_position + 1
            tokens[code] += int(self.target_counts[index])
            global_position += 1
            rows.append(index)

        saved_positions = {int(key): int(value) for key, value in state["positions"].items()}
        saved_epochs = {int(key): int(value) for key, value in state["epochs_by_language"].items()}
        saved_tokens = {int(key): int(value) for key, value in state["tokens_by_language"].items()}
        saved_permutations = state["permutations"]
        permutation_mismatch = any(
            (permutations[code] is None) != (saved_permutations.get(code, saved_permutations.get(str(code))) is None)
            or (permutations[code] is not None and not torch.equal(
                permutations[code], saved_permutations.get(code, saved_permutations.get(str(code)))
            ))
            for code in self.weights
        )
        if (positions != saved_positions or epochs != saved_epochs or tokens != saved_tokens
                or permutation_mismatch or global_position != int(state["position"])
                or global_epoch != int(state["epoch"])
                or not torch.equal(generator.get_state(), state["generator_state"])):
            raise ValueError(
                "legacy sampler schedule cannot be reproduced exactly from its saved seed and cursors; "
                "refusing heuristic resume"
            )
        return rows

    def load_state_dict(self, state: dict, *, expected_consumed_tokens: int | None = None,
                        expected_consumed_rows: int | None = None) -> None:
        if int(state["dataset_length"]) != len(self.data_source) or state["dataset_identity"] != self.dataset_identity:
            raise ValueError("saved token-mix sampler dataset identity differs from the current training split")
        if int(state["seed"]) != self.seed:
            raise ValueError("saved token-mix sampler seed differs from the current training configuration")
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
        self.position = int(state["position"])
        self.epoch = int(state["epoch"])
        if not 0 <= self.position <= len(self.data_source):
            raise ValueError("saved token-mix sampler cursor is invalid")
        state_version = int(state.get("state_version", 1))
        if state_version not in {1, 2}:
            raise ValueError(f"unsupported token-mix sampler state version: {state_version}")

        saved_tokens = {int(key): int(value) for key, value in state["tokens_by_language"].items()}
        if state_version == 1:
            if expected_consumed_tokens is None or expected_consumed_rows is None:
                raise ValueError(
                    "legacy sampler migration requires checkpoint consumed-token and consumed-row totals"
                )
            self.scheduled_tokens_by_language = dict(saved_tokens)
        else:
            self.scheduled_tokens_by_language = {
                int(key): int(value) for key, value in state["scheduled_tokens_by_language"].items()
            }
            self.tokens_by_language = dict(saved_tokens)
            self.consumed_rows_by_language = {
                int(key): int(value) for key, value in state["consumed_rows_by_language"].items()
            }
            self.pending_indices = [int(index) for index in state["pending_indices"]]
            if any(not 0 <= index < len(self.data_source) for index in self.pending_indices):
                raise ValueError("saved pending sampler indices are outside the training dataset")

        scheduled_rows_by_language: dict[int, int] = {}
        for code in self.weights:
            permutation = self.permutations[code]
            epochs = self.epochs_by_language[code]
            position = self.positions[code]
            if epochs < 0 or self.tokens_by_language[code] < 0:
                raise ValueError("saved per-language sampler counters are negative")
            if permutation is None:
                if epochs != 0 or position != 0 or self.tokens_by_language[code] != 0:
                    raise ValueError("saved empty language sampler state has nonzero counters")
                if self.scheduled_tokens_by_language[code] != 0:
                    raise ValueError("saved empty language sampler has scheduled tokens")
                scheduled_rows_by_language[code] = 0
                continue
            if epochs <= 0 or not 0 <= position <= len(permutation):
                raise ValueError("saved per-language sampler cursor is invalid")
            completed_cycle_tokens = int(np.sum(self.target_counts[self.indices[code]], dtype=np.int64))
            local_prefix = permutation[:position].numpy().astype(np.int64, copy=False)
            current_dataset_indices = self.indices[code][local_prefix]
            current_cycle_tokens = int(np.sum(self.target_counts[current_dataset_indices], dtype=np.int64))
            expected_tokens = (epochs - 1) * completed_cycle_tokens + current_cycle_tokens
            if self.scheduled_tokens_by_language[code] != expected_tokens:
                raise ValueError("saved scheduled-token counters differ from the row permutations")
            scheduled_rows_by_language[code] = (epochs - 1) * len(self.indices[code]) + position

        scheduled_tokens_from_rows = {code: 0 for code in self.weights}
        scheduled_rows_from_pending = {code: 0 for code in self.weights}
        for index in self.pending_indices:
            code = int(self.data_source.language_codes[index])
            if code not in self.weights:
                raise ValueError(f"saved pending row {index} has unsupported language code {code}")
            scheduled_tokens_from_rows[code] += int(self.target_counts[index])
            scheduled_rows_from_pending[code] += 1

        if state_version == 1:
            scheduled_rows = sum(scheduled_rows_by_language.values())
            if scheduled_rows < expected_consumed_rows:
                raise ValueError(
                    "legacy sampler cursor ends before the checkpoint's consumed rows: "
                    f"sampler_rows={scheduled_rows}, checkpoint_rows={expected_consumed_rows}"
                )
            replayed_rows = self._replay_legacy_schedule(state, scheduled_rows)
            consumed_indices = replayed_rows[:expected_consumed_rows]
            pending_indices = replayed_rows[expected_consumed_rows:]
            consumed_tokens_by_language = {code: 0 for code in self.weights}
            consumed_rows_by_language = {code: 0 for code in self.weights}
            for index in consumed_indices:
                code = int(self.data_source.language_codes[index])
                consumed_tokens_by_language[code] += int(self.target_counts[index])
                consumed_rows_by_language[code] += 1
            consumed_tokens = sum(consumed_tokens_by_language.values())
            if consumed_tokens != expected_consumed_tokens:
                raise ValueError(
                    "legacy sampler replay does not identify the checkpoint consumption boundary: "
                    f"replayed_prefix_tokens={consumed_tokens}, checkpoint_tokens={expected_consumed_tokens}, "
                    f"replayed_prefix_rows={expected_consumed_rows}, scheduled_rows={scheduled_rows}"
                )
            self.scheduled_tokens_by_language = dict(saved_tokens)
            self.tokens_by_language = consumed_tokens_by_language
            self.consumed_rows_by_language = consumed_rows_by_language
            self.pending_indices = pending_indices
            pending_tokens_by_language = {code: 0 for code in self.weights}
            pending_rows_by_language = {code: 0 for code in self.weights}
            for index in self.pending_indices:
                code = int(self.data_source.language_codes[index])
                pending_tokens_by_language[code] += int(self.target_counts[index])
                pending_rows_by_language[code] += 1
            for code in self.weights:
                if self.scheduled_tokens_by_language[code] != (
                    self.tokens_by_language[code] + pending_tokens_by_language[code]
                ) or scheduled_rows_by_language[code] != (
                    self.consumed_rows_by_language[code] + pending_rows_by_language[code]
                ):
                    raise ValueError("legacy sampler migration failed scheduled/consumed/pending accounting")
        else:
            for code in self.weights:
                if self.scheduled_tokens_by_language[code] != self.tokens_by_language[code] + scheduled_tokens_from_rows[code]:
                    raise ValueError("scheduled and consumed language-token totals differ from pending rows")
                if self.consumed_rows_by_language[code] + scheduled_rows_from_pending[code] \
                        != scheduled_rows_by_language[code]:
                    raise ValueError("scheduled and consumed row totals differ from pending rows")
            if expected_consumed_tokens is not None and sum(self.tokens_by_language.values()) != expected_consumed_tokens:
                raise ValueError("sampler consumed-token total differs from the training checkpoint")
            if expected_consumed_rows is not None and sum(self.consumed_rows_by_language.values()) != expected_consumed_rows:
                raise ValueError("sampler consumed-row total differs from the training checkpoint")

        scheduled_rows = sum(scheduled_rows_by_language.values())
        if scheduled_rows != self.epoch * len(self.data_source) + self.position:
            raise ValueError("global and per-language sampler cursors disagree")
