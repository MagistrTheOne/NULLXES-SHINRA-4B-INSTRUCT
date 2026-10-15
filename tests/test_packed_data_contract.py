from __future__ import annotations

import json

import pytest
import torch

from data.pack import PackContractError
from data.packed_data import count_label_tokens, count_target_tokens_post_shift, write_shard_bin, verify_shard


def test_optimizer_token_counter_counts_post_shift_targets():
    labels = torch.tensor([[1, 8, 9, -100]])
    assert count_label_tokens(labels) == 3
    assert count_target_tokens_post_shift(labels) == 2


def test_shard_writer_rejects_special_tokens_and_writes_sha(tmp_path):
    with pytest.raises(PackContractError):
        write_shard_bin([{"input_ids": [1, 21, 2, 3]}], tmp_path / "bad.bin", stage="pretrain", shard_id="bad")
    meta = write_shard_bin([{"input_ids": [1, 42, 43, 19]}], tmp_path / "good.bin", stage="pretrain", shard_id="good")
    assert verify_shard(tmp_path / "good.bin")["sha256"] == meta["sha256"]
