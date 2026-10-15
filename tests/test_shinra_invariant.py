from __future__ import annotations

import pytest

from configs.loader import load_config
from data.pack import PackContractError, wrap_pretrain_document


def test_identity_and_random_initialization_are_frozen():
    config = load_config()
    assert config["identity"]["invariant"] == "SHINRA is SHINRA. Created by NULLXES."
    assert config["identity"]["initialization"] == "random"
    assert config["identity"]["historical_weights_allowed"] is False


def test_eot_and_document_end_are_distinct_pretraining_controls():
    config = load_config()
    assert config["model"]["eos_token_id"] == 2
    assert config["model"]["document_end_token_id"] == 4
    assert wrap_pretrain_document([42, 43]) == [1, 42, 43, 4]
    with pytest.raises(PackContractError):
        wrap_pretrain_document([42, 2, 43])
    with pytest.raises(PackContractError):
        wrap_pretrain_document([42, 43], end_id=2)
