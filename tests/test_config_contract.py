from __future__ import annotations

import copy
from pathlib import Path

import pytest

from configs.loader import CONFIG_PATH, ConfigError, load_config, validate_config
from model.configuration_shinra import ShinraConfig


def test_single_config_is_complete_and_training_is_gated():
    config = load_config(CONFIG_PATH)
    assert config["status"] == "data_tokenizer_and_runtime_audit_required"
    assert config["training"]["enabled"] is False
    assert config["data"]["training_enabled"] is False
    assert config["runtime"]["colab_launch_allowed"] is False
    assert config["identity"]["invariant"] == "SHINRA is SHINRA. Created by NULLXES."


def test_ready_configuration_can_enable_colab_only_after_all_review_gates():
    config = load_config()
    config["status"] = "ready_for_training"
    config["training"]["enabled"] = True
    config["data"]["training_enabled"] = True
    config["runtime"]["colab_launch_allowed"] = True
    config["data"]["sources"] = [{"id": "audited/source", "revision": "immutable", "status": "approved"}]
    config["tokenizer"]["artifact_sha256"] = "a" * 64
    config["data"]["manifest_sha256"] = "b" * 64
    validate_config(config, require_ready=True)


def test_model_config_uses_separate_eot_and_document_end():
    config = ShinraConfig.from_yaml(str(CONFIG_PATH))
    assert config.eos_token_id == 2
    assert config.document_end_token_id == 4


def test_excluded_corpora_cannot_be_reintroduced():
    config = load_config()
    altered = copy.deepcopy(config)
    altered["data"]["sources"].append({"id": "org/FineWeb-Edu", "revision": "sha", "status": "approved"})
    with pytest.raises(ConfigError, match="excluded source"):
        validate_config(altered)


def test_special_token_ids_are_config_driven_and_unique():
    config = load_config()
    altered = copy.deepcopy(config)
    altered["tokenizer"]["special_token_ids"]["document"] = 5
    with pytest.raises(ConfigError):
        validate_config(altered)
