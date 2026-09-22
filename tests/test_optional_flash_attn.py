"""Regression tests for optional FlashAttention packaging."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from model.configuration_shinra import ShinraConfig
from model.modeling_attn import ShinraFlashAttention2


def test_modeling_attn_has_no_static_flash_attn_import():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "model"
        / "modeling_attn.py"
    )

    tree = ast.parse(source_path.read_text())

    static_imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            static_imports.extend(
                alias.name
                for alias in node.names
            )

        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                static_imports.append(node.module)

    assert not any(
        name == "flash_attn"
        or name.startswith("flash_attn.")
        for name in static_imports
    )


def test_flash_attention_backend_reports_missing_optional_dependency(
    monkeypatch,
):
    config = ShinraConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=32768,
        attention_implementation="flash_attention_2",
    )

    attention = ShinraFlashAttention2(
        config,
        layer_idx=0,
    )

    original_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "flash_attn":
            raise ImportError(
                "synthetic missing flash_attn"
            )

        return original_import_module(
            name,
            package,
        )

    monkeypatch.setattr(
        importlib,
        "import_module",
        fake_import_module,
    )

    hidden_states = __import__("torch").zeros(
        1,
        2,
        config.hidden_size,
    )

    position_embeddings = (
        __import__("torch").ones(
            1,
            2,
            config.head_dim,
        ),
        __import__("torch").zeros(
            1,
            2,
            config.head_dim,
        ),
    )

    with pytest.raises(
        ImportError,
        match="optional.*flash_attn",
    ):
        attention(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
