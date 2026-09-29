"""SHINRA v2 architecture contract. CPU / meta only. No Hub, no GPU, no training."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from architecture.param_count import ShinraSpec, count_parameters
from model.configuration_shinra import ShinraConfig
from model.modeling_attn import ShinraAttention
from model.modeling_shinra import ShinraForCausalLM

ROOT = Path(__file__).resolve().parents[1]
V2_PARAMS = 3_969_056_256
YAML_PATH = ROOT / "configs" / "architecture_v2.yaml"


def test_analytical_param_count_is_v2():
    counts = count_parameters(ShinraSpec())
    assert counts["total"] == V2_PARAMS
    spec = ShinraSpec()
    assert spec.hidden_size == 2560
    assert spec.num_hidden_layers == 36
    assert spec.num_attention_heads == 32
    assert spec.num_key_value_heads == 8
    assert spec.head_dim == 128
    assert spec.intermediate_size == 9728
    assert spec.vocab_size == 131072
    q_width = spec.num_attention_heads * spec.head_dim
    assert q_width == 4096
    assert q_width != spec.hidden_size


def test_yaml_matches_v2_geometry():
    payload = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    model = payload["model"]
    assert model["hidden_size"] == 2560
    assert model["intermediate_size"] == 9728
    assert model["num_hidden_layers"] == 36
    assert model["num_attention_heads"] == 32
    assert model["num_key_value_heads"] == 8
    assert model["head_dim"] == 128
    assert model["vocab_size"] == 131072
    assert model["bos_token_id"] == 1
    assert model["eos_token_id"] == 2
    assert model["document_end_token_id"] == 18
    assert model["eos_token_id"] != model["document_end_token_id"]
    assert model["pad_token_id"] == 3
    assert model["qk_norm"] is True
    assert model["tie_word_embeddings"] is True
    cfg = ShinraConfig.from_yaml(YAML_PATH)
    assert cfg.hidden_size == 2560
    assert cfg.document_end_token_id == 18
    assert cfg.eos_token_id == 2
    assert cfg.document_end_token_id != cfg.eos_token_id
    assert cfg.num_attention_heads * cfg.head_dim == 4096
    assert cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size


def test_default_config_is_v2_and_decoupled():
    cfg = ShinraConfig()
    assert cfg.hidden_size == 2560
    assert cfg.num_hidden_layers == 36
    assert cfg.num_attention_heads == 32
    assert cfg.head_dim == 128
    assert cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size
    assert cfg.document_end_token_id == 18
    assert cfg.eos_token_id == 2


def test_config_refuses_collapsed_eos_and_document_end():
    with pytest.raises(ValueError, match="document_end"):
        ShinraConfig(document_end_token_id=2)


def test_decoupled_q_width_constructs_attention():
    cfg = ShinraConfig(
        vocab_size=64,
        hidden_size=80,
        intermediate_size=160,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    assert cfg.num_attention_heads * cfg.head_dim == 128
    assert cfg.hidden_size == 80
    attn = ShinraAttention(cfg, layer_idx=0)
    assert attn.q_proj.in_features == 80
    assert attn.q_proj.out_features == 128
    assert attn.o_proj.in_features == 128
    assert attn.o_proj.out_features == 80
    assert attn.k_proj.out_features == 64


def test_coupled_geometry_still_constructs():
    cfg = ShinraConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    assert cfg.num_attention_heads * cfg.head_dim == cfg.hidden_size


def test_gqa_must_divide():
    with pytest.raises(ValueError, match="divisible"):
        ShinraConfig(
            vocab_size=64,
            hidden_size=80,
            intermediate_size=160,
            num_hidden_layers=1,
            num_attention_heads=5,
            num_key_value_heads=2,
            head_dim=32,
        )


def test_v2_attention_projection_shapes():
    cfg = ShinraConfig(num_hidden_layers=1, vocab_size=256, use_cache=False)
    attn = ShinraAttention(cfg, layer_idx=0)
    assert attn.q_proj.in_features == 2560
    assert attn.q_proj.out_features == 4096
    assert attn.k_proj.out_features == 1024
    assert attn.v_proj.out_features == 1024
    assert attn.o_proj.in_features == 4096
    assert attn.o_proj.out_features == 2560


def test_tiny_decoupled_cpu_forward_shapes():
    cfg = ShinraConfig(
        vocab_size=64,
        hidden_size=80,
        intermediate_size=160,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        use_cache=False,
        attention_implementation="eager",
    )
    cfg._attn_implementation = "eager"
    model = ShinraForCausalLM(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    assert tuple(out.logits.shape) == (1, 5, 64)
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_flash2_is_not_advertised():
    from model.modeling_shinra import ShinraPreTrainedModel

    assert ShinraPreTrainedModel._supports_flash_attn_2 is False
    assert ShinraPreTrainedModel._supports_sdpa is True
