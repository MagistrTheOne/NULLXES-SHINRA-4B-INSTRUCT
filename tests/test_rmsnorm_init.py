"""RMSNorm numerics and single residual-init path. CPU only."""

from __future__ import annotations

import math

import torch

from model.configuration_shinra import ShinraConfig
from model.modeling_norm import ShinraRMSNorm
from model.modeling_shinra import ShinraForCausalLM, ShinraModel


def _tiny() -> ShinraConfig:
    return ShinraConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        use_cache=False,
        attention_implementation="eager",
    )


def test_rmsnorm_matches_fp32_reference_from_fp16_input():
    torch.manual_seed(0)
    norm = ShinraRMSNorm(8, eps=1e-6)
    x = torch.randn(2, 5, 8, dtype=torch.float16)
    out = norm(x)
    assert out.dtype == torch.float16
    hidden = x.float()
    variance = hidden.pow(2).mean(-1, keepdim=True)
    ref = (norm.weight.float() * (hidden * torch.rsqrt(variance + 1e-6))).to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


def test_rmsnorm_does_not_call_fused_kernel(monkeypatch):
    import torch.nn.functional as F

    def boom(*_args, **_kwargs):
        raise AssertionError("F.rms_norm must not be used")

    monkeypatch.setattr(F, "rms_norm", boom, raising=False)
    norm = ShinraRMSNorm(4)
    y = norm(torch.ones(1, 3, 4, dtype=torch.float16))
    assert torch.isfinite(y).all()


def test_causal_lm_applies_residual_scale_once():
    torch.manual_seed(1)
    cfg = _tiny()
    cfg._attn_implementation = "eager"
    model = ShinraForCausalLM(cfg)
    expected = cfg.initializer_range / math.sqrt(2 * cfg.num_hidden_layers)
    o_std = float(model.model.layers[0].self_attn.o_proj.weight.detach().std())
    down_std = float(model.model.layers[0].mlp.down_proj.weight.detach().std())
    q_std = float(model.model.layers[0].self_attn.q_proj.weight.detach().std())
    assert abs(o_std - expected) < abs(o_std - cfg.initializer_range)
    assert abs(down_std - expected) < abs(down_std - cfg.initializer_range)
    assert abs(q_std - cfg.initializer_range) < abs(q_std - expected)


def test_inner_model_does_not_call_post_init_itself():
    source = ShinraModel.__init__.__code__.co_names
    assert "post_init" not in source
    assert "_scaled_residual_init" not in source
