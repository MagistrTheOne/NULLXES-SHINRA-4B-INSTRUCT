import torch

from model.configuration_shinra import ShinraConfig
from model.modeling_attn import ShinraRotaryEmbedding


def _tiny_config() -> ShinraConfig:
    return ShinraConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=32768,
        rope_theta=1_000_000.0,
    )


def test_rope_inv_freq_remains_fp32_after_fp16_cast():
    config = _tiny_config()

    rope = ShinraRotaryEmbedding(
        config=config,
        device=torch.device("cpu"),
    )

    assert rope.inv_freq.dtype == torch.float32
    assert torch.isfinite(rope.inv_freq).all()

    rope = rope.to(dtype=torch.float16)

    assert rope.inv_freq.dtype == torch.float32
    assert torch.isfinite(rope.inv_freq).all()
    assert not torch.isnan(rope.inv_freq).any()
    assert not torch.isinf(rope.inv_freq).any()


def test_rope_outputs_remain_finite_after_fp16_cast():
    config = _tiny_config()

    rope = ShinraRotaryEmbedding(
        config=config,
        device=torch.device("cpu"),
    ).to(dtype=torch.float16)

    x = torch.zeros(
        1,
        7,
        config.hidden_size,
        dtype=torch.float16,
    )

    position_ids = torch.arange(
        7,
        dtype=torch.long,
    ).unsqueeze(0)

    cos, sin = rope(x, position_ids)

    assert cos.dtype == torch.float16
    assert sin.dtype == torch.float16

    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()
