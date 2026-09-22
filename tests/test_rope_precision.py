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

def test_rope_inv_freq_recovers_after_to_empty():
    config = ShinraConfig()
    rope = ShinraRotaryEmbedding(config, device=torch.device("cpu"))

    expected = rope._build_inv_freq(device=torch.device("cpu")).float()

    rope.to_empty(device=torch.device("cpu"))

    assert rope.inv_freq.dtype == torch.float32
    assert torch.isfinite(rope.inv_freq).all()
    torch.testing.assert_close(
        rope.inv_freq,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_rope_forward_ignores_corrupted_nonpersistent_buffer():
    config = _tiny_config()

    rope = ShinraRotaryEmbedding(
        config=config,
        device=torch.device("cpu"),
    )

    # Simulate a loader replacing the non-persistent derived buffer
    # with arbitrary but finite garbage.
    rope.inv_freq = torch.zeros_like(rope.inv_freq)

    x = torch.zeros(
        1,
        7,
        config.hidden_size,
        dtype=torch.float32,
    )

    position_ids = torch.arange(
        7,
        dtype=torch.long,
    ).unsqueeze(0)

    cos_corrupt, sin_corrupt = rope(
        x,
        position_ids,
    )

    reference = ShinraRotaryEmbedding(
        config=config,
        device=torch.device("cpu"),
    )

    cos_reference, sin_reference = reference(
        x,
        position_ids,
    )

    torch.testing.assert_close(
        cos_corrupt,
        cos_reference,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        sin_corrupt,
        sin_reference,
        rtol=0.0,
        atol=0.0,
    )
