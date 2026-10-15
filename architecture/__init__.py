# NULLXES SHINRA CORE
#
# Decoder-only Transformer — Language Intelligence Layer
# Model: NULLXES SHINRA-4B-INSTRUCT (architecture is configured by configs/shinra.yaml)

from dataclasses import dataclass


@dataclass(frozen=True)
class ShinraSpec:
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 65536
    tie_word_embeddings: bool = True
    qk_norm: bool = True


def count_parameters(spec: ShinraSpec) -> dict[str, int]:
    hidden, intermediate = spec.hidden_size, spec.intermediate_size
    q_width = spec.num_attention_heads * spec.head_dim
    kv_width = spec.num_key_value_heads * spec.head_dim
    attention = hidden * (q_width + 2 * kv_width) + q_width * hidden
    mlp = 3 * hidden * intermediate
    qk_norm = 2 * spec.head_dim if spec.qk_norm else 0
    norms = 2 * hidden + qk_norm
    layer_count = (attention + mlp + norms) * spec.num_hidden_layers
    embeddings = spec.vocab_size * hidden
    lm_head = 0 if spec.tie_word_embeddings else embeddings
    final_norm = hidden
    return {
        "embeddings": embeddings, "attention_per_layer": attention, "mlp_per_layer": mlp,
        "norm_per_layer": norms, "per_layer": attention + mlp + norms, "all_layers": layer_count,
        "lm_head": lm_head, "final_norm": final_norm,
        "total": embeddings + layer_count + lm_head + final_norm,
    }


def from_config(path="configs/shinra.yaml") -> ShinraSpec:
    from configs.loader import load_config

    model = load_config(path)["model"]
    return ShinraSpec(
        hidden_size=int(model["hidden_size"]), intermediate_size=int(model["intermediate_size"]),
        num_hidden_layers=int(model["num_hidden_layers"]), num_attention_heads=int(model["num_attention_heads"]),
        num_key_value_heads=int(model["num_key_value_heads"]), head_dim=int(model["head_dim"]),
        vocab_size=int(model["vocab_size"]), tie_word_embeddings=bool(model["tie_word_embeddings"]),
        qk_norm=bool(model["qk_norm"]),
    )

SHINRA_4B = from_config()
SHINRA_4B_COUNTS = count_parameters(SHINRA_4B)

__all__ = ["SHINRA_4B", "SHINRA_4B_COUNTS", "ShinraSpec", "count_parameters", "from_config"]
