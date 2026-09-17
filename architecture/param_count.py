"""Count SHINRA parameters from architecture hyperparameters."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ShinraSpec:
    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_hidden_layers: int = 32
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 131072
    tie_word_embeddings: bool = True
    qk_norm: bool = True


def count_parameters(spec: ShinraSpec) -> dict[str, int]:
    h = spec.hidden_size
    i = spec.intermediate_size
    q = spec.num_attention_heads * spec.head_dim
    kv = spec.num_key_value_heads * spec.head_dim
    attn = h * q + h * kv + h * kv + q * h
    mlp = h * i + h * i + i * h
    qk = (spec.head_dim * 2) if spec.qk_norm else 0
    norms = h * 2
    per_layer = attn + mlp + qk + norms
    layers = per_layer * spec.num_hidden_layers
    embeddings = spec.vocab_size * h
    lm_head = 0 if spec.tie_word_embeddings else spec.vocab_size * h
    final_norm = h
    total = embeddings + layers + lm_head + final_norm
    return {
        "embeddings": embeddings,
        "attention_per_layer": attn,
        "mlp_per_layer": mlp,
        "norm_per_layer": norms + qk,
        "per_layer": per_layer,
        "all_layers": layers,
        "lm_head": lm_head,
        "final_norm": final_norm,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NULLXES SHINRA parameter accounting")
    parser.add_argument("--untied", action="store_true")
    args = parser.parse_args()
    spec = ShinraSpec(tie_word_embeddings=not args.untied)
    counts = count_parameters(spec)
    print(json.dumps({"spec": spec.__dict__, "counts": counts, "billions": counts["total"] / 1e9}, indent=2))


if __name__ == "__main__":
    main()
