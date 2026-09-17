"""NULLXES SHINRA configuration — Hugging Face PretrainedConfig."""

from __future__ import annotations

from typing import Any
import os

from transformers.configuration_utils import PretrainedConfig


class ShinraConfig(PretrainedConfig):
    """Decoder-only SHINRA Transformer configuration.

    Parameter target (tied embeddings): ~3.93B
      - vocab 131072 × hidden 3072
      - 32 layers, GQA 24/8, SwiGLU intermediate 9216
    """

    model_type = "nullxes_shinra"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 3072,
        intermediate_size: int = 9216,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        head_dim: int | None = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 3,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        rope_theta: float = 1_000_000.0,
        rope_scaling: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        mlp_bias: bool = False,
        qk_norm: bool = True,
        sliding_window: int | None = None,
        layer_types: list[str] | None = None,
        attention_implementation: str = "sdpa",
        z_loss_coefficient: float = 1e-5,
        attention_multiplier: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must equal num_attention_heads "
                f"({num_attention_heads}) * head_dim ({self.head_dim})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = dict(rope_scaling) if rope_scaling is not None else {"rope_type": "default"}
        # Default RoPE never uses a scaling factor; HF5 validates this during
        # PretrainedConfig initialization (including legacy YAML/config loads).
        if self.rope_scaling.get("rope_type", "default") == "default":
            self.rope_scaling.pop("factor", None)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout
        self.mlp_bias = mlp_bias
        self.qk_norm = qk_norm
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.attention_implementation = attention_implementation
        self.z_loss_coefficient = z_loss_coefficient
        self.attention_multiplier = attention_multiplier
        self.mlp_hidden_act = hidden_act
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self._validate_rope()

    def _validate_rope(self) -> None:
        rope_type = self.rope_scaling.get("rope_type", "default") if self.rope_scaling else "default"
        allowed = {"default", "linear", "dynamic", "yarn", "ntk"}
        if rope_type not in allowed:
            raise ValueError(f"Unknown rope_type={rope_type}. Allowed: {sorted(allowed)}")

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def _auto_map(self) -> dict[str, str]:
        return {
            "AutoConfig": "configuration_shinra.ShinraConfig",
            "AutoModel": "modeling_shinra.ShinraModel",
            "AutoModelForCausalLM": "modeling_shinra.ShinraForCausalLM",
        }

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["auto_map"] = self._auto_map()
        data["architectures"] = ["ShinraForCausalLM"]
        return data

    def to_diff_dict(self) -> dict[str, Any]:
        data = super().to_diff_dict()
        data["auto_map"] = self._auto_map()
        data["architectures"] = ["ShinraForCausalLM"]
        return data

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> ShinraConfig:
        """Load architecture from YAML.

        Train recipes (e.g. pretrain_colab_100m.yaml) have no `model:` block.
        Architecture always comes from that file's `model:` section or from
        sibling `shinra_4b.yaml`. Training hyperparams are ignored here.
        """
        from pathlib import Path

        import yaml

        yaml_path = Path(path)
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        model_raw = payload.get("model")
        if not model_raw:
            arch_path = yaml_path.parent / "shinra_4b.yaml"
            if not arch_path.exists():
                raise FileNotFoundError(
                    f"{yaml_path} has no model: section and {arch_path} is missing"
                )
            model_raw = (yaml.safe_load(arch_path.read_text(encoding="utf-8")) or {}).get("model")
        if not model_raw:
            raise ValueError(f"No model architecture in {yaml_path}")
        allowed = {
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "hidden_act",
            "max_position_embeddings",
            "initializer_range",
            "rms_norm_eps",
            "use_cache",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "tie_word_embeddings",
            "rope_theta",
            "rope_scaling",
            "attention_bias",
            "attention_dropout",
            "residual_dropout",
            "embedding_dropout",
            "mlp_bias",
            "qk_norm",
            "sliding_window",
            "layer_types",
            "attention_implementation",
            "z_loss_coefficient",
            "attention_multiplier",
        }
        kwargs = {k: v for k, v in model_raw.items() if k in allowed}
        return cls(**kwargs)
