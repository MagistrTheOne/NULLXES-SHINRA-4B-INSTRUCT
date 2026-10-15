"""NULLXES SHINRA model package — registers Hugging Face Auto classes."""

from __future__ import annotations

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_shinra import ShinraConfig
from .modeling_shinra import ShinraDecoderLayer, ShinraForCausalLM, ShinraModel, ShinraPreTrainedModel

for registry, key, value in (
    (AutoConfig, "nullxes_shinra", ShinraConfig),
    (AutoModel, ShinraConfig, ShinraModel),
    (AutoModelForCausalLM, ShinraConfig, ShinraForCausalLM),
):
    try:
        registry.register(key, value)
    except ValueError:
        pass

__all__ = [
    "ShinraConfig",
    "ShinraModel",
    "ShinraForCausalLM",
    "ShinraPreTrainedModel",
    "ShinraDecoderLayer",
]
