"""NULLXES SHINRA model package — registers Hugging Face Auto classes."""

from __future__ import annotations

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_shinra import ShinraConfig
from .modeling_shinra import ShinraDecoderLayer, ShinraForCausalLM, ShinraModel, ShinraPreTrainedModel

AutoConfig.register("nullxes_shinra", ShinraConfig)
AutoModel.register(ShinraConfig, ShinraModel)
AutoModelForCausalLM.register(ShinraConfig, ShinraForCausalLM)

__all__ = [
    "ShinraConfig",
    "ShinraModel",
    "ShinraForCausalLM",
    "ShinraPreTrainedModel",
    "ShinraDecoderLayer",
]
