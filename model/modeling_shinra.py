"""NULLXES SHINRA core Transformer — Hugging Face compatible decoder-only LM."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

try:
    from .configuration_shinra import ShinraConfig
    from .modeling_attn import SHINRA_ATTENTION_CLASSES, ShinraRotaryEmbedding
    from .modeling_mlp import ShinraMLP
    from .modeling_norm import ShinraRMSNorm
except ImportError:
    from configuration_shinra import ShinraConfig
    from modeling_attn import SHINRA_ATTENTION_CLASSES, ShinraRotaryEmbedding
    from modeling_mlp import ShinraMLP
    from modeling_norm import ShinraRMSNorm

logger = logging.get_logger(__name__)


class ShinraDecoderLayer(nn.Module):
    def __init__(self, config: ShinraConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        attn_impl = getattr(config, "_attn_implementation", None) or config.attention_implementation
        if attn_impl not in SHINRA_ATTENTION_CLASSES:
            attn_impl = "sdpa"
        self.self_attn = SHINRA_ATTENTION_CLASSES[attn_impl](config, layer_idx)
        self.mlp = ShinraMLP(config)
        self.input_layernorm = ShinraRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ShinraRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_dropout = config.residual_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if self.residual_dropout > 0:
            hidden_states = F.dropout(hidden_states, p=self.residual_dropout, training=self.training)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.residual_dropout > 0:
            hidden_states = F.dropout(hidden_states, p=self.residual_dropout, training=self.training)
        hidden_states = residual + hidden_states
        outputs: tuple[torch.Tensor, ...] = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class ShinraPreTrainedModel(PreTrainedModel):
    config_class = ShinraConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ShinraDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _tied_weights_keys = ["lm_head.weight"]

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, ShinraRMSNorm):
            module.weight.data.fill_(1.0)

    def _scaled_residual_init(self) -> None:
        residual_std = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name.endswith("o_proj") or name.endswith("down_proj"):
                module.weight.data.normal_(mean=0.0, std=residual_std)


class ShinraModel(ShinraPreTrainedModel):
    def __init__(self, config: ShinraConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embedding_dropout) if config.embedding_dropout > 0 else nn.Identity()
        self.layers = nn.ModuleList(
            [ShinraDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = ShinraRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = ShinraRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()
        self._scaled_residual_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast | tuple[torch.Tensor, ...]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + hidden_states.shape[1], device=hidden_states.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, hidden_states, cache_position, past_key_values, output_attentions
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        all_self_attns: tuple[torch.Tensor, ...] | None = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in (hidden_states, past_key_values, all_hidden_states, all_self_attns) if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache | None,
        output_attentions: bool,
    ) -> torch.Tensor | None:
        attn_impl = getattr(self.config, "_attn_implementation", self.config.attention_implementation)
        if attn_impl == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static = isinstance(past_key_values, StaticCache)
        target_length = past_key_values.get_max_cache_shape() if using_static else (past_seen + input_tensor.shape[1])
        if attention_mask is not None and attention_mask.ndim == 4:
            return attention_mask
        dtype = input_tensor.dtype
        device = input_tensor.device
        sequence_length = input_tensor.shape[1]
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(dtype)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
        if attn_impl == "sdpa" and not output_attentions and attention_mask is not None:
            if torch.all(attention_mask == 1) and sequence_length > 1:
                return None
        return causal_mask


class ShinraForCausalLM(ShinraPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: ShinraConfig) -> None:
        super().__init__(config)
        self.model = ShinraModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.model._scaled_residual_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: ShinraModel) -> None:
        self.model = decoder

    def get_decoder(self) -> ShinraModel:
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)
        if not return_dict:
            output = (logits,) + (outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = F.cross_entropy(shift_logits.float(), shift_labels, ignore_index=-100)
        z_coeff = getattr(self.config, "z_loss_coefficient", 0.0) or 0.0
        if z_coeff > 0:
            valid = shift_labels != -100
            if torch.any(valid):
                log_z = torch.logsumexp(shift_logits[valid].float(), dim=-1)
                loss = loss + z_coeff * (log_z**2).mean()
        return loss

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        has_cache = past_key_values is not None and past_key_values.get_seq_length() > 0
        if has_cache:
            if cache_position is not None and input_ids.shape[1] > cache_position.shape[0]:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            else:
                input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and not has_cache:
            model_inputs: dict[str, Any] = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}
        if cache_position is None:
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_length, past_length + input_ids.shape[1], device=input_ids.device)
        model_inputs.update(
            {
                "position_ids": kwargs.get("position_ids"),
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
