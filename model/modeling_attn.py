"""SHINRA attention: RoPE (incl. YaRN/NTK), GQA, FlashAttention/SDPA, KV cache."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache

try:
    from .configuration_shinra import ShinraConfig
    from .modeling_norm import ShinraRMSNorm
except ImportError:
    from configuration_shinra import ShinraConfig
    from modeling_norm import ShinraRMSNorm


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_idx: int, max_idx: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if min_idx == max_idx:
        max_idx += 1
    linear_func = (torch.arange(dim, dtype=dtype, device=device) - min_idx) / (max_idx - min_idx)
    return torch.clamp(linear_func, 0, 1)


class ShinraRotaryEmbedding(nn.Module):
    def __init__(self, config: ShinraConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.rope_type = (config.rope_scaling or {}).get("rope_type", "default")
        self.scaling_factor = float((config.rope_scaling or {}).get("factor", 1.0))
        self.original_max_position_embeddings = int(
            (config.rope_scaling or {}).get("original_max_position_embeddings", config.max_position_embeddings)
        )
        self.beta_fast = float((config.rope_scaling or {}).get("beta_fast", 32.0))
        self.beta_slow = float((config.rope_scaling or {}).get("beta_slow", 1.0))
        self.attention_factor = (config.rope_scaling or {}).get("attention_factor")
        inv_freq = self._build_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._attention_scaling = self._compute_attention_scaling()

    def _build_inv_freq(self, device: torch.device | None) -> torch.Tensor:
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
        if self.rope_type == "yarn" and self.scaling_factor > 1.0:
            freq_extra = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
            )
            freq_inter = 1.0 / (
                self.scaling_factor
                * self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
            )
            low, high = _yarn_find_correction_range(
                self.beta_fast,
                self.beta_slow,
                self.dim,
                self.base,
                self.original_max_position_embeddings,
            )
            ramp = _yarn_linear_ramp_mask(low, high, self.dim // 2, device=inv_freq.device, dtype=inv_freq.dtype)
            inv_freq = freq_inter * (1 - ramp) + freq_extra * ramp
        elif self.rope_type == "linear" and self.scaling_factor > 1.0:
            inv_freq = inv_freq / self.scaling_factor
        elif self.rope_type == "ntk" and self.scaling_factor > 1.0:
            base = self.base * (self.scaling_factor ** (self.dim / (self.dim - 2)))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        return inv_freq

    def _compute_attention_scaling(self) -> float:
        if self.attention_factor is not None:
            return float(self.attention_factor)
        if self.rope_type == "yarn" and self.scaling_factor > 1.0:
            return 0.1 * math.log(self.scaling_factor) + 1.0
        return 1.0

    def _rebuild_inv_freq(self) -> None:
        device = self.inv_freq.device
        self.inv_freq = self._build_inv_freq(device=device).to(
            device=device,
            dtype=torch.float32,
        )

    def _apply(self, fn, recurse=True):
        # inv_freq is derived runtime state rather than checkpoint state.
        # Rebuild it after module transformations so operations such as to(),
        # half(), to_empty(), and device moves cannot leave it stale or
        # uninitialized.
        super()._apply(fn, recurse=recurse)
        self._rebuild_inv_freq()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # inv_freq is derived runtime state and is intentionally absent from
        # checkpoints. Some loading/materialization paths can replace a
        # non-persistent buffer with uninitialized storage. Build the
        # authoritative FP32 frequencies at the execution boundary instead
        # of trusting checkpoint-external buffer contents.
        inv_freq = self._build_inv_freq(device=x.device)
        if self.rope_type == "dynamic":
            seq_len = int(position_ids.max().item()) + 1
            if seq_len > self.original_max_position_embeddings:
                base = self.base * (
                    (self.scaling_factor * seq_len / self.original_max_position_embeddings) - (self.scaling_factor - 1)
                ) ** (self.dim / (self.dim - 2))
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=x.device) / self.dim)
                )
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self._attention_scaling
            sin = emb.sin() * self._attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class ShinraAttention(nn.Module):
    """Multi-head grouped-query attention with RoPE, optional QK-norm, KV cache."""

    def __init__(self, config: ShinraConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.scaling = config.attention_multiplier or (self.head_dim ** -0.5)
        self.sliding_window = self._resolve_window(config, layer_idx)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.q_norm = ShinraRMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = ShinraRMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.qk_norm else None

    @staticmethod
    def _resolve_window(config: ShinraConfig, layer_idx: int) -> int | None:
        if config.layer_types and layer_idx < len(config.layer_types):
            if config.layer_types[layer_idx] == "sliding":
                return config.sliding_window
            return None
        return config.sliding_window

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch: int, num_heads: int) -> torch.Tensor:
        return tensor.view(batch, seq_len, num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _project(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: Cache | None,
        cache_position: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        return query_states, key_states, value_states

    def _build_causal_mask(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        batch, _, q_len, _ = query_states.shape
        kv_len = key_states.shape[-2]
        dtype = query_states.dtype
        device = query_states.device
        min_value = torch.finfo(dtype).min
        causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=kv_len - q_len)
        if self.sliding_window is not None:
            window_mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).triu(
                diagonal=kv_len - q_len - self.sliding_window
            )
            causal = causal & window_mask
        mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
        mask = mask.masked_fill(~causal, min_value)
        mask = mask[None, None, :, :].expand(batch, 1, q_len, kv_len)
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                expanded = attention_mask[:, None, None, :].to(dtype=dtype)
                mask = mask.masked_fill(expanded == 0, min_value)
            elif attention_mask.ndim == 4:
                mask = mask + attention_mask.to(dtype=dtype)
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_states, key_states, value_states = self._project(
            hidden_states, position_embeddings, past_key_value, cache_position
        )
        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        causal_mask = self._build_causal_mask(query_states, key_states, attention_mask)
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class ShinraSdpaAttention(ShinraAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                output_attentions=True,
                **kwargs,
            )
        query_states, key_states, value_states = self._project(
            hidden_states, position_embeddings, past_key_value, cache_position
        )
        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)
        causal_mask = None
        is_causal = False
        q_len = query_states.shape[-2]
        kv_len = key_states.shape[-2]
        if attention_mask is not None and attention_mask.ndim == 4:
            causal_mask = attention_mask
        elif attention_mask is not None and attention_mask.ndim == 2 and not torch.all(attention_mask == 1):
            causal_mask = self._build_causal_mask(query_states, key_states, attention_mask)
        elif self.sliding_window is not None:
            causal_mask = self._build_causal_mask(query_states, key_states, attention_mask)
        else:
            is_causal = q_len > 1 and q_len == kv_len
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class ShinraFlashAttention2(ShinraAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        output_attentions: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                output_attentions=True,
                **kwargs,
            )
        from flash_attn import flash_attn_func, flash_attn_varlen_func

        query_states, key_states, value_states = self._project(
            hidden_states, position_embeddings, past_key_value, cache_position
        )
        # flash_attn: [B, S, H, D]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        dropout_p = self.attention_dropout if self.training else 0.0
        softmax_scale = self.scaling
        window_size = (-1, -1) if self.sliding_window is None else (self.sliding_window, 0)
        if cu_seqlens is not None:
            batch, seqlen, n_heads, head_dim = query_states.shape
            query_flat = query_states.reshape(batch * seqlen, n_heads, head_dim)
            key_flat = key_states.reshape(batch * seqlen, self.num_kv_heads, head_dim)
            value_flat = value_states.reshape(batch * seqlen, self.num_kv_heads, head_dim)
            attn_output = flash_attn_varlen_func(
                query_flat,
                key_flat,
                value_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen or seqlen,
                max_seqlen_k=max_seqlen or seqlen,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
            )
            attn_output = attn_output.view(batch, seqlen, n_heads, head_dim)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
            )
        attn_output = attn_output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


SHINRA_ATTENTION_CLASSES: dict[str, type[ShinraAttention]] = {
    "eager": ShinraAttention,
    "sdpa": ShinraSdpaAttention,
    "flash_attention_2": ShinraFlashAttention2,
}
