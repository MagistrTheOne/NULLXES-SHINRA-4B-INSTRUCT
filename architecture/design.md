# NULLXES SHINRA CORE — Architecture Specification

**Model:** NULLXES SHINRA-4B  
**Role:** Language Intelligence Layer  
**Family:** RAIDEN (reasoning) · CERBER (vision) · **SHINRA (language)** · AION (embodied)

SHINRA is a decoder-only dense Transformer designed and implemented by NULLXES. It does not wrap Llama, Mistral, Qwen, or GPT-NeoX classes. Proven components (RMSNorm, RoPE, GQA, SwiGLU, FlashAttention kernels) are used as mathematical primitives inside a proprietary `Shinra*` implementation.

## Target

| Field | Value |
|-------|-------|
| Type | dense decoder-only |
| Parameters | 3.93B (tied embeddings) |
| Precision | BF16 |
| Train context | 8192 |
| Native / expandable context | 32768 (YaRN / NTK / linear rope_scaling) |
| Vocabulary | 131072 Unigram + byte fallback |
| Hardware | 8× NVIDIA A100 80GB (FSDP FULL_SHARD) |

## Block

```
x → RMSNorm → GQA + RoPE (+ QK-norm) → residual
  → RMSNorm → SwiGLU MLP              → residual
```

Final: RMSNorm → tied LM head.

## Dimensions

| Field | Value |
|-------|-------|
| Layers | 32 |
| d_model | 3072 |
| Heads | 24 |
| KV heads (GQA) | 8 |
| Head dim | 128 |
| SwiGLU intermediate | 9216 |
| RoPE θ | 1e6 |
| QK-norm | yes |
| Bias | none |
| Z-loss | 1e-5 |

## Parameter accounting

```
embeddings (tied): 131072 × 3072                 = 402,653,184
attention / layer: Q 9.44M + K 3.15M + V 3.15M + O 9.44M = 25,165,824
SwiGLU / layer:    3 × 3072 × 9216               = 84,934,656
norms / layer:     2 × 3072 + 2 × 128            = 6,400
32 layers                                        = 3,523,419,648
final RMSNorm                                    = 3,072
total                                            = 3,926,075,904  (~3.93B)
```

Recompute with `python -m architecture.param_count`.

## Attention

- Grouped-query attention, group size 3 (24 query heads / 8 KV heads).
- FlashAttention-2 when `flash-attn` is installed (`attention_implementation: flash_attention_2`).
- Default training path: PyTorch 2.x SDPA, which selects the A100 FlashAttention backend.
- KV cache via Hugging Face `DynamicCache` / `StaticCache`.
- Optional hybrid sliding-window via `layer_types` + `sliding_window` for 32k inference cost control.

## Training plan

| Stage | Output | Tokens / steps | LR |
|-------|--------|----------------|----|
| Pretrain | SHINRA-4B-BASE | 200B tokens, WSD, 2,097,152 tok/step | 3e-4 |
| SFT | SHINRA-4B-INSTRUCT | packed assistant-masked chat | 2e-5 |
| DPO | SHINRA-4B-INSTRUCT (aligned) | preference pairs, β=0.1 | 5e-7 |

Chinchilla-optimal for 4B is ~80B tokens. SHINRA overtrains to 200B on a high-quality mix (FineWeb-Edu, Wikipedia, licensed code, OpenWebMath, peS2o, ProofPile) because recent dense 3–4B models gain from compute-heavy data.

## Risks

| Risk | Mitigation |
|------|------------|
| Logit blow-up at 4B | Z-loss 1e-5, QK-norm, residual init `0.02/sqrt(2L)` |
| A100 activation memory at 8k | gradient checkpointing, micro-batch 2, FSDP |
| Long-context degradation | train 8k with θ=1e6, extend with YaRN, needle eval |
| Tokenizer unk on code | byte fallback + digit split + 10B-char representative mix |
| Data contamination | MinHash LSH 0.80, exact SHA256, source license gates |

## Implementation order (this repository)

1. Spec + param count (`architecture/`)
2. Model code (`model/`) — HF `AutoModelForCausalLM`
3. Tokenizer training (`tokenizer/`)
4. Data mix, clean, pack (`data/`)
5. Pretrain → SFT → DPO (`training/`)
6. Eval + Hub release (`evaluation/`, `scripts/`)
