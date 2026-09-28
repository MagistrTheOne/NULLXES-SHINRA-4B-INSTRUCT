# NULLXES SHINRA CORE — Architecture Specification

**Model:** NULLXES SHINRA-4B  
**Contract:** **v2**  
**Role:** Language Intelligence Layer  
**Family:** RAIDEN (reasoning) · CERBER (vision) · **SHINRA (language)** · AION (embodied)

SHINRA v1 (32 × 3072, GQA 24/8, Q width = residual) is closed as experimental lineage. No v1 weights, identity checkpoints, P0 optimizer states, or post-training LR decisions are inherited.

SHINRA v2 is a NULLXES implementation: `ShinraConfig` / `ShinraForCausalLM`, own Unigram tokenizer 131072, random initialization, own training lineage. Geometry is SHINRA's: residual 2560, Q width 4096 (32×128), 36 layers, GQA 32/8, SwiGLU 9728. Hugging Face `PreTrainedModel` is an interface, not ancestry. Do not load foreign checkpoints into this lineage.

**Invariant:** SHINRA is SHINRA. Created by NULLXES. Compatibility with other model families is not SHINRA's identity or lineage. Hugging Face compatibility is an interface property, not model ancestry. Full text: [`docs/SHINRA_INVARIANT.md`](../docs/SHINRA_INVARIANT.md)

Random init → real pretraining (language) → only then identity / instruction. S0–S2 must not identity-condition.

## Target

| Field | Value |
|-------|-------|
| Type | dense decoder-only |
| Parameters | 3,969,056,256 (tied embeddings) |
| Precision | BF16 |
| Train context (S0–S2 G4) | 2048 |
| Train context (cluster recipe, not this cycle) | 8192 |
| Native context | 32768 |
| Vocabulary | 131072 Unigram + byte fallback |
| Init node | Google Colab **G4** (RTX PRO 6000 Blackwell) |
| Cluster recipe | not frozen in this contract |

## Block

```
x → RMSNorm → GQA + RoPE (+ QK-norm) → residual
  → RMSNorm → SwiGLU MLP              → residual
```

Final: RMSNorm → tied LM head.

Q projection width is `num_attention_heads × head_dim` and **is not required** to equal `hidden_size`.

## Dimensions (v2)

| Field | Value |
|-------|-------|
| Layers | 36 |
| d_model (residual) | 2560 |
| Q heads | 32 |
| KV heads (GQA) | 8 |
| Head dim | 128 |
| Q width | 4096 |
| KV width | 1024 |
| SwiGLU intermediate | 9728 |
| RoPE θ | 1e6 |
| QK-norm | yes |
| Bias | none |
| Z-loss | 1e-5 |
| bos / eos / pad | 1 / 2 / 3 |

v1 (closed): 32 layers, residual 3072, Q width 3072, 24Q/8KV, SwiGLU 9216, 3,926,076,416 params. Mathematically valid. Not this contract.

Expanded Q (4096 vs residual 2560) is a **SHINRA 4B layout choice**. Q width is not required to equal `hidden_size`.

## Parameter accounting

```
q  = 32 × 128 = 4096
kv =  8 × 128 = 1024

embeddings (tied): 131072 × 2560                         = 335,544,320
attention / layer: Q 10.49M + K 2.62M + V 2.62M + O 10.49M
                 = 2560×4096 + 2560×1024 + 2560×1024 + 4096×2560
                 = 26,214,400
SwiGLU / layer:    3 × 2560 × 9728                       = 74,711,040
norms / layer:     2 × 2560 + 2 × 128                    = 5,376
per layer:                                               = 100,930,816
36 layers:                                               = 3,633,509,376
final RMSNorm:                                           = 2,560
total:                                                   = 3,969,056,256
```

Recompute with `python -m architecture.param_count`.  
Meta-device check: `python -m scripts.verify_architecture`.

## Attention

- Grouped-query attention, group size 4 (32 query heads / 8 KV heads).
- `q_proj`: hidden → Q width; `o_proj`: Q width → hidden.
- FlashAttention-2 when `flash-attn` is installed (`attention_implementation: flash_attention_2`).
- Default path: PyTorch SDPA.
- KV cache via Hugging Face `DynamicCache` / `StaticCache`.

## Tokenizer (unchanged DNA)

Do not retarget vocab. SHINRA Unigram 131072 and specials 0–18 stay frozen.

| ID | Token | Role |
|----|--------|------|
| 0 | `<\|unk\|>` | unknown |
| 1 | `<\|bos\|>` | sequence start |
| 2 | `<\|eot\|>` | turn / generation stop (HF eos) |
| 3 | `<\|pad\|>` | pad |
| 18 | `<\|end_of_text\|>` | document stop in pretrain |

## Training plan (order only)

Optimizer, WSD, packing isolation, RMSNorm numerics, dual-BOS, and residual init path are **not** frozen here. They are the gate before the first v2 pretrain.

| Stage | Output |
|-------|--------|
| Init | random `ShinraForCausalLM` on Colab G4 |
| Pretrain | SHINRA-4B-BASE |
| SFT | SHINRA-4B-INSTRUCT |
| DPO | aligned instruct |

## Open before first v2 pretrain

These are implementation risks, not the geometry contract:

1. **RMSNorm numerics** — `ShinraRMSNorm` uses `F.rms_norm` when present (input dtype). Keep variance in fp32 before long pretrain.
2. **Initialization** — `post_init` runs on `ShinraModel` and again on `ShinraForCausalLM`, then `_scaled_residual_init` on `o_proj`/`down_proj`. Audit the path before trusting `0.02/sqrt(2L)`.
3. **Dual BOS** — `chat_template.jinja` emits `bos_token`; SentencePiece→HF `TemplateProcessing` also prepends `<|bos|>` when `add_special_tokens=True`. One authoritative encode path is required.
4. **Pretrain packing** — `data/pack.py` concatenates documents into one causal stream (`attention_mask` all ones). Decide isolated vs concatenated documents as a pretrain contract, separately from closed P0.

Then optimizer / LR / WSD.

## Risks

| Risk | Mitigation |
|------|------------|
| Logit blow-up at 4B | Z-loss 1e-5, QK-norm, residual init (after init audit) |
| Colab G4 activation memory | gradient checkpointing; keep init seq modest |
| Tokenizer unk on code | byte fallback + digit split (DNA already) |
| Vocab retarget | forbidden — keep 131072 |

## Implementation order

1. Spec + param count (`architecture/`) — this file
2. Config allows decoupled Q (`ShinraConfig`)
3. Model code (`model/`) — already decoupled in attention
4. Random init on Colab G4 (`scripts/phase01_bringup.py`)
5. Tokenizer DNA unchanged (`tokenizer/`)
6. Pretrain after the open-items audit
