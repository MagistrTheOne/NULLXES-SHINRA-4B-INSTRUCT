# NULLXES SHINRA CORE — Architecture Specification

**Model:** NULLXES SHINRA-4B-INSTRUCT
**Status:** Historical design note; `configs/shinra.yaml` is authoritative.
**Role:** Language Intelligence Layer  
**Family:** RAIDEN (reasoning) · CERBER (vision) · **SHINRA (language)** · AION (embodied)

SHINRA v1 (32 × 3072, GQA 24/8, Q width = residual) is closed as experimental lineage. No v1 weights, identity checkpoints, P0 optimizer states, or post-training LR decisions are inherited.

SHINRA is a NULLXES implementation: `ShinraConfig` / `ShinraForCausalLM`, an own Unigram tokenizer with provisional 65,536 vocabulary, random initialization, and an own training lineage. Geometry is SHINRA's: residual 2560, Q width 4096 (32×128), 36 layers, GQA 32/8, SwiGLU 9728. Hugging Face `PreTrainedModel` is an interface, not ancestry. Do not load foreign checkpoints into this lineage. The approved configuration source is `configs/shinra.yaml`; this design note is historical and non-authoritative.

**Invariant:** SHINRA is SHINRA. Created by NULLXES. Compatibility with other model families is not SHINRA's identity or lineage. Hugging Face compatibility is an interface property, not model ancestry. Full text: [`docs/SHINRA_INVARIANT.md`](../docs/SHINRA_INVARIANT.md)

Random init → base pretraining (language) → only then a future instruction-tuning phase.

## Target

| Field | Value |
|-------|-------|
| Type | dense decoder-only |
| Parameters | 3.801B (YAML-derived, tied embeddings) |
| Precision | BF16 |
| Train context | 4096 |
| Native context | 32768 |
| Vocabulary | 65,536 Unigram + byte fallback (provisional; YAML is authoritative) |
| Init node | Google Colab **G4** (RTX PRO 6000 Blackwell) |
| Cluster recipe | not frozen in this contract |

## Block

```
x → RMSNorm → GQA + RoPE (+ QK-norm) → residual
  → RMSNorm → SwiGLU MLP              → residual
```

Final: RMSNorm → tied LM head.

Q projection width is `num_attention_heads × head_dim` and **is not required** to equal `hidden_size`.

## Dimensions (from the current YAML snapshot)

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

The closed experimental lineage is not part of this training contract.

Expanded Q (4096 vs residual 2560) is a **SHINRA 4B layout choice**. Q width is not required to equal `hidden_size`.

## Parameter accounting

```
q  = 32 × 128 = 4096
kv =  8 × 128 = 1024

Compute the parameter count directly from the approved YAML:
```

`python -m architecture.param_count` reads `configs/shinra.yaml`.
Meta-device check: `python -m scripts.verify_architecture`.

## Attention

- Grouped-query attention, group size 4 (32 query heads / 8 KV heads).
- `q_proj`: hidden → Q width; `o_proj`: Q width → hidden.
- FlashAttention-2 when `flash-attn` is installed (`attention_implementation: flash_attention_2`).
- Default path: PyTorch SDPA.
- KV cache via Hugging Face `DynamicCache` / `StaticCache`.

## Tokenizer (unchanged DNA)

Do not change vocabulary or special IDs without updating and revalidating `configs/shinra.yaml`.

| ID | Token | Role |
|----|--------|------|
| 0 | `<\|unk\|>` | unknown |
| 1 | `<\|bos\|>` | sequence start |
| 2 | `<\|eot\|>` | turn / generation stop (HF eos) |
| 3 | `<\|pad\|>` | pad |
| 4 | `<\|end_of_text\|>` | document stop in pretrain |

## Training plan (order only)

Optimizer, packing policy, tokenizer behavior, and initialization are governed by `configs/shinra.yaml` and implementation reviews.

| Stage | Output |
|-------|--------|
| Pretrain | random-init SHINRA base diagnostic checkpoint |
| Future | instruction tuning in this codebase, after base pretraining review |

## Historical implementation notes requiring review

The following historical risk notes need reconciliation against the current implementation before training:

1. **RMSNorm numerics** — `ShinraRMSNorm` uses `F.rms_norm` when present (input dtype). Keep variance in fp32 before long pretrain.
2. **Initialization** — `post_init` runs on `ShinraModel` and again on `ShinraForCausalLM`, then `_scaled_residual_init` on `o_proj`/`down_proj`. Audit the path before trusting `0.02/sqrt(2L)`.
3. **Dual BOS** — `chat_template.jinja` emits `bos_token`; SentencePiece→HF `TemplateProcessing` also prepends `<|bos|>` when `add_special_tokens=True`. One authoritative encode path is required.
4. **Pretrain packing** — confirm the effective cross-document attention policy against the approved data contract before enabling training.

Then optimizer / LR / WSD.

## Risks

| Risk | Mitigation |
|------|------------|
| Logit blow-up at 4B | Z-loss 1e-5, QK-norm, residual init (after init audit) |
| Colab G4 activation memory | gradient checkpointing; keep init seq modest |
| Tokenizer unk on code | byte fallback + digit split (DNA already) |
| Vocab retarget | requires a reviewed YAML update and full tokenizer/model validation |

## Implementation order

1. Treat `configs/shinra.yaml` as the only runtime authority.
2. Audit data and tokenizer artifacts.
3. Review this design note for remaining stale assumptions.
4. Enable training only after readiness gates and acceptance review.
