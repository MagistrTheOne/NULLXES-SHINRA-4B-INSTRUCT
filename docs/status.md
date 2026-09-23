# NULLXES SHINRA — Research Program status

**Training ledger (23-09-2026):** [`TRAINING_STATUS.md`](TRAINING_STATUS.md). S0 не закрыт. Следующий ход — S0.5 Blind-80, inference only, optimizer запрещён.

**Status:** Architecture Frozen / Tokenizer DNA Frozen / Runtime contract added  
**Repo:** https://github.com/MagistrTheOne/NULLXES-SHINRA-4B-INSTRUCT  
**Hypothesis node:** Google Colab Pro A100 **80GB** (167 GB RAM). RTX 2080 forbidden.  
**RunPod MCP:** not authenticated in the Cursor session — Colab is the current node.

## Frozen

- Decoder-only dense 3,926,076,416 params, GQA 24/8, SwiGLU 9216, RoPE θ=1e6, 8k train / 32k window
- Tokenizer control plane: `<|eot|>` vs `<|end_of_text|>`, JSON tool_call, no `<|tool|>` duplicate
- Chat template with BOS; reasoning policy **latent**

## Phase 0 (this brick)

```
dataset stream (sample-10BT)
 → clean/dedup
 → tokenizer Unigram 131k (once)
 → pack
 → BF16 forward/backward
 → 100M token hypothesis pretrain
 → checkpoint
```

Pilot mix 50/25/15/10, language **en 75 / ru 15 / eu 10**. No Chinese corpora. Disk <150GB. Not FineWeb 1.3T. Not 200B.

## Phase 0.1 (now)

Random `ShinraForCausalLM` on Colab A100 80GB. No `from_pretrained`. Env → param count 3,926,076,416 → tied embeddings → BF16 → forward (2, 2048) → backward → one fused AdamW step → tokenizer DNA stats.

```bash
python -m scripts.phase01_bringup --config configs/pretrain_colab_100m.yaml
```

Architecture YAML is always `configs/shinra_4b.yaml`; `from_yaml` on a train recipe pulls `model:` from that sibling file.

vLLM / SGLang / TokenSpeed — inference (`runtime/`), не FSDP pretrain.

## Roadmap

1. SHINRA-4B v1 — text, RoPE, GQA, SwiGLU, 32k window
2. SHINRA-4B v1.5 — YaRN 128k (after needle eval)
3. SHINRA-4B BASE v0.1 — 20–50B tokens on RunPod / 10TB
4. SHINRA-4B BASE v1 — 100–200B
5. SHINRA-Agent — SGLang/vLLM + tools + repo tokens
6. SHINRA-M — CERBER vision encoder + multimodal adapter

