# NULLXES SHINRA — Research Program status

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

Pilot mix 50/25/15/10, language 70/15/10/5, disk <150GB. Not FineWeb 1.3T. Not 200B.

## What Qwen3.8-27B is (Hub-verified)

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B): `image-text-to-text`, `Qwen3_5ForConditionalGeneration`, vision encoder, Gated DeltaNet hybrid, MRoPE `[11,11,10]`, native 262k, YaRN to 1M **in vLLM/SGLang/TokenSpeed**. That is a VLM serving stack. SHINRA v1 is language-only.

vLLM/SGLang/TokenSpeed = **inference engines**. SHINRA uses them as `runtime/`, not in FSDP pretrain. Do not copy MRoPE into SHINRA-4B.

## Roadmap

1. SHINRA-4B v1 — text, RoPE, GQA, SwiGLU, 32k window
2. SHINRA-4B v1.5 — YaRN 128k (after needle eval)
3. SHINRA-4B BASE v0.1 — 20–50B tokens on RunPod / 10TB
4. SHINRA-4B BASE v1 — 100–200B
5. SHINRA-Agent — SGLang/vLLM + tools + repo tokens
6. SHINRA-M — CERBER vision encoder + multimodal adapter (MRoPE only then)
