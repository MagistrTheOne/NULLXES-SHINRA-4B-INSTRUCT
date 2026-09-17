---
language:
  - en
  - ru
  - de
  - fr
  - es
  - zh
  - ja
  - ko
  - ar
license: other
license_name: nullxes-research
license_link: LICENSE
tags:
  - nullxes
  - shinra
  - causal-lm
  - decoder-only
  - instruction-tuned
library_name: transformers
pipeline_tag: text-generation
base_model_relation: finetune
---

# NULLXES SHINRA-4B-INSTRUCT

**NULLXES SHINRA-4B-INSTRUCT** is the Language Intelligence Layer of the NULLXES system.

| Layer | Role |
|-------|------|
| RAIDEN | Reasoning Intelligence |
| CERBER | Vision Intelligence |
| **SHINRA** | **Language Intelligence** |
| AION | Embodied Intelligence |

SHINRA is responsible for multilingual understanding, coding intelligence, instruction following, structured outputs, and agent preparation. This checkpoint is the instruction-tuned (and optionally DPO-aligned) 4B-class dense decoder.

## Architecture

Proprietary `ShinraForCausalLM` (not a Llama / Mistral / Qwen / GPT-NeoX wrapper).

| | |
|---|---|
| Type | Decoder-only Transformer |
| Parameters | 3.93B (tied embeddings) |
| Hidden size | 3072 |
| Layers | 32 |
| Attention | GQA 24 query / 8 KV heads, head dim 128 |
| MLP | SwiGLU, intermediate 9216 |
| Norm | RMSNorm, pre-norm + QK-norm |
| Position | RoPE, θ = 1e6, YaRN-ready |
| Context | 8192 train / 32768 native window |
| Vocab | 131072 SentencePiece Unigram + byte fallback |
| Precision | BF16 |
| Attention kernels | PyTorch SDPA Flash / FlashAttention-2 |

Block:

`RMSNorm → GQA+RoPE → residual → RMSNorm → SwiGLU → residual` then final RMSNorm and tied LM head.

Load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "NULLXES/SHINRA-4B-INSTRUCT",
    torch_dtype="bfloat16",
    trust_remote_code=True,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("NULLXES/SHINRA-4B-INSTRUCT", trust_remote_code=True)
messages = [
    {"role": "system", "content": "You are SHINRA, the NULLXES language intelligence layer."},
    {"role": "user", "content": "Explain grouped-query attention."},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
out = model.generate(**tok(prompt, return_tensors="pt").to(model.device), max_new_tokens=256)
```

Special tokens: `<|bos|>` `<|eot|>` `<|system|>` `<|user|>` `<|assistant|>` `<|reasoning|>` `<|code|>` `<|language|>` `<|tool_call|>` `<|tool_response|>` `<|document|>` `<|end_of_text|>`. Generation stop is `<|eot|>`. Document stop is `<|end_of_text|>`.

## Training data

Three stages.

**Pretrain → NULLXES SHINRA-4B-BASE (200B tokens)**  
FineWeb-Edu, Wikipedia (en + ru and additional multilingual dumps), Project Gutenberg / PG19, SmolLM educational + Python-edu, licensed The Stack shards, OpenWebMath, ProofPile-2, peS2o, arXiv CS abstracts. Documents pass ftfy normalization, Gopher/FineWeb quality gates, script/language ID, toxicity heuristics, code AST/minified filters, and MinHash-LSH near-dedup (Jaccard 0.80).

**SFT → NULLXES SHINRA-4B-INSTRUCT**  
Tulu-v2 mixture, OpenHermes-2.5, CodeFeedback, OpenMathInstruct-2, SmolTalk. Packed to 8192 with loss on assistant tokens only.

**DPO**  
UltraFeedback-binarized and cleaned Orca preference pairs, β = 0.10.

Exact mix weights live in `configs/data_mix.yaml`. Tokenizer trained on a ≥10B-character representative sample of the same mix.

## Intended use

- Research and internal NULLXES product integration (language layer behind RAIDEN / AION agents).
- Instruction following, coding assistance, multilingual generation, structured JSON/tool drafts.
- Further domain adaptation by NULLXES.

Out of scope without additional alignment and policy layers: autonomous high-stakes decisions, medical/legal advice, open internet agents with unconstrained tools.

## Limitations

- 3.93B dense capacity: weaker than 70B-class models on multi-hop reasoning and rare languages.
- Pretrain budget 200B tokens is overtrained vs Chinchilla-80B but far below frontier token counts.
- Toxicity and safety filters are heuristic plus optional classifiers; residual harmful content is possible.
- Long context above 8192 uses RoPE extrapolation (YaRN). Always re-run needle-in-haystack after extension.
- Custom architecture requires `trust_remote_code=True` on Hugging Face loaders.

## Evaluation

Run:

```bash
python -m evaluation.perplexity --model $CKPT --data-dir data/packed/pretrain
python -m evaluation.harness --model $CKPT
python -m evaluation.needle --model $CKPT
```

Suite: ARC-Challenge, HellaSwag, WinoGrande, TruthfulQA, MMLU, GSM8K, HumanEval, MBPP, needle-in-haystack at 2k–32k.

## Hardware

Trained for NVIDIA A100 80GB, 8-GPU FSDP FULL_SHARD, BF16, gradient checkpointing, fused AdamW, PyTorch 2.x SDPA.

## License

Source code: NULLXES Research License (see `LICENSE`).  
Weights: proprietary NULLXES asset. Redistribution of checkpoints requires a written grant.

## Citation

```
@misc{nullxes-shinra-4b-instruct,
  title  = {NULLXES SHINRA-4B-INSTRUCT},
  author = {NULLXES Research},
  year   = {2026},
  note   = {Language Intelligence Layer of the NULLXES system}
}
```
