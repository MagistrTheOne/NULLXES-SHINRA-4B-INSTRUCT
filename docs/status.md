# NULLXES SHINRA — Research Program status

**Architecture contract:** SHINRA **v2** frozen. v1 lineage closed.  
**Tokenizer DNA:** frozen (Unigram 131072, specials 0–18).  
**Weights:** none inherited. Random init on Colab G4, then pretrain.

S0 / S0.5 / P0 остаются R&D evidence в [`TRAINING_STATUS.md`](TRAINING_STATUS.md). Это не фундамент v2.

**Repo:** https://github.com/MagistrTheOne/NULLXES-SHINRA-4B-INSTRUCT  
**Init node:** Google Colab **G4** (RTX PRO 6000 Blackwell). RTX 2080 forbidden.

## Frozen (v2)

- Decoder-only dense **3,969,056,256** params, residual 2560, Q width 4096, GQA 32/8, 36 layers, SwiGLU 9728, RoPE θ=1e6, 8k train / 32k window
- Tokenizer control plane: `<|eot|>` vs `<|end_of_text|>`, JSON tool_call, no `<|tool|>` duplicate
- Chat template with BOS; reasoning policy **latent**

Not frozen yet: optimizer, WSD, packing isolation, RMSNorm numerics, dual-BOS encode path, residual `post_init`.

## Next

```
random ShinraForCausalLM (v2)
 → Colab G4 BF16 init (`scripts.phase01_bringup`)
 → audit open items
 → real pretrain
 → language
 → identity / instruction
```

```bash
python -m scripts.phase01_bringup --config configs/architecture_v2.yaml
```

Architecture YAML is `configs/architecture_v2.yaml` (legacy `configs/shinra_4b.yaml` removed).
