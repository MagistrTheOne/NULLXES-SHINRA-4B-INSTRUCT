# SHINRA lineage invariant

Frozen. Do not soften this in README, model cards, Hub text, or chat.

SHINRA is SHINRA. Created by NULLXES. Compatibility with other model families is not SHINRA's identity or lineage. Hugging Face compatibility is an interface property, not model ancestry.

## What SHINRA is

- NULLXES Language Intelligence Layer
- SHINRA v2 implementation: `ShinraConfig` / `ShinraForCausalLM`
- Random initialization. Own Unigram tokenizer **131072**. Own training lineage
- Geometry: **2560 / 9728 / 36 / 32Q / 8KV / 128**
- PreTrainedModel, SDPA, safetensors, HF Hub are **interfaces**, not parents

## What SHINRA is not

- Not a derived model, a “compatible” model, or a continuation of foreign weights
- Not `from_pretrained` of a foreign checkpoint and not a wrapper around another `*ForCausalLM`
- Not identity-conditioned in S0–S2 (language first; “who are you?” later)

## Token split (do not collapse)

| id | piece | role |
|---:|---|---|
| 1 | `<\|bos\|>` | sequence start |
| 2 | `<\|eot\|>` | HF `eos_token_id`: generation / later chat turn stop. **Forbidden in S0–S2 packed pretrain** |
| 3 | `<\|pad\|>` | pad; labels `-100` |
| 18 | `<\|end_of_text\|>` | **document terminator** for pretrain wrap |

S0–S2 wrap is only `[BOS] + body + [END_OF_TEXT]`. Do not call `chat_template.jinja`. Do not use `config.eos_token_id` as document stop.

Created by **NULLXES**. SHINRA remains SHINRA.
