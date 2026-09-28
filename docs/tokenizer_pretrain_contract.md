# SHINRA v2 pretrain tokenizer contract

Vocab DNA is frozen: Unigram 131072. This file is the **encode path**, not a retokenize.

## Pretrain wrap

Authoritative packer: `data/pack.py` version `shinra-v2-pack.v1`.

```
encode(text, add_special_tokens=False)
→ [BOS] + ids + [END_OF_TEXT]
```

| token | id | pretrain |
|---|---:|---|
| `<\|bos\|>` | 1 | sequence/document start |
| `<\|end_of_text\|>` | 18 | document stop |
| `<\|pad\|>` | 3 | pad; labels `-100` |
| `<\|eot\|>` | 2 | **forbidden** in packed pretrain |
| chat/code/tool/`<\|document\|>` | 4–17 | **forbidden** in packed pretrain |

`eos_token` stays `<|eot|>` for later generation. Pretrain document stop is **not** eos.

Pack never uses `add_special_tokens=True`. Chat `TemplateProcessing` BOS is for SFT later, not this packer.

If `END_OF_TEXT` is missing, unk, or equal to `eos_token_id`, pack raises. No fallback onto EOT. Training code must use `document_end_token_id` (18), never `config.eos_token_id` (2), as the document terminator.

## Concatenation

Documents are concatenated into fixed-length causal streams. Cross-document attention is allowed. Isolated-document masks are not the v2 default.
