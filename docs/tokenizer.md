# Tokenizer contract

Tokenizer algorithm, provisional vocabulary size, normalization, numeric rewrite policy, special-token strings, and every token ID are defined in `configs/shinra.yaml`. Current vocabulary size is 65,536 and remains provisional until audited EN/RU corpus slices are available.

| ID | Token | Purpose |
|---:|---|---|
| 0 | `<|unk|>` | Unknown piece; byte fallback should prevent ordinary text from requiring it |
| 1 | `<|bos|>` | Sequence/document start |
| 2 | `<|eot|>` | Generation end-of-turn (`eos_token_id`) |
| 3 | `<|pad|>` | Padding; masked from loss |
| 4 | `<|end_of_text|>` | Pretraining document terminator |
| 5–18 | Configured control tokens | Chat, code, tool, and document metadata; excluded from base pretraining bodies |

Control IDs 5–18 are presently reserved and validated, but not every symbol is used by the base pretraining pipeline. Their future training use requires a concrete data-format contract.

The corpus cleaner canonicalizes text to NFC; SentencePiece then uses identity normalization so it adds no further character transformations. Packer admission requires exact decode(encode(text)) equality. `<|eot|>` and `<|end_of_text|>` are distinct. The packer emits `[BOS] + encoded document + [end_of_text]`; it never uses EOT as a document terminator. Ordinary text is encoded without automatic special tokens. The chat template inserts exactly one BOS explicitly; the tokenizer backend does not add one. Numeric spans must remain byte-for-byte semantically intact through cleanup and decode/encode audit.

The assistant turn stop currently uses `<|eot|>` (ID 2), including the final assistant message. There is no separate end-of-assistant-turn token. `<|end_of_text|>` (ID 4) is reserved for pretraining document boundaries and is not the generation stop token.

Build only from audited local source records:

```powershell
python -m tokenizer.train_tokenizer --input <audited-corpus> --output-dir tokenizer/artifacts --max-chars <approved-cap> --input-sentence-size <approved-sample-size>
```

Tokenizer contract validation is **not run** against a built artifact because no audited corpus or approved tokenizer exists. The build path is configured to fail on invalid UTF-8, numeric-span changes, ID drift, missing control tokens, failed exact-roundtrip probes, or unknown tokens in the holdout sample. It prints an artifact SHA; that value must be reviewed and frozen manually in the single YAML after evaluation. No tokenizer has yet been approved.
