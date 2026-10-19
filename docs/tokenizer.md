# Tokenizer contract

`configs/shinra.yaml` defines the provisional 65,536 vocabulary and fixed IDs:
UNK=0, BOS=1, EOT=2, PAD=3, EOD=4; configured controls retain IDs 5–18.

## Text and boundaries

- Unicode normalization is **NFC**. SentencePiece normalization is separately `identity`.
- Equality is `decode(encode(text)) == NFC(text)`, not byte equality with original Unicode.
- Cleaning preserves whitespace, CR/LF, tabs, digits, signs and separators. Numeric spans are checked.
- Native SentencePiece byte fallback and decoding are delegated to the verified SentencePiece model.
  The active bundle is Native-only and does not insert implicit BOS/EOS.
- Literal control strings in ordinary text are rejected, including user messages.
  Literal U+2581 (`▁`) is also rejected because it is reserved by Metaspace. No silent escaping or replacement.
- Use the repository `encode_text` / `encode_chat` helpers. Raw HF `encode` is a low-level
  interface and does not apply the repository's NFC/content-admission policy.
- Packing is one document per row: BOS + body + EOD + right PAD. BOS/PAD labels are masked;
  EOD is a target. Empty bodies, UNK, body control IDs, invalid IDs and overlength documents fail.
- Chat formatting inserts one leading BOS. Context overflow fails; neither left nor right truncation is allowed.
- Chat supports system/user/assistant text and stops on EOT. Document continuation stops on EOD.
  Other controls are suppressed during generation. Tools, reasoning channels and code-role messages
  are unavailable until a distinct tool-handoff/final-turn protocol is designed and trained.

## Candidate build (remote only; not yet run)

Input is a previously audited corpus directory containing `manifest.json`, `train.jsonl`,
`validation.jsonl`, and `blind.jsonl`. Arbitrary text files and permissive JSON fallback were removed.
Malformed records fail with file/line context. All split hashes and document/work/text intersections
are checked before training; validation and blind documents never train the tokenizer.
Near-duplicate/semantic/source-family leakage still needs the upstream corpus audit.

Set `tokenizer.build.max_chars`, `input_sentence_size` (>100), and `num_threads` in the single YAML
following the corpus audit. `max_sentence_bytes` bounds documents before SentencePiece can skip them.
Then, only after authorization:

```text
python -m tokenizer.train_tokenizer --corpus-dir data/cleaned --output-dir tokenizer/artifacts
```

The Native builder writes a `candidate_not_frozen` artifact when YAML has no model SHA. Review its
actual model hash, then freeze `native_model_sha256` in the one YAML and rebuild for a complete bundle.
If the configured hash differs, publication fails and no bundle is emitted. Finalization checks actual
validation text in EN/RU, SentencePiece round-trip, zero UNK, numeric/Unicode/whitespace probes and
chat boundary encoding. These checks are code only; **none has been executed**. No candidate, bundle,
fertility measurements or acceptance result currently exist.

## Bundle identity

Native bundle manifests use schema `nullxes.shinra.native-tokenizer-bundle.v1` and hash these files:

- `tokenizer.model`, `tokenizer.vocab`;
- `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`;
- `tokenizer_stats.json`, `corpus_stats.json`.

Fast conversion is excluded from the active YAML validator and installed tokenizer build path. Native
loading never uses `tokenizer.json` or falls back to Fast.

The Native manifest also contains the configuration contract and provenance. Its `manifest_sha256` is SHA256 of the
sorted, ASCII-escaped JSON object **without** its own hash field. It is not the hash of the
pretty-printed file. After review, freeze this identity in `tokenizer.artifact_sha256` in YAML.
That key now means the entire bundle manifest identity, never just `tokenizer.json`.

The model's SHA is independently pinned as `tokenizer.native_model_sha256`; the loader also reads
the SentencePiece model proto and checks Unigram type, vocabulary size, meta IDs, byte fallback,
digit splitting and `identity` normalization against YAML. The adjacent `.vocab` entries are
checked against model piece order and scores.

Packed v2 split manifests and every shard row record `tokenizer_bundle_sha256`.
Checkpoints copy the complete bundle byte-for-byte and record the same identity in training state.
Missing files, changed bytes/configuration, absent frozen hash, legacy packed v1 and mismatched
resume state cause refusal. Existing experimental shards/checkpoints require rebuilding, not relabeling.

No remote runtime is yet approved or evidenced. This repository revision does not contain a generated
Native bundle or an executed integration report. Bundle loading, tokenizer persistence, packed-shard
roundtrip and downstream behavior remain unverified; corpus quality and blind generalization are
separate gates.
