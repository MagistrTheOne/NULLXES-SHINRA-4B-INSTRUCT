# Tokenizer contract

`configs/shinra.yaml` defines the provisional 65,536 vocabulary and fixed IDs:
UNK=0, BOS=1, EOT=2, PAD=3, EOD=4; configured controls retain IDs 5–18.

## Text and boundaries

- Unicode normalization is **NFC**. SentencePiece normalization is separately `identity`.
- Equality is `decode(encode(text)) == NFC(text)`, not byte equality with original Unicode.
- Cleaning preserves whitespace, CR/LF, tabs, digits, signs and separators. Numeric spans are checked.
- Native SentencePiece byte fallback and decoding are delegated to the verified SentencePiece model.
  The legacy Fast backend uses its own byte-fallback decoder. Neither backend inserts implicit BOS/EOS.
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

## Candidate build (requires separate authorization)

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

The legacy Fast build stages output separately and publishes it only after full EN/RU validation-split
round-trips, SP/HF ID parity, zero UNK, numeric/Unicode/whitespace probes and BOS/control checks.
These checks are implemented, **not executed** in this revision. Metrics are collected from real
validation records; contract probes never substitute for a corpus or enter training.

## Bundle identity

Native bundle manifests use schema `nullxes.shinra.native-tokenizer-bundle.v1` and hash these files:

- `tokenizer.model`, `tokenizer.vocab`;
- `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`;
- `tokenizer_stats.json`, `corpus_stats.json`.

The original Fast schema v1 remains separately supported and continues to require
`tokenizer.json`. Native loading never uses that file or falls back to Fast.

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

The Native model artifact and validation measurements are reported as user-confirmed Colab results.
This repository revision does not yet contain a generated Native bundle or an executed integration
report. Bundle loading, wrapper parity, packed-shard roundtrip and downstream runtime behavior remain
unverified here; corpus licensing, quality and mix approval are separate gates.
