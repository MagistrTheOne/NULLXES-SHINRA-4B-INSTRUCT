# Pretraining token stream

The active packer is `data.pack`. It reads the strict cleaned JSONL split and the root corpus manifest, verifies both hashes, encodes records with `add_special_tokens=False`, wraps each document as `[BOS] + body + [END_OF_TEXT]`, and writes fixed-length Parquet token shards with per-shard SHA256 entries.

ID assignments come from `configs/shinra.yaml`: BOS=1, EOT=2, PAD=3, document end=4. EOT and all chat/control tokens are rejected inside document bodies. BOS supplies context at the start of every independent continuation window and is masked as a target; EOD remains an ordinary causal target only on the final window of a source document. PAD and the first target after PAD are masked. Non-overlapping token offsets cover each source document exactly once, with one attention-isolated row per window and no cross-document packing. Continuation windows have no previous-window context.

Packed schema v3 requires `tokenizer_bundle_sha256` in the split manifest and every row, plus source IDs, chunk offsets, language and provenance hashes.
The loader refuses older packed schemas, audit-only data, mismatched bundle identities, incomplete document windows, missing labels and invalid row masks.
The complete bundle must match the frozen YAML hash before packing; checkpoints retain all bundle
files unchanged. See [the bundle contract](tokenizer.md). These checks have not been executed here.
