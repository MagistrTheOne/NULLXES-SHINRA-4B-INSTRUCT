# Pretraining token stream

The active packer is `data.pack`. It reads the strict cleaned JSONL split and the root corpus manifest, verifies both hashes, encodes records with `add_special_tokens=False`, wraps each document as `[BOS] + body + [END_OF_TEXT]`, and writes fixed-length Parquet token shards with per-shard SHA256 entries.

ID assignments come from `configs/shinra.yaml`: BOS=1, EOT=2, PAD=3, document end=4. EOT and all chat/control tokens are rejected inside document bodies. Padding is masked from loss; document end remains an ordinary causal target. Cross-document attention is currently allowed after packing. No alternate binary shard format is supported.
