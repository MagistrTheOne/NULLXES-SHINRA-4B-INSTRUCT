# Pretraining token stream

The active packer is `data.pack`. It reads the strict cleaned JSONL split and the root corpus manifest, verifies both hashes, encodes records with `add_special_tokens=False`, wraps each document as `[BOS] + body + [END_OF_TEXT]`, and writes fixed-length Parquet token shards with per-shard SHA256 entries.

ID assignments come from `configs/shinra.yaml`: BOS=1, EOT=2, PAD=3, document end=4. EOT and all chat/control tokens are rejected inside document bodies. BOS supplies the initial context and is masked as a target; EOD remains an ordinary causal target. PAD and the first target after PAD are masked. Documents occupy separate rows with isolated attention; overlength documents are rejected until a chunking contract is approved. No alternate binary shard format is supported.
