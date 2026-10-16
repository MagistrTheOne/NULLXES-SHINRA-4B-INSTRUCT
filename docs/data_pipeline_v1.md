# Data Pipeline V1: Colab Runbook and Contracts

This pipeline acquires pinned Hugging Face Parquet sources, audits candidate data, cleans and deduplicates whole documents, assigns deterministic splits, tokenizes with the configured Native SHINRA bundle, and atomically publishes each packed split. It does not approve rights or start training.

## Admission gates

The corpus cleaner stores fixed-size MinHash signatures and LSH bands in SQLite; it does not retain all document shingles in process memory or in the index.

The two configured candidates currently have `status: audit_only`, `rights_review_status: audit_only` and `provenance_review_status: audit_only`. A pinned commit makes an acquisition reproducible; it does not establish permission to train. Wikimedia pages can carry different terms, and the PleIAs public-domain statement is a publisher claim that still needs work-level review. A bounded acquisition (`--max-documents`) is always marked `audit_only`.

Production acquisition refuses sources until both reviews and the source status are approved in `configs/shinra.yaml`. Production cleaning refuses an unapproved source snapshot. The packer can serialize an `audit_only` corpus into explicitly audit-only packed artifacts; they require an explicit `allow_audit_data=True` when loaded directly and are rejected by the training engine. Training remains blocked by the three existing false gates and additionally requires a frozen tokenizer bundle SHA and packed data manifest SHA.

## Colab commands

Run in the repository at `/content/NULLXES-SHINRA-4B-INSTRUCT`. The environment must already contain `pyarrow`, `datasketch`, `huggingface_hub`, `fsspec`, SentencePiece and the SHINRA dependencies. The project declares `huggingface_hub` and `fsspec` in the optional `data` extra; the pipeline does not install or upgrade packages.

Acquire both candidates for an audit snapshot. These commands stream the full pinned Parquet revisions unless `--max-documents` is supplied; they may require tens of gigabytes of local storage for resulting JSONL records.

```bash
python -m data.sources --source-id Wikimedia/wikipedia \
  --output-dir /content/drive/MyDrive/NULLXES/SHINRA/data/raw/wikipedia-v1 --admission audit
python -m data.sources --source-id PleIAs/Russian-PD \
  --output-dir /content/drive/MyDrive/NULLXES/SHINRA/data/raw/russian-pd-v1 --admission audit
```

The current YAML keeps both sources audit-only, so the commands above cannot produce production-admitted data. After separate rights/provenance approval and an approved immutable revision is fixed in the YAML, acquire a fresh production snapshot into a new directory with `--admission production`. Do not relabel or reuse an audit snapshot.

`--max-documents N` caps the number of accepted records. Malformed rows encountered before the Nth accepted record are written to `rejections.jsonl` and do not consume the cap. The manifest records accepted, scanned and rejected totals and per-reason counts. Missing required Parquet columns, unreadable files and pinned-revision mismatches remain fatal.

Prepare an audit corpus from the two acquired snapshots:

```bash
python -m data.numeric_integrity \
  --input /content/drive/MyDrive/NULLXES/SHINRA/data/raw/wikipedia-v1 \
         /content/drive/MyDrive/NULLXES/SHINRA/data/raw/russian-pd-v1 \
  --output-dir /content/drive/MyDrive/NULLXES/SHINRA/data/clean/audit-v1 \
  --admission audit
```

After both sources are approved, acquire fresh production snapshots as above and use `--admission production` with `data.numeric_integrity`. The cleaner stages the complete output beside the destination and publishes its directory atomically. Existing output paths are never overwritten. The destination filesystem must support same-filesystem atomic directory `os.replace`; this behavior has not been verified on the Colab Drive mount, so verify it before a production build or stage on a filesystem where that guarantee holds.

Pack the existing audit corpus into three separate split directories with one command. The command uses the configured Native SentencePiece bundle, validates the corpus SHA pinned in `configs/shinra.yaml`, chunks each document without overlap, and keeps split boundaries separate. All outputs remain `audit_only` and cannot be used by the trainer:

```bash
python -m data.pack \
  --input-dir /content/drive/MyDrive/NULLXES/SHINRA/data/clean/shinra-audit-v2 \
  --tokenizer /content/drive/MyDrive/NULLXES/SHINRA/tokenizer/artifacts/shinra_unigram_65k_v1_verified \
  --output-dir /content/drive/MyDrive/NULLXES/SHINRA/data/packed/shinra-audit-v2 \
  --split all
```

Each split is published atomically and independently. If the all-split command is interrupted, already published split directories remain immutable; resume the interrupted split with its specific `--split train|validation|blind --resume` command, using that split directory as `--output-dir`. The sibling `.<split>.partial` directory contains a self-hashed journal and checksummed completed shards. Processing replays from the last journaled shard boundary, discarding an unjournaled in-progress shard. Input split, corpus manifest, tokenizer bundle and packing configuration must match. A missing/corrupt journal or committed shard stops recovery. An existing published split is immutable.

No `shinra-pretrain` command is approved by this runbook while the three config gates are false. FSDP is unavailable; the current trainer requires one supported Colab GPU and BF16.

## Artifact schemas

| Artifact | Schema | Integrity and contents |
|---|---|---|
| Raw source snapshot | `nullxes.shinra.raw-source.v2` | Exact source revision, source-specific license claim and review status, upstream file path/SHA-256 list, accepted/scanned/rejected counts, rejection reason counts, `records.jsonl` and redacted `rejections.jsonl` SHA-256 values, self-hashed manifest; accepted rows have stable document ID, original text SHA-256, upstream file hash, row position and URL/archive identifier. Rejection rows never contain source text. |
| Clean corpus | `nullxes.shinra.corpus.v2` | Admission, source revision and raw manifest identities, frozen split policy, cleaner/dedup versions, total and per-language counts, train/validation/blind/duplicates/quarantine SHA-256 values, self-hashed manifest. Duplicate and quarantine records preserve source provenance. |
| Packed split | `nullxes.shinra.packed.v3` | Corpus manifest SHA, Native tokenizer bundle SHA, split, per-shard SHA-256, token accounting and chunking contract, self-hashed manifest. Each Parquet row includes source/revision/document IDs, language, split, chunk index/count, half-open token offsets, total document token count and raw/clean/upstream hashes. Training binds only the train and validation packed-manifest hashes; it does not open or evaluate the blind split.

## Chunking and language mix

Each source document is tokenized after deduplication and split assignment. Body token ranges are ordered, half-open, non-overlapping and cover every source token exactly once. Each packed row starts with BOS; the first body token in each independent continuation window is predicted from that window's BOS. Intermediate windows omit EOD; the final window alone has EOD and trains it. BOS is masked, EOD is a target, PAD and post-PAD targets are masked. There is no overlap and no cross-document packing. Continuation windows do not receive previous-window context.

The pretraining sampler uses actual supervised target-token counts from packed EN/RU rows. It schedules the language with the lower normalized cumulative token count, weights the per-row mean loss by its supervised-token count, re-shuffles a language's rows when that language permutation is exhausted, and saves seed, permutations, token counters and cursors in the checkpoint. Its report records row presentations and presentations caused by reusing exhausted language pools. With 50/50 YAML weights the final count is checked against one maximum packed-row target tolerance; counts, fractions and imbalance are recorded. `max_steps` is a hard safety limit: if the token budget is still short, the ledger records `budget_incomplete`, never `complete`.

## External admission work still required

- Legal review of Wikimedia page terms/attribution obligations and Russian-PD work-level public-domain status for the intended deployment territories.
- Provenance review of upstream row IDs, file SHA metadata, archive/page resolvability, and any source-specific redistribution conditions.
- OCR/table quality policy sign-off. Current quarantine signals are deterministic safeguards, not a human quality review.
- Near-duplicate candidate recall/precision review on the real source revisions; configured MinHash similarity is scalable and approximate.
- Fresh production acquisition after review, corpus artifact audit, frozen split hashes, tokenizer bundle SHA and `data.manifest_sha256` approval.
- Same-filesystem atomic rename behavior on the chosen Colab storage mount; otherwise the run must use a separately validated publication filesystem/protocol.
- Separate approval to change `data.training_enabled`, `training.enabled` and `runtime.colab_launch_allowed`; this code does not change them.
