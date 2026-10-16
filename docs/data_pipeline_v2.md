# SHINRA EN/RU Corpus Build V2

This pipeline creates a reproducible **audit-only** corpus candidate from immutable Hugging Face revisions. It streams source files, writes provenance and rejection ledgers, performs source-aware cleanup and disk-backed document deduplication, assigns deterministic document splits before token chunking, then packs isolated Native SentencePiece windows. It does not change the source rights status, enable training, or start training.

## Pinned sources and admission

`configs/shinra.yaml` is the exact-revision source allowlist. The configured sources are FineWeb-Edu (English), DCLM-Baseline-1.0 (English), and FineWeb-2 `rus_Cyrl` (Russian). FineWeb-Edu and FineWeb-2 use pinned Parquet files. DCLM uses its pinned original `jsonl.zst` shards, not a generated viewer export. File selection is deterministic and bounded by the configured maximum number of source files.

All sources remain `audit_only`; rights and provenance review are still required. In particular, DCLM's upstream dataset card limits its stated use to research and flags rights/PII review as unresolved. No source is automatically approved by this code. Training gates remain false and the corpus build manifest records `audit_only`.

Candidate acquisition caps are measured using the exact Native tokenizer bundle configured by the build command and include one EOD target per accepted document. They are pre-clean estimates; cleanup, deduplication, and split assignment reduce the final inventory. The build refuses publication unless packed train has at least 500M supervised target tokens for each language. The training sampler separately schedules the actual presented training targets at 50/50 and records language counts/reuse. Corpus inventory is not itself a claim that a training run consumed exactly one billion tokens.

The initial caps are 400M targets from FineWeb-Edu, 400M from DCLM, and 800M from FineWeb-2 Russian. These are candidate acquisition ceilings, not promised clean yields. If the measured clean/packed inventory is below 500M for either language, the command fails and retains the resumable partial build; review measured filtering/dedup loss before increasing caps in a new build destination.

## Colab build and resume

Run from `/content/NULLXES-SHINRA-4B-INSTRUCT`, with the repository dependencies already available. The command does not install or upgrade packages. It requires `pyarrow`, `datasketch`, `huggingface_hub`, `fsspec`, `zstandard`, and the Native tokenizer's existing dependencies. Configure Hugging Face access in the environment if required; do not put tokens in command history.

One production-path build command (still audit-only):

```bash
python -m data.build_corpus \
  --tokenizer /content/drive/MyDrive/NULLXES/SHINRA/tokenizer/artifacts/shinra_native_unigram_65k_bundle_e1876904a4 \
  --output-dir /content/drive/MyDrive/NULLXES/SHINRA/data/build/shinra-enru-1b-v1
```

After an interruption, rerun the same command with `--resume`. The partial directory is beside the output and contains immutable source snapshots, a cursor/checksum journal, clean-corpus dedup state, and completed packed split artifacts. Resume validates identities and hashes; it truncates uncommitted stream tails and never overwrites a completed output. Use a new output directory if changing source caps, revisions, tokenizer bundle, or the cleaning/split contract.

The builder checks available disk space before each phase and pauses while preserving resumable artifacts if free space drops below `runtime.minimum_free_disk_gb` (80 GiB by default). Raw source JSONL, clean records, and all packed splits are retained in the published build for auditability, so this reserve is a minimum safety check, not a full capacity estimate. Confirm the Drive quota and free space before starting. Atomic directory rename semantics depend on the mounted filesystem; for a filesystem without atomic same-filesystem rename, first stage the build on a supported local filesystem and publish/copy using a separately verified integrity procedure.

The result directory contains:

- `raw/<source-id>/`: accepted raw records, redacted rejection ledger, upstream file SHA inventory, pinned revision and acquisition counters.
- `clean/`: cleaned/split train, validation and blind JSONL; duplicates and quarantine ledgers; self-hashed manifest.
- `packed/{train,validation,blind}/`: independently packed Parquet shards and manifests. The training engine only loads train and validation; blind remains held out.
- `build_manifest.json`: source/corpus/packed-manifest SHA identities, measured token totals by language, target status and explicit `audit_only` admission.

Do not pass the audit-only artifacts to pretraining. The future data gate requires legal/provenance approval, a reviewed corpus manifest, frozen manifest SHA in YAML, and explicit approval to change the training gates.

## Integrity and processing contract

- Source rows are streamed from exact pinned revisions; each upstream file's immutable SHA is recorded. Parquet schema/revision/read errors are fatal. Malformed or incomplete JSONL records are rejected into a text-free ledger with source, revision, file, row, reason, and available identifiers.
- Source-specific language labels, script labels, and quality scores are validated against YAML. The DCLM Latin-script check is a script guard, not a semantic language detector.
- HTML/script/style markup is cleaned under a versioned profile. Numeric spans are checked before and after cleaning. Dedup uses disk-backed SQLite MinHash/LSH and preserves duplicate provenance.
- Split assignment is deterministic by source work/document ID and frozen salt before packing. No split is assembled from tokenizer-validation data.
- Packing uses the pinned Native tokenizer, NFC contract, non-overlapping windows no longer than the configured 4096-token sequence, BOS per window, EOD only on the final window, and no cross-document packing. Provenance and half-open source token offsets are stored per row. Each split is packed separately with resumable checksummed shards.
- Packed manifests report exact supervised target counts by language. The trainer's EN/RU sampler uses those counts and persists its state; it does not mix validation or blind into training.

## Upstream source references

- [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [DCLM-Baseline-1.0 dataset card](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0)
- [FineWeb-2 dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)

Dataset cards and license statements can change; the pinned commit, per-file hashes, and a separate current rights review are required for each admission decision.
