# Data preparation and audit status

FineWeb, FineWeb-Edu, FineWeb-2, and SmolLM-Corpus are excluded. Sources listed in `configs/shinra.yaml` are candidates for review, not approved training sources. Revision, license, language, source provenance, and numeric integrity must be checked before any source can be admitted.

`data.numeric_integrity.prepare_local_corpus` strictly decodes local records, checks original and cleaned hashes, preserves numeric spans, performs exact and MinHash near-deduplication, assigns deterministic whole-record splits, and records rejected duplicates in `duplicates.jsonl`. Its manifest hashes each split and the duplicate ledger. This is an audit preparation step; it does not establish license approval, language classification, source-aware markup cleanup, or table-context quality.

After those audits are complete, pack one split at a time:

```powershell
python -m data.numeric_integrity --input <audited-source> --output-dir data/cleaned
python -m data.pack --input-dir data/cleaned --tokenizer tokenizer/artifacts --output-dir data/packed/train --split train
```

Repeat the pack command for `validation` and `blind`, writing to the corresponding subdirectories under `data/packed`. The source corpus manifest is copied to `data/packed/manifest.json`; each packed split has its own manifest and shard checksums. The packer rejects missing rows, malformed provenance, changed numbers, altered manifests, and token IDs that violate the pretraining contract.

Train, validation, and blind splits must be frozen before training. No corpus is currently approved, and `training_enabled` remains false.
