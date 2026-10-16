# SHINRA codebase consolidation

**Current state:** audit and validation in progress. Training gates remain closed; no Colab machine has been launched.

## Active design

`configs/shinra.yaml` is the single model/data/tokenizer/training/evaluation/runtime configuration. It defines random initialization, provisional 65,536-entry vocabulary, the 20M-token diagnostic budget, source exclusions, special token IDs, evaluation tasks, and hardware requirements. Candidate datasets remain audit-only.

The token contract is `unk=0`, `bos=1`, `eot=2`, `pad=3`, document end `4`, and other controls `5..18`. EOT is the chat message/final assistant turn stop. EOD is exclusively a pretraining document boundary. BOS is context-only, EOD is trained, and PAD plus the first target after PAD are masked. Each non-overlapping continuation window occupies its own attention-isolated row; every source token appears in one window only, each window starts with a masked BOS, and only the final document window contains EOD.

## Active pipeline

The tokenizer trainer requires strict UTF-8, numeric-span preservation, exact control IDs, held-out probes, and an artifact hash. The data preparation path validates provenance, records exact/near duplicates, assigns deterministic splits, and writes a manifest. The packer consumes one cleaned JSONL split at a time, verifies source hashes and tokenizer round-trips, and emits checksummed Parquet shards. Training has one random-init entrypoint and supports single-process resume with optimizer, scheduler, RNG, sampler cursor, and frozen identity hashes.

Old binary shard handling, synthetic needle evaluation, obsolete stage configs, legacy training status, and stale architecture/tokenizer documentation have been removed or replaced. The OpenAI-compatible server reports unavailable until a compatible local checkpoint loads; streaming remains explicitly unimplemented.

## Remaining readiness gates

No source has passed license/source-level review. The data processor does not yet implement language identification, source-aware markup handling, or table-context quarantine; candidate corpora therefore remain unavailable for training. Tokenizer artifact and corpus manifest hashes are unset. Evaluation data, full training, checkpoint interruption/recovery, and Colab/GPU behavior have not been executed. Do not mark the model ready until those gates pass.
