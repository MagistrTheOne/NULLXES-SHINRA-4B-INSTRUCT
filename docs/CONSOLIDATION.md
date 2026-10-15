# SHINRA codebase consolidation

Status: `data_tokenizer_and_runtime_audit_required`. Training and all launchers remain disabled.

## Single configuration

`configs/shinra.yaml` is the single source for model geometry, tokenizer IDs, corpus policy, training, optimization, precision, checkpointing, evaluation, and hardware. `configs/loader.py` validates consistency and refuses training until all dataset and tokenizer audit gates are frozen.

The reserved control IDs are `unk=0`, `bos=1`, `eot=2`, `pad=3`, system/user/assistant and other controls `5..18`, with document termination `<|end_of_text|>=19`. `<|eot|>` is reserved for generation turn boundaries. Vocabulary size remains provisional pending EN/RU corpus audit.

## Data boundary

Only corpus candidates appear in `data.sources`; none is approved for training. FineWeb, FineWeb-Edu, FineWeb-2, and SmolLM-Corpus are excluded. `data.numeric_integrity` performs strict local input parsing, requires source revision, license and original raw SHA, preserves numeric spans through Unicode normalization, deduplicates on cleaned content, assigns whole-document splits, records provenance, and writes per-split hashes. Malformed rows fail the build.

The token packer accepts one explicit split at a time and requires source provenance and content hashes in each Parquet row. Tokenizer encoding must preserve exact text roundtrip and all control IDs before the tokenizer artifact SHA can be frozen.

## Training boundary

`training.pretrain` is the sole training entrypoint. It creates a random-initialized model and refuses to proceed unless hardware, BF16, free disk, training/data gates, tokenizer hash, corpus manifest hash, and distinct train/validation/blind inventories are present. Each optimizer update counts only post-shift target tokens. Loss, gradients, and gradient norm fail on NaN/Inf. Checkpoints are written to a temporary directory and atomically renamed with optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, data cursor, and manifest identities.

Checkpoint resume remains disabled until real immutable data and tokenizer hashes exist; the old cross-stage/shard resume mechanism was removed because its cursor semantics could falsely complete and delete unconsumed shards.

## Identity

`SHINRA is SHINRA. Created by NULLXES.` Model identity remains in the model/config. The base corpus does not repeat this phrase as a training pattern. Instruction tuning is future work under the same codebase and will not start in this pretraining pipeline.

## Validation status

Review tests cover config invariants, EOT/document-end separation, number-preserving cleanup, strict provenance, split separation, packed token counts, and shard checksums. They must pass before a training-readiness decision. No GPU run or Colab launch is part of consolidation.
