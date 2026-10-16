# Training engine

`training.pretrain` is the single causal pretraining entrypoint. It constructs SHINRA from random initialization and reads architecture, optimizer, schedule, precision, token budget, and checkpoint policy from `configs/shinra.yaml`. The CLI supplies data and artifact paths only.

The initial run is a 20,000,000 target-token diagnostic, not a foundation training claim. The diagnostic and foundation targets are separate YAML budgets selected with `--budget`; the foundation target remains unset until approved. Base pretraining does not repeat the identity phrase or use chat-role examples. BF16, real forward/backward, gradient accumulation, activation checkpointing, finite-loss/gradient checks, held-out evaluation, actual EN/RU supervised-token accounting, token/step metrics, and atomic checkpoint writes are part of the engine. Exhausted per-language row permutations are reshuffled from saved deterministic RNG state; the schedule epoch counts `len(train_split)` selections and does not claim a full traversal when the language mix is imbalanced. Reaching `max_steps` before the target budget records `budget_incomplete` and cannot be reported as complete.

Resume checks architecture, tokenizer, corpus, and train plus validation packed-manifest hashes, objective, stage, selected token budget and training-config hashes. Blind data is not opened by the trainer and is not used for training or hyperparameter selection. It restores model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, and the EN/RU token-mix sampler seed, permutations and counters. Multi-process training is explicitly unavailable until distributed checkpoint and token accounting are implemented. Checkpoint files have a self-hashed manifest with per-file SHA-256 verification. No GPU run has been made; the readiness gates are still closed.

Tokenizer identity is now `tokenizer_bundle_sha256` in checkpoint state. The complete bundle,
including SentencePiece model/vocab, is copied byte-for-byte. Resume refuses missing/tampered
bundle files or a different YAML contract before model allocation. Old tokenizer.json-only
checkpoints and packed v1 inputs are incompatible and must not be relabeled to pass admission.

The engine refuses to start until `configs/shinra.yaml` is marked ready and frozen data/tokenizer hashes, immutable source revisions, hardware support, and complete train/validation artifacts are present. The corpus manifest must describe a distinct blind split, but the trainer does not open its packed files. Do not use Colab for training before those gates and review acceptance.
