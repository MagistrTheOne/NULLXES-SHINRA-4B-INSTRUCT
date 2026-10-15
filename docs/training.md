# Training engine

`training.pretrain` is the single causal pretraining entrypoint. It constructs SHINRA from random initialization and reads architecture, optimizer, schedule, precision, token budget, and checkpoint policy from `configs/shinra.yaml`. The CLI supplies data and artifact paths only.

The initial run is a 20,000,000 target-token diagnostic, not a foundation training claim. Base pretraining does not repeat the identity phrase or use chat-role examples. BF16, real forward/backward, gradient accumulation, activation checkpointing, finite-loss/gradient checks, held-out evaluation, token/step metrics, and atomic checkpoint writes are part of the engine.

Resume checks architecture, tokenizer, corpus manifest, objective, stage, and training-config hashes. It restores model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, and a deterministic sampler cursor. Multi-process training is explicitly unavailable until distributed checkpoint and token accounting are implemented. No GPU run has been made; the readiness gates are still closed.

The engine refuses to start until `configs/shinra.yaml` is marked ready and frozen data/tokenizer hashes, immutable source revisions, hardware support, and a complete train/validation/blind corpus are present. Do not use Colab before those gates and review acceptance.
