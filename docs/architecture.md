# SHINRA architecture

The approved geometry and token IDs live in [`configs/shinra.yaml`](../configs/shinra.yaml). `model/configuration_shinra.py` consumes that configuration; there is no parallel architecture YAML.

Current design values: decoder-only causal model, hidden size 2560, 36 layers, 32 query heads, 8 KV heads, head dimension 128, SwiGLU intermediate size 9728, tied embeddings, 32,768-token maximum position window, and 65,536 provisional vocabulary entries. Initialization is random. No previous experimental checkpoint is a foundation weight source.

The vocabulary size and tokenizer remain provisional until English/Russian corpus audits and tokenizer round-trip checks are complete. The YAML-derived parameter counter reports 3,801,284,096 parameters for the current geometry; hardware feasibility remains unverified.
# Model architecture

`configs/shinra.yaml` is the single authoritative source for model geometry, vocabulary, token IDs, training, evaluation, inference, and runtime policy. `architecture/design.md` is a historical design note and does not override the YAML.
