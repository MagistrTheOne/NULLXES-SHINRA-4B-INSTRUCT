# SHINRA status

**Status:** data, tokenizer, and runtime audit required. Training and Colab launch are disabled in `configs/shinra.yaml`.

SHINRA is designed for random initialization with the invariant **“SHINRA is SHINRA. Created by NULLXES.”** Historical experiment weights are not allowed as foundation initialization or resume sources. Identity text is excluded from base pretraining; instruction tuning remains a future phase.

The 20M-token budget is a diagnostic run intended to test early language acquisition. It is not a foundation-model training budget. No new training run, approved tokenizer artifact, audited corpus, held-out score, or Colab/GPU result is recorded here.

Current implementation and limitations are summarized in [`CONSOLIDATION.md`](CONSOLIDATION.md). The single active configuration is [`configs/shinra.yaml`](../configs/shinra.yaml).
