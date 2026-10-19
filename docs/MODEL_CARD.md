# NULLXES SHINRA-4B-INSTRUCT

**Identity:** SHINRA is SHINRA. Created by NULLXES.

This repository is preparing a random-initialized decoder-only language model. It is not a trained or usable instruction checkpoint. The `INSTRUCT` release name is retained for project continuity; instruction tuning has not been performed.

## Design

Architecture and tokenizer IDs are defined only in [`configs/shinra.yaml`](../configs/shinra.yaml). Current provisional design: hidden size 2560, 36 layers, 32 query heads, 8 KV heads, head dimension 128, intermediate size 9728, 32,768 position window, and 65,536 vocabulary entries. Vocabulary and tokenizer remain subject to corpus audit.

Initialization is random. No previous experiment weights are used. New training data is specified to come only from the NULLXES Synthetic Engine; external text corpora and third-party LLM outputs are excluded. The generator, tokenizer, training pipeline, and model capabilities have not been runtime-verified or accepted. No training run is authorized by this model card.

## Status and limitations

- Training gates are disabled pending corpus, tokenizer, and runtime review.
- There is no trained checkpoint, evaluation score, or demonstrated language capability to report.
- Instruction following, identity behavior, long-context retrieval, safety, and deployment performance are unmeasured.
- No Google Colab or GPU run has been performed.

Do not present this repository as a completed model release until training, held-out evaluation, and review are complete.
