# Evaluation

Evaluation settings and benchmark task lists live in `configs/shinra.yaml`. No evaluation scores or model capability claims are accepted before an approved checkpoint and frozen held-out data exist.

- `evaluation.perplexity` scores the packed validation split and reports causal held-out cross-entropy and perplexity over non-masked next-token targets.
- `evaluation.continuation_eval` runs deterministic prompt continuations from a caller-supplied EN/RU JSON prompt set. It records generated IDs and unknown-token counts; the prompt set and its provenance must be frozen separately before a result can be compared.
- `evaluation.harness` runs the configured lm-evaluation-harness benchmark groups. It requires a local checkpoint; task data and licenses are handled by lm-eval.
- `evaluation.generate_eval` writes qualitative outputs for manual inspection only. It is not a benchmark score.

The deleted synthetic needle probe and its hard-coded codeword are not evaluation evidence. Long-context capability remains unmeasured until a licensed, held-out task set and a real checkpoint are available.
