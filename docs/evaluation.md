# Evaluation

Evaluation settings and benchmark task lists live in `configs/shinra.yaml`. No evaluation scores or model capability claims are accepted before an approved checkpoint and frozen held-out data exist.

- `evaluation.perplexity` scores the packed validation split and reports causal held-out cross-entropy and perplexity over non-masked next-token targets.
- `evaluation.continuation_eval` runs deterministic prompt continuations from a caller-supplied EN/RU JSON prompt set. It records generated IDs and unknown-token counts; the prompt set and its provenance must be frozen separately before a result can be compared.
- lm-evaluation-harness / HumanEval / MBPP integration is unavailable. The old wrappers were removed:
  they bypassed bundle/NFC/control validation and passed an unregistered custom model type to
  AutoModel with `trust_remote_code=False`. A reviewed adapter is required before those tasks run;
  task names in YAML are a future evaluation plan.
- `evaluation.generate_eval` writes qualitative outputs for manual inspection only. It is not a benchmark score.

The deleted synthetic needle probe and its hard-coded codeword are not evaluation evidence. Long-context capability remains unmeasured until a licensed, held-out task set and a real checkpoint are available.

The generic continuation evaluator currently stops on EOD. Its prompts must be document-completion probes, not assistant chat turns. Chat generation stops on EOT; mixing these contracts would invalidate the result.
