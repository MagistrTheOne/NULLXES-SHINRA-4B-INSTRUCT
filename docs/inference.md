# Inference

Inference accepts only a local checkpoint directory produced by this repository. The loader checks tokenizer size and core IDs, document-end ID, model type, and model vocabulary against `configs/shinra.yaml`; it does not fetch arbitrary Hub checkpoints. There is no default system prompt, so base inference does not precondition the model on an identity statement.

```powershell
python -m inference.generate --model <local-checkpoint> --prompt "Explain grouped-query attention."
python -m inference.chat --model <local-checkpoint>
python -m inference.server --model <local-checkpoint> --port 8000
```

The server exposes `POST /v1/chat/completions`, `GET /v1/models`, and `GET /health`. It reports an empty model list until a checkpoint loads. `stream: true` returns HTTP 501 because token-by-token streaming is not implemented. Usage counters are omitted instead of returning fabricated zeros.

There is no trained checkpoint in the approved lineage yet, so successful generation is not currently available or claimed. Instruction tuning is a later phase in this codebase.

The complete tokenizer bundle and checkpoint training-state identity are verified before loading
model weights. Only system/user/assistant text messages are supported. Content is NFC-normalized;
literal control strings and reserved U+2581 are rejected. Unsupported roles, invalid content and
context overflow return an error (HTTP 422 in the server), without silent truncation.
Chat has one BOS, stops at EOT=2, and suppresses EOD=4 and other controls. Document evaluation
stops at EOD instead. Tool calls need a future handoff/turn protocol. Output whitespace is preserved;
the server distinguishes a real EOT stop from reaching the length limit. None of these paths was run.
