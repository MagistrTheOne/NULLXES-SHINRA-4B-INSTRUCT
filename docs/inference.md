# Инференс SHINRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "outputs/shinra-4b-instruct/final/step-00008000"
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="bfloat16", trust_remote_code=True, device_map="auto"
)
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
```

CLI:

```bash
python -m inference.generate --model $CKPT --prompt "Write RMSNorm in PyTorch."
python -m inference.chat --model $CKPT
python -m inference.server --model $CKPT --port 8000
```

HTTP: OpenAI-совместимый `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`.

KV cache включён (`use_cache=True`). Для 32k контекста выставить в конфиге:

```yaml
rope_scaling:
  rope_type: yarn
  factor: 4.0
  original_max_position_embeddings: 8192
```

и прогнать `python -m evaluation.needle --model $CKPT`.
