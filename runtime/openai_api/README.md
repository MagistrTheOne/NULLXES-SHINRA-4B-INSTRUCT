# OpenAI-compatible API

Phase 0 serving is the in-repo FastAPI app:

```bash
python -m inference.server --model $CKPT --port 8000
```

Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`.

vLLM/SGLang/TokenSpeed expose the same Chat Completions surface once a native SHINRA backend exists. Until then `runtime/*/serve.sh` fall back to this server (`SHINRA_*_BACKEND=transformers_http`).
