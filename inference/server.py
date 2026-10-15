"""OpenAI-compatible HTTP interface for a local SHINRA checkpoint."""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from configs.loader import load_config
from .generate import generate_chat, load_generate_stack

_APPROVED = load_config()
_MODEL_ID = _APPROVED["identity"]["release"]
_GENERATION_DEFAULTS = _APPROVED["inference"]
app = FastAPI(title="NULLXES SHINRA", version="1.0.0")
_STATE: dict[str, Any] = {}


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "code"]
    content: str


class ChatRequest(BaseModel):
    model: str = _MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = _GENERATION_DEFAULTS["default_max_new_tokens"]
    temperature: float = _GENERATION_DEFAULTS["default_temperature"]
    top_p: float = _GENERATION_DEFAULTS["default_top_p"]
    stream: bool = False


@app.on_event("startup")
def _startup() -> None:
    path = _STATE.get("model_path")
    if not path:
        return
    tokenizer, model = load_generate_stack(path)
    _STATE["tokenizer"] = tokenizer
    _STATE["model"] = model


@app.get("/v1/models")
def list_models() -> dict:
    models = ([{"id": _MODEL_ID, "object": "model", "owned_by": "nullxes"}]
              if "model" in _STATE else [])
    return {"object": "list", "data": models}


@app.get("/health")
def health() -> dict:
    loaded = "model" in _STATE
    return {"status": "ok" if loaded else "unavailable", "model_loaded": loaded}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> dict:
    if "model" not in _STATE:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if req.model != _MODEL_ID:
        raise HTTPException(status_code=404, detail="Requested model is not served")
    if req.stream:
        raise HTTPException(status_code=501, detail="Token streaming is not implemented")
    if req.max_tokens <= 0:
        raise HTTPException(status_code=422, detail="max_tokens must be positive")
    if req.temperature < 0 or not 0 < req.top_p <= 1:
        raise HTTPException(status_code=422, detail="temperature or top_p is outside the supported range")
    if req.temperature == 0 and req.top_p != 1:
        raise HTTPException(status_code=422, detail="top_p is only applied when temperature is greater than zero")
    messages = [message.model_dump() for message in req.messages]
    text = generate_chat(
        _STATE["model_path"], messages, max_new_tokens=req.max_tokens, temperature=req.temperature,
        top_p=req.top_p, tokenizer=_STATE["tokenizer"], model=_STATE["model"],
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local SHINRA checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        raise SystemExit("server requires a local SHINRA checkpoint directory")
    _STATE["model_path"] = args.model
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
