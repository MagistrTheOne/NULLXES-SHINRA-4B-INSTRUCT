"""OpenAI-compatible HTTP server for SHINRA-4B-INSTRUCT."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .generate import generate_chat, load_generate_stack

app = FastAPI(title="NULLXES SHINRA", version="1.0.0")
_STATE: dict[str, Any] = {}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "NULLXES-SHINRA-4B-INSTRUCT"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
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
    return {
        "object": "list",
        "data": [{"id": "NULLXES-SHINRA-4B-INSTRUCT", "object": "model", "owned_by": "nullxes"}],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in _STATE}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if "model" not in _STATE:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    messages = [m.model_dump() for m in req.messages]
    if req.stream:
        return StreamingResponse(_stream(req, messages), media_type="text/event-stream")
    text = generate_chat(
        _STATE["model_path"],
        messages,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        tokenizer=_STATE["tokenizer"],
        model=_STATE["model"],
    )
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _stream(req: ChatRequest, messages: list[dict[str, str]]):
    text = generate_chat(
        _STATE["model_path"],
        messages,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        tokenizer=_STATE["tokenizer"],
        model=_STATE["model"],
    )
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    final = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve SHINRA over HTTP")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    _STATE["model_path"] = args.model
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
