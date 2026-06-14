import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

from app.config import settings
from app.engine import engine
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
)

import logging
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.start()
    yield

logging.basicConfig(level=logging.INFO)

app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.model_path}


@app.get("/metrics")
def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        return StreamingResponse(
            _stream_sse(req), media_type="text/event-stream"
        )

    text = await engine.submit_batch(
        req.messages,
        req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )
    return ChatCompletionResponse(
        model=req.model,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
    )


def _stream_sse(req: ChatCompletionRequest):
    created = int(time.time())
    base = {
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
    }
    for piece in engine.submit_stream(
        req.messages,
        req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    ):
        chunk = {
            **base,
            "choices": [
                {"index": 0, "delta": {"content": piece}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    final = {
        **base,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"