"""Provide localhost HTTP helpers for realistic integration-style tests."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import threading
import time
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route
import uvicorn


@contextmanager
def run_starlette_server(
    app: Starlette,
    *,
    port: int,
    startup_timeout_seconds: float = 5.0,
) -> Iterator[str]:
    """Run a Starlette app on localhost for the duration of a test."""
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _serve() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(
        target=_serve,
        name=f"test-server-{port}",
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + max(0.1, startup_timeout_seconds)
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError(f"Local test server on port {port} exited during startup.")
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Local test server on port {port} did not start within "
                f"{startup_timeout_seconds:.1f}s."
            )
        time.sleep(0.01)

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)
        if thread.is_alive():
            raise RuntimeError(f"Local test server on port {port} did not stop cleanly.")


@dataclass(slots=True)
class LocalOpenAICompatibleServer:
    """Capture requests and serve queued streaming chat responses."""

    port: int
    model_id: str = "fake-model"
    chat_requests: list[dict[str, Any]] = field(default_factory=list)
    _queued_chunks: deque[list[dict[str, Any]]] = field(default_factory=deque)

    @property
    def base_url(self) -> str:
        """Return the localhost base URL for the running server."""
        return f"http://127.0.0.1:{self.port}"

    def enqueue_stream(self, chunks: Sequence[dict[str, Any]]) -> None:
        """Queue one streamed chat-completions response."""
        self._queued_chunks.append(list(chunks))

    def build_app(self) -> Starlette:
        """Build the Starlette application that serves queued responses."""

        async def _models(_request: Request) -> JSONResponse:
            return JSONResponse(
                {
                    "object": "list",
                    "data": [{"id": self.model_id, "object": "model"}],
                }
            )

        async def _chat_completions(request: Request):
            payload = await request.json()
            self.chat_requests.append(payload)
            if not self._queued_chunks:
                return PlainTextResponse("No queued chat response.", status_code=500)

            chunks = self._queued_chunks.popleft()

            async def _emit() -> Iterator[str]:
                for chunk in chunks:
                    yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_emit(), media_type="text/event-stream")

        return Starlette(
            routes=[
                Route("/models", _models, methods=["GET"]),
                Route("/v1/models", _models, methods=["GET"]),
                Route("/chat/completions", _chat_completions, methods=["POST"]),
                Route("/v1/chat/completions", _chat_completions, methods=["POST"]),
            ]
        )


@contextmanager
def run_openai_compatible_server(
    *,
    port: int,
    model_id: str = "fake-model",
) -> Iterator[LocalOpenAICompatibleServer]:
    """Run a local OpenAI-compatible streaming test server."""
    server = LocalOpenAICompatibleServer(port=port, model_id=model_id)
    with run_starlette_server(server.build_app(), port=port):
        yield server
