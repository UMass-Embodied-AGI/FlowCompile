"""FastAPI server for FlashFlow."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from flashflow.runtime import FlashFlowRuntime


def _handle_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def create_app(runtime: FlashFlowRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await runtime.startup()
        try:
            yield
        finally:
            await runtime.shutdown()

    app = FastAPI(title="FlashFlow", lifespan=lifespan)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        try:
            return JSONResponse(await runtime.handle_chat(body))
        except Exception as exc:
            raise _handle_error(exc) from exc

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        try:
            return JSONResponse(await runtime.handle_completion(body))
        except Exception as exc:
            raise _handle_error(exc) from exc

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse(runtime.list_models())

    @app.post("/v1/flashflow/token_usage/reset")
    async def reset_token_usage():
        return JSONResponse(await runtime.reset_token_usage())

    @app.get("/v1/flashflow/token_usage")
    async def get_token_usage():
        return JSONResponse(await runtime.get_token_usage())

    return app
