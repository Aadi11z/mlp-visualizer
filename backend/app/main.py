from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import run_agent
from .tools import MLP_URL

app = FastAPI(title="MLP Tutor", version="0.1.0")

# CORS is only needed if someone points a browser at this backend directly
# (e.g. dev tooling). Vite dev server and the production single-origin deploy
# both use the same origin and wouldn't hit CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[dict[str, Any]] | None = None


class AskResponse(BaseModel):
    answer: str
    trace: list[dict[str, Any]]
    stop_reason: str | None = None
    error: str | None = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/ask", response_model=AskResponse)
async def agent_ask(req: AskRequest) -> AskResponse:
    result = await run_agent(req.question, req.history)
    if result.get("error") == "missing_api_key":
        raise HTTPException(status_code=503, detail=result["answer"])
    return AskResponse(
        answer=result["answer"],
        trace=result["trace"],
        stop_reason=result.get("stop_reason"),
    )


# --- MLP proxy -----------------------------------------------------------
# Browsers only talk to the Python origin. Python forwards MLP traffic to the
# Rust server, which listens on localhost and is never exposed externally.

MLP_ALLOWED: set[str] = {"state", "step", "randomize", "set_lr", "forward", "healthz"}


@app.api_route("/mlp/{path:path}", methods=["GET", "POST"])
async def mlp_proxy(path: str, request: Request) -> Response:
    if path not in MLP_ALLOWED:
        raise HTTPException(status_code=404, detail=f"unknown mlp path: {path}")
    body = await request.body()
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.request(
            request.method,
            f"{MLP_URL}/{path}",
            content=body,
            headers={"content-type": request.headers.get("content-type", "application/json")}
            if body
            else {},
        )
    return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


# --- Static frontend (production only) -----------------------------------
FRONTEND_DIST = os.environ.get("FRONTEND_DIST")
if FRONTEND_DIST and Path(FRONTEND_DIST).is_dir():
    # Mounting last so the API routes above win.
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
