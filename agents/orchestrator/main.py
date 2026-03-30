"""
PA Orchestrator — discovers Cost Agent via Agent Card, proxies A2A SSE to the frontend.
Structured for future multi-agent ADK orchestration; HTTP layer is the integration seam for Phase 1.

Phase 3 — Proactive workflows: Cloud Scheduler POSTs to /proactive/morning-brief with header
X-API-Key (value from env PROACTIVE_API_KEY). Configure the scheduler OIDC or VPC as needed;
the API key is a basic guard against fully public unauthenticated triggers.

Phase 4 — Vertex Claude Haiku/Sonnet routing, session compression, then A2A to Cost Agent when needed.
Set ENABLE_VERTEX_ROUTING=true and GOOGLE_CLOUD_PROJECT; optional X-Session-Id / body.session_id for memory.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import agent_engine_chat

from intelligence import (
    DEFAULT_OUT_OF_SCOPE_REPLY,
    ENABLE_VERTEX_ROUTING,
    IntentResult,
    classify_intent_haiku,
    classify_intent_local,
    compress_session_context,
    parse_sse_bytes_to_text,
    refine_task_sonnet,
    stream_synthetic_a2a,
)
from telemetry import setup_observability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

COST_AGENT_CARD_URL = os.environ.get(
    "COST_AGENT_CARD_URL",
    "http://127.0.0.1:8001/.well-known/agent.json",
)
COST_AGENT_TASKS_URL = os.environ.get(
    "COST_AGENT_TASKS_URL",
    "http://127.0.0.1:8001/tasks/send",
)

AGENT_CARD_RETRY_SECONDS = int(os.environ.get("AGENT_CARD_RETRY_SECONDS", "90"))
AGENT_CARD_RETRY_INTERVAL = float(os.environ.get("AGENT_CARD_RETRY_INTERVAL", "2"))

# Phase 3: required for /proactive/* (e.g. mount from Secret Manager on Cloud Run).
PROACTIVE_API_KEY = os.environ.get("PROACTIVE_API_KEY", "").strip()

# Synthetic user utterance for scheduled morning brief (routed to Cost Agent via A2A).
MORNING_BRIEF_PROMPT = (
    "Summarize the cloud costs for the last 24 hours, highlighting any spikes."
)

SPECIALIST_CARD: dict[str, Any] | None = None

# Short-term session memory (replace with Agent Engine sessions in production).
sessions: dict[str, list[dict[str, str]]] = {}
session_lock = asyncio.Lock()


def send_push_notification(summary_text: str) -> None:
    """Placeholder for email / FCM / Slack; Phase 3 logs only."""
    logger.info(
        "[push_notification] (placeholder) delivered morning brief — %d characters\n%s",
        len(summary_text),
        summary_text,
    )


def _extract_a2a_text(payload: dict[str, Any]) -> str:
    """Pull assistant-visible text from an A2A-style SSE JSON object."""
    status = payload.get("status")
    if not isinstance(status, dict):
        status = {}
    message = status.get("message")
    if not isinstance(message, dict):
        message = {}
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        artifact = {}
    from_msg = message.get("parts")
    from_art = artifact.get("parts")
    parts = from_msg if isinstance(from_msg, list) and from_msg else from_art
    if not isinstance(parts, list):
        return ""
    out: list[str] = []
    for p in parts:
        if isinstance(p, dict) and p.get("text"):
            out.append(str(p["text"]))
    return "".join(out)


async def verify_proactive_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> None:
    if not PROACTIVE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Proactive routes disabled: set PROACTIVE_API_KEY",
        )
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if len(x_api_key) != len(PROACTIVE_API_KEY) or not secrets.compare_digest(
        x_api_key, PROACTIVE_API_KEY
    ):
        raise HTTPException(status_code=401, detail="Invalid API key")


async def run_cost_agent_task_to_completion(message: str) -> str:
    """
    Non-interactive A2A call: POST /tasks/send, consume SSE until stream ends, return concatenated text.
    """
    collected: list[str] = []
    buffer = ""
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            COST_AGENT_TASKS_URL,
            json={"message": message},
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise HTTPException(
                    status_code=502,
                    detail=f"Cost agent error {response.status_code}: {body[:800]}",
                )
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    raw_event, buffer = buffer.split("\n\n", 1)
                    line = next(
                        (ln for ln in raw_event.split("\n") if ln.startswith("data:")),
                        None,
                    )
                    if not line:
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("error"):
                        raise HTTPException(
                            status_code=502,
                            detail=str(obj.get("detail", "cost agent stream error")),
                        )
                    text = _extract_a2a_text(obj)
                    if text:
                        collected.append(text)
    return "".join(collected)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SPECIALIST_CARD
    SPECIALIST_CARD = None
    deadline = time.monotonic() + AGENT_CARD_RETRY_SECONDS
    async with httpx.AsyncClient(timeout=30.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(COST_AGENT_CARD_URL)
                r.raise_for_status()
                SPECIALIST_CARD = r.json()
                logger.info(
                    "Loaded specialist Agent Card: %s",
                    SPECIALIST_CARD.get("name", "(unknown)"),
                )
                break
            except Exception as e:
                logger.warning(
                    "Agent Card fetch failed (%s), retrying in %.1fs...",
                    e,
                    AGENT_CARD_RETRY_INTERVAL,
                )
                await asyncio.sleep(AGENT_CARD_RETRY_INTERVAL)
        if SPECIALIST_CARD is None:
            logger.warning(
                "Could not load Agent Card from %s after ~%ss; /health will show unloaded",
                COST_AGENT_CARD_URL,
                AGENT_CARD_RETRY_SECONDS,
            )
    yield


app = FastAPI(title="PA Orchestrator", version="1.0.0", lifespan=lifespan)

# CORS: comma-separated origins, or "*" for any origin (credentials disabled — OK for tokenless SSE/chat).
# Include 127.0.0.1 and localhost — browsers treat them as different origins; dev servers often use either.
_cors_raw = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).strip()
if _cors_raw == "*":
    _cors_origins: list[str] = ["*"]
    _cors_credentials = False
else:
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
    _cors_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)


class ChatMessage(BaseModel):
    message: str = Field(..., description="User question for the assistant")
    session_id: str | None = Field(
        default=None,
        description="Client session id for compressed multi-turn context",
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "specialist_card_loaded": SPECIALIST_CARD is not None,
        "specialist_name": (SPECIALIST_CARD or {}).get("name"),
        "vertex_routing_enabled": ENABLE_VERTEX_ROUTING,
        "agent_engine_chat_enabled": agent_engine_chat.is_agent_engine_chat_enabled(),
        "orchestrator_engine_resource_set": bool(
            agent_engine_chat.resolved_engine_resource()
        ),
    }


@app.get("/.well-known/orchestrator.json")
async def orchestrator_meta():
    """Lightweight discovery for debugging; not part of A2A spec."""
    return JSONResponse(
        {
            "name": "PA Orchestrator",
            "routesCostQueriesTo": COST_AGENT_TASKS_URL,
            "specialistCardUrl": COST_AGENT_CARD_URL,
            "cardCached": SPECIALIST_CARD,
        }
    )


async def proxy_cost_agent_sse(message: str) -> AsyncIterator[bytes]:
    payload = {"message": message}
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            async with client.stream(
                "POST",
                COST_AGENT_TASKS_URL,
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    err = body.decode("utf-8", errors="replace")
                    yield f"data: {json.dumps({'error': True, 'detail': err, 'status': response.status_code})}\n\n".encode()
                    return
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.RequestError as e:
            yield f"data: {json.dumps({'error': True, 'detail': str(e)})}\n\n".encode()


async def chat_orchestrated_loop(
    body: ChatMessage, session_id: str
) -> AsyncIterator[bytes]:
    """
    Phase 4 main loop:
    1) Append user turn to session.
    2) compress_session_context(...) → bounded context for classifiers / Sonnet.
    3) Haiku intent: chitchat / clear → Haiku-style direct SSE; metrics → Sonnet task + A2A cost agent SSE.
    """
    user_text = body.message.strip()

    async with session_lock:
        hist = sessions.setdefault(session_id, [])
        hist.append({"role": "user", "content": user_text})
        history_snapshot = list(hist)

    compressed = await compress_session_context(history_snapshot)

    if not ENABLE_VERTEX_ROUTING:
        intent = classify_intent_local(user_text, compressed)
    else:
        try:
            intent = await classify_intent_haiku(compressed, user_text)
        except Exception as e:
            logger.warning("Haiku intent classification failed; routing to metrics: %s", e)
            intent = IntentResult(intent="metrics", reply="")

    if intent.intent == "clear":
        async with session_lock:
            sessions[session_id] = []
        reply = intent.reply or "Chat cleared. Ask me anything about your cloud costs."
        async for chunk in stream_synthetic_a2a(reply):
            yield chunk
        return

    if intent.intent == "chitchat":
        reply = intent.reply or (
            "Hi! I can help you explore cloud cost and usage metrics. What would you like to know?"
        )
        async for chunk in stream_synthetic_a2a(reply):
            yield chunk
        async with session_lock:
            sessions.setdefault(session_id, []).append(
                {"role": "assistant", "content": reply}
            )
        return

    if intent.intent == "out_of_scope":
        reply = intent.reply or DEFAULT_OUT_OF_SCOPE_REPLY
        async for chunk in stream_synthetic_a2a(reply):
            yield chunk
        async with session_lock:
            sessions.setdefault(session_id, []).append(
                {"role": "assistant", "content": reply}
            )
        return

    try:
        if ENABLE_VERTEX_ROUTING:
            specialist_task = await refine_task_sonnet(compressed, user_text)
        else:
            specialist_task = user_text
    except Exception as e:
        logger.warning("Sonnet task refinement failed; using raw user message: %s", e)
        specialist_task = user_text

    buf = bytearray()
    async for chunk in proxy_cost_agent_sse(specialist_task):
        buf.extend(chunk)
        yield chunk
    assistant_text = parse_sse_bytes_to_text(bytes(buf))
    async with session_lock:
        sessions.setdefault(session_id, []).append(
            {"role": "assistant", "content": assistant_text}
        )


@app.post("/chat/stream")
async def chat_stream(body: ChatMessage):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    session_id = body.session_id or str(uuid.uuid4())

    if agent_engine_chat.is_agent_engine_chat_enabled():
        stream = agent_engine_chat.stream_chat_via_agent_engine(
            body.message, session_id
        )
    else:
        stream = chat_orchestrated_loop(body, session_id)

    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        },
    )


@app.post("/proactive/morning-brief")
async def proactive_morning_brief(_: None = Depends(verify_proactive_api_key)):
    """
    Phase 3 — invoked by GCP Cloud Scheduler (HTTP POST + X-API-Key).
    Runs the morning prompt through the Cost Metrics specialist (A2A), then notifies (log placeholder).
    """
    summary_text = await run_cost_agent_task_to_completion(MORNING_BRIEF_PROMPT)
    send_push_notification(summary_text)
    return {
        "status": "ok",
        "workflow": "morning-brief",
        "summary_char_count": len(summary_text),
    }


setup_observability(app, "pa-orchestrator")
