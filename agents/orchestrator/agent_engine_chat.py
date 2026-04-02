"""
Forward UI chat to Vertex AI Agent Engine (stream_query) and re-emit A2A-shaped SSE.

Browsers cannot call reasoningEngines:query directly (auth). The FastAPI orchestrator
uses Application Default Credentials and streams results in the format the Next.js UI expects.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import uuid
from typing import Any, AsyncIterator

import vertexai
import vertexai.agent_engines as agent_engines

from intelligence import sse_pack_a2a

logger = logging.getLogger(__name__)

_ORCHESTRATOR_RESOURCE = os.environ.get(
    "ORCHESTRATOR_AGENT_ENGINE_RESOURCE", ""
).strip()
_ORCHESTRATOR_QUERY_URL = os.environ.get(
    "ORCHESTRATOR_AGENT_ENGINE_QUERY_URL", ""
).strip()
_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1").strip()

_ae_sessions: dict[str, tuple[str, str]] = {}
_ae_lock = asyncio.Lock()


def _resource_from_query_url(url: str) -> str:
    m = re.search(
        r"(projects/[^/]+/locations/[^/]+/reasoningEngines/[^/:]+)",
        url,
        re.I,
    )
    return m.group(1) if m else ""


def resolved_engine_resource() -> str:
    if _ORCHESTRATOR_RESOURCE:
        return _ORCHESTRATOR_RESOURCE
    if _ORCHESTRATOR_QUERY_URL:
        return _resource_from_query_url(_ORCHESTRATOR_QUERY_URL)
    return ""


def is_agent_engine_chat_enabled() -> bool:
    # Local dev: UI should hit FastAPI /chat/stream → cost agent on :8001 (BigQuery).
    # Set ORCHESTRATOR_LOCAL_CHAT=1 when ORCHESTRATOR_AGENT_ENGINE_RESOURCE is set but
    # your user/SA lacks reasoningEngines.get/query (otherwise the stream crashes → browser "Network error").
    if os.environ.get("ORCHESTRATOR_LOCAL_CHAT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    return bool(resolved_engine_resource() and _PROJECT)


def _extract_text_from_part(p: dict) -> str:
    if p.get("text"):
        return str(p["text"])
    fc = p.get("function_call")
    if isinstance(fc, dict):
        name = fc.get("name", "")
        args = fc.get("args") if "args" in fc else fc.get("arguments")
        return (
            f"[tool call: {name}] "
            f"{json.dumps(args, ensure_ascii=False) if args is not None else ''}"
        ).strip()
    fr = p.get("function_response")
    if isinstance(fr, dict):
        inner = fr.get("response")
        if inner is not None:
            if isinstance(inner, (dict, list)):
                return json.dumps(inner, ensure_ascii=False)
            return str(inner)
        return json.dumps(fr, ensure_ascii=False)
    if isinstance(fr, str):
        return fr
    return ""


def _extract_text_from_vertex_event(event: dict) -> str:
    content = event.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    out: list[str] = []
    for p in parts:
        if isinstance(p, dict):
            chunk = _extract_text_from_part(p)
            if chunk:
                out.append(chunk)
    return "\n".join(out).strip()


async def _ensure_ui_session(client_session_id: str) -> tuple[str, str]:
    """Map UI session id -> (Agent Engine user_id, engine session id)."""
    async with _ae_lock:
        if client_session_id in _ae_sessions:
            return _ae_sessions[client_session_id]

    resource = resolved_engine_resource()

    def _create() -> tuple[str, str]:
        vertexai.init(project=_PROJECT, location=_LOCATION)
        engine = agent_engines.get(resource)
        user_id = f"ui-{client_session_id}"
        sess = engine.create_session(user_id=user_id)
        sid = sess.get("id") if isinstance(sess, dict) else None
        if not sid:
            raise RuntimeError("Agent Engine create_session returned no session id")
        return user_id, str(sid)

    user_id, engine_sid = await asyncio.to_thread(_create)

    async with _ae_lock:
        if client_session_id in _ae_sessions:
            return _ae_sessions[client_session_id]
        _ae_sessions[client_session_id] = (user_id, engine_sid)
        return user_id, engine_sid


async def _iter_stream_query(
    message: str, user_id: str, engine_session_id: str
) -> AsyncIterator[dict]:
    """Run synchronous stream_query in a worker thread; async-iterate events."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[Any] = asyncio.Queue(maxsize=512)
    _DONE = object()

    resource = resolved_engine_resource()

    def worker() -> None:
        try:
            vertexai.init(project=_PROJECT, location=_LOCATION)
            engine = agent_engines.get(resource)
            for ev in engine.stream_query(
                message=message,
                user_id=user_id,
                session_id=engine_session_id,
            ):
                asyncio.run_coroutine_threadsafe(q.put(ev), loop).result(timeout=180)
            asyncio.run_coroutine_threadsafe(q.put(_DONE), loop).result(timeout=30)
        except Exception as e:
            logger.exception("Agent Engine stream_query failed")
            asyncio.run_coroutine_threadsafe(q.put(("__error__", e)), loop).result(
                timeout=30
            )

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await q.get()
        if item is _DONE:
            return
        if isinstance(item, tuple) and item[0] == "__error__":
            raise item[1]
        if isinstance(item, dict):
            yield item
        else:
            logger.debug("Skipping non-dict Agent Engine event: %s", type(item).__name__)


async def stream_chat_via_agent_engine(
    message: str, client_session_id: str
) -> AsyncIterator[bytes]:
    """
    Stream one user turn through Vertex Agent Engine; output matches frontend SSE parser.
    """
    user_id, engine_sid = await _ensure_ui_session(client_session_id)
    task_id = f"task-{uuid.uuid4().hex[:12]}"

    try:
        async for ev in _iter_stream_query(message.strip(), user_id, engine_sid):
            if ev.get("code"):
                err = f"Agent Engine error {ev.get('code')}: {ev.get('message')}"
                payload = json.dumps({"error": True, "detail": err}, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode()
                return
            delta = _extract_text_from_vertex_event(ev)
            if delta:
                yield sse_pack_a2a(
                    task_id, "working", delta, completed=False
                ).encode()
        yield sse_pack_a2a(task_id, "completed", "", completed=True).encode()
    except Exception as e:
        logger.exception("agent_engine_chat stream failed")
        payload = json.dumps(
            {"error": True, "detail": f"Agent Engine request failed: {e}"},
            ensure_ascii=False,
        )
        yield f"data: {payload}\n\n".encode()
