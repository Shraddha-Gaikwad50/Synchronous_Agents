"""
Phase 4 — Latency / cost optimization: Haiku intent routing, Sonnet + A2A for metrics,
and Haiku-based session context compression (Vertex AI Claude).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Literal

logger = logging.getLogger(__name__)

# Approximate token budget: use char length / 4 as a cheap stand-in for tokens.
CONTEXT_TOKEN_APPROX_LIMIT = int(os.environ.get("SESSION_CONTEXT_TOKEN_APPROX", "4000"))

ENABLE_VERTEX_ROUTING = os.environ.get("ENABLE_VERTEX_ROUTING", "false").lower() in (
    "1",
    "true",
    "yes",
)

VERTEX_REGION = os.environ.get("VERTEX_AI_LOCATION", "us-east5")
VERTEX_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
    "GCP_PROJECT", ""
)

# Vertex model resource IDs — override to match your region / Model Garden publish names.
VERTEX_CLAUDE_HAIKU_MODEL = os.environ.get(
    "VERTEX_CLAUDE_HAIKU_MODEL",
    "claude-haiku-4-5@20250514",
)
VERTEX_CLAUDE_SONNET_MODEL = os.environ.get(
    "VERTEX_CLAUDE_SONNET_MODEL",
    "claude-sonnet-4-6@20250514",
)


def _approx_token_count(session_history: list[dict[str, str]]) -> int:
    total_chars = sum(len(m.get("content", "")) for m in session_history)
    return max(1, total_chars // 4)


def _get_vertex_client():
    from anthropic import AnthropicVertex

    if not VERTEX_PROJECT:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT or GCP_PROJECT is required for Vertex")
    return AnthropicVertex(region=VERTEX_REGION, project_id=VERTEX_PROJECT)


def _call_claude_sync(
    model: str,
    system: str,
    user_prompt: str,
    max_tokens: int = 1024,
) -> str:
    client = _get_vertex_client()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts: list[str] = []
    for block in msg.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts).strip()


async def call_claude(
    model: str,
    system: str,
    user_prompt: str,
    max_tokens: int = 1024,
) -> str:
    return await asyncio.to_thread(
        _call_claude_sync, model, system, user_prompt, max_tokens
    )


async def compress_session_context(session_history: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Middleware: if history is large, summarize older turns with Haiku into one dense block
    and keep the 3 most recent turns for the orchestrator / classifier.

    Threshold: approximate tokens > CONTEXT_TOKEN_APPROX_LIMIT OR raw char length exceeds that
    value (string-length proxy per Phase 4 spec).
    """
    if not session_history:
        return []

    approx_tok = _approx_token_count(session_history)
    # Token budget: char_length/4 as a cheap token proxy (~16k chars ≈ 4k tokens).
    over = approx_tok > CONTEXT_TOKEN_APPROX_LIMIT

    if not over or len(session_history) <= 3:
        return list(session_history)

    recent = session_history[-3:]
    older = session_history[:-3]
    older_blob = "\n".join(
        f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in older
    )
    if not ENABLE_VERTEX_ROUTING:
        return [
            {
                "role": "user",
                "content": f"[Truncated prior turns — {len(older)} messages omitted]\n",
            },
            *recent,
        ]

    system = (
        "Summarize the following chat turns into one dense bullet summary for an assistant. "
        "Preserve facts, numbers, product names, and cost-related constraints. "
        "Do not greet; output plain text only."
    )
    try:
        summary = await call_claude(
            VERTEX_CLAUDE_HAIKU_MODEL,
            system,
            older_blob,
            max_tokens=800,
        )
    except Exception as e:
        logger.warning("compress_session_context Haiku failed, truncating: %s", e)
        summary = f"[{len(older)} prior messages omitted due to compression error]"

    compressed_prefix = {
        "role": "user",
        "content": f"[COMPRESSED SESSION CONTEXT]\n{summary}",
    }
    return [compressed_prefix, *recent]


IntentLabel = Literal["chitchat", "clear", "metrics", "out_of_scope"]

DEFAULT_OUT_OF_SCOPE_REPLY = (
    "I only help with cloud cost and usage questions — for example spend by service, "
    "environment, trends, budgets, or billing data. I can’t assist with that request, "
    "but ask me anything about your cloud costs."
)


@dataclass
class IntentResult:
    intent: IntentLabel
    reply: str


_JSON_INTENT = re.compile(r"\{[\s\S]*\}")


def _parse_intent_json(text: str) -> IntentResult:
    m = _JSON_INTENT.search(text)
    raw = m.group(0) if m else text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return IntentResult(intent="metrics", reply="")
    intent = str(data.get("intent", "metrics")).lower()
    reply = str(data.get("reply", "")).strip()
    if intent in ("greeting", "hello", "chitchat", "smalltalk"):
        intent_t: IntentLabel = "chitchat"
    elif intent in ("clear", "reset", "new_chat"):
        intent_t = "clear"
    elif intent in ("out_of_scope", "off_topic", "unrelated", "refuse"):
        intent_t = "out_of_scope"
    elif intent in ("metrics", "data", "cost", "complex", "query"):
        intent_t = "metrics"
    else:
        intent_t = "metrics"
    return IntentResult(intent=intent_t, reply=reply)


# --- Local routing (no Vertex): fast heuristics for localhost / tests ---

_CLEAR_LOCAL = re.compile(
    r"^\s*(clear|reset|start\s+over|new\s+chat|forget\s+(everything|history))\s*\.?!*\s*$",
    re.I,
)

_METRICS_LOCAL = re.compile(
    r"\b(cost|costs|spend|spending|billing|bill|bills|invoice|invoices|budget|budgets|usage|"
    r"charge|charges|charged|usd|\$|dollar|cloud\s+cost|cloud\s+spend|finops|"
    r"gcp|google\s+cloud|aws|azure|bigquery|big\s+query|storage|compute|kubernetes|gke|"
    r"cloud\s+run|sql|postgres|postgresql|database|environment|env\b|prod|production|"
    r"staging|dev\b|development|metric|metrics|trend|trends|spike|spikes|anomal|"
    r"yesterday|last\s+week|last\s+month|this\s+week|this\s+month|quarter|service|services|"
    r"resource|resources|skus?|line\s+item|export|table\s+cloud_costs|project)\b",
    re.I,
)

_OFF_TOPIC_LOCAL = re.compile(
    r"\b(weather|forecast|temperature|humidity|recipe|cook\b|baking|restaurant|"
    r"joke|jokes|funny|poem|haiku|write\s+a\s+story|translate\s+this|horoscope|"
    r"politics|election|president\b|sports?\b|football|cricket|basketball|"
    r"stock\s+price|bitcoin|crypto|ethereum|"
    r"debug\s+my\s+code|leetcode|homework|write\s+my\s+essay|"
    r"quantum\s+physics|philosophy|movie|who\s+won\s+the|capital\s+of\s+)\b",
    re.I,
)

_GREETING_LOCAL = re.compile(
    r"^\s*(hi|hello|hey|hiya|good\s+(morning|afternoon|evening)|"
    r"thanks|thank\s+you|thx|ty|ok\s*ok|okay|cheers)\s*[!?.]*\s*$",
    re.I,
)


def classify_intent_local(
    latest_user_message: str,
    _orchestrator_context: list[dict[str, str]],
) -> IntentResult:
    """
    Rule-based router when ENABLE_VERTEX_ROUTING is off (typical local dev).
    Prefer routing cost questions to the specialist; refuse clearly off-topic requests.
    """
    text = latest_user_message.strip()
    if not text:
        return IntentResult(intent="out_of_scope", reply=DEFAULT_OUT_OF_SCOPE_REPLY)

    if _CLEAR_LOCAL.match(text):
        return IntentResult(
            intent="clear",
            reply="Chat cleared. Ask me about your cloud costs when you’re ready.",
        )

    if _METRICS_LOCAL.search(text):
        return IntentResult(intent="metrics", reply="")

    if _OFF_TOPIC_LOCAL.search(text):
        return IntentResult(intent="out_of_scope", reply=DEFAULT_OUT_OF_SCOPE_REPLY)

    if len(text) <= 120 and _GREETING_LOCAL.match(text):
        return IntentResult(
            intent="chitchat",
            reply=(
                "Hi — I’m your cost assistant. Ask about spend, services, environments, "
                "or trends in your cloud usage data."
            ),
        )

    # Short vague lines without cost signals: steer to cost topics instead of hitting the DB.
    if len(text.split()) <= 4 and not _METRICS_LOCAL.search(text):
        return IntentResult(
            intent="chitchat",
            reply=(
                "I can look up your cloud cost and usage metrics. Try asking for top services, "
                "costs by environment, or spend over a date range."
            ),
        )

    return IntentResult(intent="out_of_scope", reply=DEFAULT_OUT_OF_SCOPE_REPLY)


async def classify_intent_haiku(
    orchestrator_context: list[dict[str, str]],
    latest_user_message: str,
) -> IntentResult:
    """
    Lightweight router on claude-haiku (Vertex). Returns intent + optional direct Haiku reply.
    """
    ctx_lines = "\n".join(
        f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
        for m in orchestrator_context
    )
    system = (
        "You classify the user's latest message for a cost-metrics assistant. "
        "Respond with ONLY valid JSON (no markdown): "
        '{"intent":"chitchat"|"clear"|"metrics"|"out_of_scope","reply":"<short assistant reply; empty string if metrics>"} '
        "Use intent=chitchat for brief greetings, thanks, or tiny talk that should NOT query cost data. "
        "Use intent=clear if the user wants to reset or clear the conversation. "
        "Use intent=out_of_scope for requests unrelated to cloud cost/usage/billing "
        "(e.g. weather, recipes, politics, sports, general knowledge, coding homework, creative writing). "
        "Reply for out_of_scope must politely say you only help with cloud cost and usage questions. "
        "Use intent=metrics for anything needing database cost/usage numbers, trends, spikes, budgets, "
        "services, environments, GCP/AWS/Azure spend, SKU or line-item questions, queries that name a "
        "GCP project id (e.g. my-app-prod or org-training-486405), or multi-step reasoning over metrics."
    )
    user_prompt = f"Conversation context:\n{ctx_lines}\n\nLatest user message:\n{latest_user_message}"
    text = await call_claude(
        VERTEX_CLAUDE_HAIKU_MODEL,
        system,
        user_prompt,
        max_tokens=256,
    )
    return _parse_intent_json(text)


async def refine_task_sonnet(
    orchestrator_context: list[dict[str, str]],
    latest_user_message: str,
) -> str:
    """
    claude-sonnet (Vertex) prepares a single natural-language task for the Cost Metrics A2A agent.
    """
    ctx_lines = "\n".join(
        f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
        for m in orchestrator_context
    )
    system = (
        "You are the orchestrator planner. Output ONE concise English instruction for a specialist "
        "that queries a PostgreSQL table of cloud_costs (date, service_name, environment, cost_usd). "
        "Include filters implied by the user. No SQL required — natural language only. "
        "Output plain text only, no JSON."
    )
    user_prompt = (
        f"Context:\n{ctx_lines}\n\nUser request:\n{latest_user_message}\n\nSpecialist task:"
    )
    task = await call_claude(
        VERTEX_CLAUDE_SONNET_MODEL,
        system,
        user_prompt,
        max_tokens=512,
    )
    return task.strip() or latest_user_message


def sse_pack_a2a(task_id: str, state: str, text: str, completed: bool = False) -> str:
    if completed:
        body = {
            "id": task_id,
            "status": {"state": "completed"},
            "artifact": {"parts": [{"text": text}]},
        }
    else:
        body = {
            "id": task_id,
            "status": {
                "state": "working",
                "message": {"role": "agent", "parts": [{"text": text}]},
            },
        }
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


async def stream_synthetic_a2a(full_text: str) -> AsyncIterator[bytes]:
    """Emit A2A-shaped SSE so the existing frontend parser keeps working."""
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    chunk = 160
    for i in range(0, len(full_text), chunk):
        yield sse_pack_a2a(
            task_id, "working", full_text[i : i + chunk], completed=False
        ).encode()
        await asyncio.sleep(0.01)
    yield sse_pack_a2a(task_id, "completed", "", completed=True).encode()


def parse_sse_bytes_to_text(raw: bytes) -> str:
    """Recover assistant-visible text from an accumulated A2A SSE byte stream."""
    buffer = raw.decode("utf-8", errors="replace")
    collected: list[str] = []
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
        status = obj.get("status") if isinstance(obj.get("status"), dict) else {}
        message = status.get("message") if isinstance(status.get("message"), dict) else {}
        artifact = obj.get("artifact") if isinstance(obj.get("artifact"), dict) else {}
        parts = message.get("parts") or artifact.get("parts") or []
        if not isinstance(parts, list):
            continue
        for p in parts:
            if isinstance(p, dict) and p.get("text"):
                collected.append(str(p["text"]))
    return "".join(collected)
