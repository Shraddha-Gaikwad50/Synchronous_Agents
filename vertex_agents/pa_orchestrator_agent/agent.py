"""Platform assistant orchestrator: routes cost questions to deployed Agent Engine."""

from __future__ import annotations

import json
import os
import re
import uuid

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
import vertexai
import vertexai.agent_engines as agent_engines

_ORCHESTRATOR_INSTRUCTION = """You are a platform assistant for GCP operations and cost visibility.
- When the user asks about cloud spend, costs, services, environments (prod/dev), or usage trends, call the query_cost_specialist tool.
- For greetings or general platform guidance not requiring live cost data, answer directly.
- Keep answers concise; when the specialist returns data, synthesize clearly.
- If the tool returns an error, explain the failure and suggest checking specialist configuration."""

_QUERY_URL = os.environ.get("COST_AGENT_QUERY_ENDPOINT", "").strip()
_RESOURCE_NAME = os.environ.get("COST_AGENT_ENGINE_RESOURCE", "").strip()
_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1").strip()


def _resource_from_endpoint(url: str) -> str:
    m = re.search(
        r"(projects/[^/]+/locations/[^/]+/reasoningEngines/[^/:]+)",
        url,
        re.I,
    )
    return m.group(1) if m else ""


def _resolve_resource_name() -> str:
    if _RESOURCE_NAME:
        return _RESOURCE_NAME
    if _QUERY_URL:
        return _resource_from_endpoint(_QUERY_URL)
    return ""


def _extract_text_from_part(p: dict) -> str:
    """Pull human-readable or tool output text from one Gemini content part."""
    if p.get("text"):
        return str(p["text"])
    fc = p.get("function_call")
    if isinstance(fc, dict):
        name = fc.get("name", "")
        args = fc.get("args") if "args" in fc else fc.get("arguments")
        return f"[tool call: {name}] {json.dumps(args, ensure_ascii=False) if args is not None else ''}".strip()

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


def _extract_text(event: dict) -> str:
    content = event.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    out: list[str] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        chunk = _extract_text_from_part(p)
        if chunk:
            out.append(chunk)
    return "\n".join(out).strip()


def _summarize_events_for_empty_response(events: list) -> str:
    """When no extractable text was found, explain what the stream contained."""
    if not events:
        return "Specialist stream returned no events (empty iterator)."
    lines: list[str] = ["Specialist returned no model text; raw stream summary:"]
    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            lines.append(f"  [{i}] {type(ev).__name__}: {repr(ev)[:500]}")
            continue
        if ev.get("code"):
            lines.append(f"  [{i}] error: {ev.get('code')} — {ev.get('message')}")
            continue
        keys = sorted(ev.keys())
        raw = json.dumps(ev, ensure_ascii=False, default=str)
        if len(raw) > 1500:
            raw = raw[:1500] + "…"
        lines.append(f"  [{i}] keys={keys}: {raw}")
    return "\n".join(lines)


def query_cost_specialist(question: str) -> str:
    """Query deployed cost specialist Agent Engine and return its response text."""
    resource_name = _resolve_resource_name()
    if not resource_name:
        return (
            "Cost routing is disabled: set COST_AGENT_QUERY_ENDPOINT or "
            "COST_AGENT_ENGINE_RESOURCE."
        )
    if not _PROJECT:
        return "Cost routing is disabled: set GOOGLE_CLOUD_PROJECT."
    try:
        vertexai.init(project=_PROJECT, location=_LOCATION)
        engine = agent_engines.get(resource_name)
        user_id = f"pa-orchestrator-{uuid.uuid4().hex[:8]}"
        session = engine.create_session(user_id=user_id)
        session_id = session.get("id") if isinstance(session, dict) else None
        events = list(
            engine.stream_query(message=question, user_id=user_id, session_id=session_id)
        )
        chunks: list[str] = []
        for ev in events:
            if isinstance(ev, dict) and ev.get("code"):
                return f"Specialist error {ev.get('code')}: {ev.get('message')}"
            if isinstance(ev, dict):
                t = _extract_text(ev)
                if t:
                    chunks.append(t)
        joined = "\n".join(chunks).strip()
        if joined:
            return joined
        return _summarize_events_for_empty_response(events)
    except Exception as e:
        return f"Specialist query failed: {e}"


root_agent = LlmAgent(
    name="pa_orchestrator",
    model="gemini-2.5-flash",
    instruction=_ORCHESTRATOR_INSTRUCTION,
    tools=[FunctionTool(query_cost_specialist)],
)
