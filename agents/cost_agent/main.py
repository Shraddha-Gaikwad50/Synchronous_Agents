"""
Cost Metrics Specialist — HTTP surface compatible with A2A-style discovery and /tasks/send SSE.

Data source modes:
- BigQuery billing export (preferred when configured)
- PostgreSQL cloud_costs table (fallback)
"""
from __future__ import annotations

import asyncio
import calendar
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import AsyncIterator

from google.cloud import bigquery
import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from telemetry import setup_observability

# Optional: ADK agent shell for future tool wiring (no HTTP coupling)
try:
    from google.adk.agents import Agent
    from google.adk.runners import InMemoryRunner

    _ADK_AVAILABLE = True
except Exception:  # pragma: no cover
    _ADK_AVAILABLE = False

BASE_URL = os.environ.get("COST_AGENT_PUBLIC_URL", "http://localhost:8001")

# ---------------------------------------------------------------------------
# DATABASE_URL (required in cloud; optional locally)
#
# Local dev: defaults to postgres on localhost (see below).
#
# Production (Phase 2 — hybrid cloud): compute runs on GCP (Cloud Run / Agent
# Engine), but PostgreSQL stays on-premises. The app does NOT open inbound DB
# ports on GCP; instead, expose the on-prem Postgres (or a TCP proxy in front
# of it) through a secure tunnel such as Cloudflare Tunnel, Tailscale Funnel,
# or ngrok TCP. Set DATABASE_URL to the tunnel's public DSN, for example:
#   postgresql://user:pass@db-tunnel.example.com:5432/postgres
#
# Store this value in Secret Manager and mount it into Cloud Run with
# --set-secrets (see deploy.sh). Rotate tunnel credentials independently of
# the service image.
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)
SOURCE_MODE = os.environ.get("COST_DATA_SOURCE", "auto").strip().lower()
BQ_BILLING_PROJECT = os.environ.get("BQ_BILLING_PROJECT", "").strip()
BQ_BILLING_DATASET = os.environ.get("BQ_BILLING_DATASET", "").strip()
BQ_BILLING_TABLE = os.environ.get("BQ_BILLING_TABLE", "").strip()


def get_connection():
    return psycopg.connect(DATABASE_URL)


_MONTH_NAMES: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    last = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last)


def _parse_time_period(question: str, q_lower: str, today: date) -> tuple[date | None, date | None, list[str]]:
    """Inclusive date range from natural language; also handles YYYY-MM-DD."""
    notes: list[str] = []

    m = re.search(
        r"(?:for\s+)?(?:the\s+)?(?:entire\s+month\s+of\s+)?"
        r"(january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{4})\b",
        q_lower,
    )
    if m:
        month = _MONTH_NAMES[m.group(1)]
        year = int(m.group(2))
        start, end = _month_bounds(year, month)
        notes.append(f"date range {start} to {end}")
        return start, end, notes

    if re.search(r"\bthis\s+month\b", q_lower):
        start = date(today.year, today.month, 1)
        notes.append("this month (month-to-date)")
        return start, today, notes

    if re.search(r"\blast\s+month\b", q_lower):
        first_this = date(today.year, today.month, 1)
        last_prev = first_this - timedelta(days=1)
        start, end = _month_bounds(last_prev.year, last_prev.month)
        notes.append(f"last month ({start} to {end})")
        return start, end, notes

    if re.search(r"\blast\s+week\b", q_lower):
        start_this_week = today - timedelta(days=today.weekday())
        end_last = start_this_week - timedelta(days=1)
        start_last = end_last - timedelta(days=6)
        notes.append(f"last calendar week ({start_last} to {end_last})")
        return start_last, end_last, notes

    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
    if date_match:
        d = date.fromisoformat(date_match.group(1))
        notes.append(f"filtering date={d}")
        return d, d, notes

    return None, None, notes


@dataclass(frozen=True)
class CostQueryFilters:
    env: str | None
    svc: str | None
    period_start: date | None
    period_end: date | None
    wants_total: bool
    wants_top: bool
    hint: str

    @property
    def has_period(self) -> bool:
        return self.period_start is not None and self.period_end is not None


def _mentions_prod(q: str) -> bool:
    return bool(re.search(r"\b(prod|production|prd)\b", q, re.I))


def _mentions_dev(q: str) -> bool:
    return bool(re.search(r"\b(dev|development)\b", q, re.I))


def parse_cost_query(question: str, *, today: date | None = None) -> CostQueryFilters:
    q = question.strip().lower()
    notes: list[str] = []
    env: str | None = None
    svc: str | None = None
    ref = today or date.today()

    # "Compare prod vs dev …" must not collapse to prod only (substring "prod" wins before "dev").
    if _mentions_prod(q) and _mentions_dev(q):
        env = None
        notes.append("comparing prod and dev (both environments)")
        notes.append("unlabeled projects appear as prod in export")
    elif _mentions_prod(q):
        env = "prod"
        notes.append("filtering environment=prod")
    elif _mentions_dev(q):
        env = "dev"
        notes.append("filtering environment=dev")

    svc_match = re.search(
        r"(compute engine|cloud storage|bigquery|cloud sql|artifact registry|networking|vertex ai|kubernetes engine|cloud run|cloud logging|logging)",
        q,
        re.I,
    )
    if svc_match:
        svc = svc_match.group(1).lower()
        notes.append(f"filtering service contains '{svc}'")

    ps, pe, pnotes = _parse_time_period(question, q, ref)
    notes.extend(pnotes)

    wants_total = bool(re.search(r"\b(total|sum|aggregate)\b", q))
    wants_top = bool(re.search(r"\b(top|highest|largest|biggest|most\s+expensive)\b", q))

    hint = "; ".join(notes) if notes else "no explicit filters"
    return CostQueryFilters(
        env=env,
        svc=svc,
        period_start=ps,
        period_end=pe,
        wants_total=wants_total,
        wants_top=wants_top,
        hint=hint,
    )


def nl_to_sql(question: str) -> tuple[str, str]:
    f = parse_cost_query(question)
    where: list[str] = []
    if f.env:
        where.append("environment = %s")
    if f.svc:
        where.append("LOWER(service_name) LIKE LOWER(%s)")
    if f.has_period:
        where.append("date BETWEEN %s::date AND %s::date")
    wh = " AND ".join(where) if where else "TRUE"
    if f.wants_total:
        return (
            "SELECT COALESCE(SUM(cost_usd), 0) AS total_usd FROM cloud_costs WHERE " + wh,
            f.hint,
        )
    order = "cost_usd DESC, date DESC" if f.wants_top else "date DESC, id DESC"
    return (
        f"SELECT id, date, service_name, environment, cost_usd FROM cloud_costs WHERE {wh} ORDER BY {order} LIMIT 100",
        f.hint,
    )


def _params_for_sql(sql: str, question: str) -> tuple:
    params: list = []
    f = parse_cost_query(question)
    if f.env and "environment = %s" in sql:
        params.append(f.env)
    if f.svc and "LIKE" in sql and "service_name" in sql:
        params.append(f"%{f.svc}%")
    if f.has_period and "BETWEEN" in sql:
        params.append(f.period_start.isoformat())
        params.append(f.period_end.isoformat())
    return tuple(params)


def run_query(sql: str, params: tuple) -> str:
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            colnames = [d[0] for d in cur.description] if cur.description else []
    lines = []
    for row in rows:
        lines.append(dict(zip(colnames, [str(c) for c in row])))
    return json.dumps(lines, indent=2)


def _bigquery_ready() -> bool:
    return bool(BQ_BILLING_DATASET and BQ_BILLING_TABLE)


def _normalize_env(raw: str | None) -> str:
    if not raw:
        return "prod"
    val = raw.strip().lower()
    if val in {"prod", "production", "prd"}:
        return "prod"
    if val in {"dev", "development"}:
        return "dev"
    return "prod"


def _bq_env_sql_fragment(env: str | None) -> str:
    """Filter billing rows by project label environment (resource export schema)."""
    if not env:
        return ""
    if env == "prod":
        return """ AND (
          NOT EXISTS (
            SELECT 1 FROM UNNEST(IFNULL(project.labels, [])) AS l
            WHERE LOWER(l.key) IN ('environment', 'env')
          )
          OR EXISTS (
            SELECT 1 FROM UNNEST(IFNULL(project.labels, [])) AS l
            WHERE LOWER(l.key) IN ('environment', 'env')
              AND LOWER(l.value) IN ('prod', 'production', 'prd')
          )
        )"""
    if env == "dev":
        return """ AND EXISTS (
          SELECT 1 FROM UNNEST(IFNULL(project.labels, [])) AS l
          WHERE LOWER(l.key) IN ('environment', 'env')
            AND LOWER(l.value) IN ('dev', 'development')
        )"""
    return ""


def query_bigquery(question: str) -> str:
    f = parse_cost_query(question)
    table_project = BQ_BILLING_PROJECT or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not table_project:
        raise RuntimeError("Set BQ_BILLING_PROJECT or GOOGLE_CLOUD_PROJECT for BigQuery queries.")
    table_ref = f"{table_project}.{BQ_BILLING_DATASET}.{BQ_BILLING_TABLE}"
    filters: list[str] = []
    params: list[bigquery.ScalarQueryParameter] = []
    if f.svc:
        filters.append(
            "STRPOS(LOWER(IFNULL(service.description, '')), LOWER(@service_needle)) > 0"
        )
        params.append(bigquery.ScalarQueryParameter("service_needle", "STRING", f.svc))
    if f.has_period:
        filters.append("DATE(usage_start_time) BETWEEN @period_start AND @period_end")
        params.append(
            bigquery.ScalarQueryParameter("period_start", "DATE", f.period_start.isoformat())
        )
        params.append(bigquery.ScalarQueryParameter("period_end", "DATE", f.period_end.isoformat()))
    env_sql = _bq_env_sql_fragment(f.env)
    where_core = f"{' AND '.join(filters)}" if filters else "TRUE"
    where_sql = f"WHERE {where_core}{env_sql}"
    label_sql = """COALESCE(
            (
              SELECT ANY_VALUE(l.value)
              FROM UNNEST(IFNULL(project.labels, [])) AS l
              WHERE LOWER(l.key) IN ('environment', 'env')
            ),
            'prod'
          ) AS raw_environment"""

    if f.wants_total:
        sql = f"SELECT COALESCE(SUM(cost), 0) AS total_usd FROM `{table_ref}` {where_sql}"
    elif f.wants_top:
        sql = f"""
        SELECT
          service.description AS service_name,
          {label_sql},
          SUM(cost) AS cost_usd
        FROM `{table_ref}`
        {where_sql}
        GROUP BY 1, 2
        ORDER BY cost_usd DESC
        LIMIT 40
        """
    else:
        sql = f"""
        SELECT
          DATE(usage_start_time) AS usage_date,
          service.description AS service_name,
          {label_sql},
          SUM(cost) AS cost_usd
        FROM `{table_ref}`
        {where_sql}
        GROUP BY 1, 2, 3
        ORDER BY usage_date DESC, service_name
        LIMIT 100
        """
    client = bigquery.Client(project=table_project)
    rows = list(
        client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    )
    if f.wants_total:
        total = rows[0]["total_usd"] if rows else Decimal("0")
        return json.dumps([{"total_usd": str(total)}], indent=2)

    period_label = (
        f"{f.period_start} to {f.period_end}" if f.has_period else ""
    )
    out: list[dict[str, str]] = []
    for row in rows:
        row_env = _normalize_env(row["raw_environment"])
        if f.env and row_env != f.env:
            continue
        if f.wants_top:
            out.append(
                {
                    "date": period_label or "aggregated",
                    "service_name": str(row["service_name"]),
                    "environment": row_env,
                    "cost_usd": str(row["cost_usd"]),
                }
            )
        else:
            usage_date = row["usage_date"]
            out.append(
                {
                    "date": usage_date.isoformat() if isinstance(usage_date, date) else str(usage_date),
                    "service_name": str(row["service_name"]),
                    "environment": row_env,
                    "cost_usd": str(row["cost_usd"]),
                }
            )
    return json.dumps(out[:100], indent=2)


def query_cost_data(question: str) -> tuple[str, str]:
    """Run query against configured source, with fallback to Postgres in auto mode."""
    mode = SOURCE_MODE if SOURCE_MODE in {"auto", "bigquery", "postgres"} else "auto"
    hint = parse_cost_query(question).hint
    if mode in {"auto", "bigquery"} and _bigquery_ready():
        try:
            return query_bigquery(question), f"{hint}; source=bigquery"
        except Exception as e:
            if mode == "bigquery":
                raise
            hint = f"{hint}; bigquery unavailable ({e}); fallback=postgres"
    sql, _ = nl_to_sql(question)
    params = _params_for_sql(sql, question)
    return run_query(sql, params), f"{hint}; source=postgres"


def agent_card() -> dict:
    return {
        "name": "Cost Metrics Agent",
        "description": "Enterprise tasks: query infrastructure costs, analyze usage spikes, generate budget reports.",
        "url": BASE_URL,
        "version": "1.0.0",
        "capabilities": {"streaming": True, "pushNotifications": False},
        "skills": [
            {
                "id": "metrics.query_cost",
                "name": "Cost Query",
                "description": "Query costs by service, date, or environment.",
                "inputModes": ["text"],
                "outputModes": ["text"],
            }
        ],
    }


class TaskSendBody(BaseModel):
    message: str = Field(..., description="Natural language cost question")
    id: str | None = Field(default=None, description="Optional task id")


app = FastAPI(title="Cost Metrics Agent", version="1.0.0")

if _ADK_AVAILABLE:
    _cost_adk_agent = Agent(
        model="gemini-2.0-flash",
        name="cost_metrics_adk",
        description="ADK agent placeholder; HTTP layer performs NL→SQL for Phase 1.",
    )
    _ = InMemoryRunner(agent=_cost_adk_agent)


@app.get("/.well-known/agent.json")
async def well_known_agent():
    return JSONResponse(agent_card())


def sse_pack(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def task_stream(message: str, task_id: str) -> AsyncIterator[str]:
    progress_hint = parse_cost_query(message).hint
    yield sse_pack(
        {
            "id": task_id,
            "status": {
                "state": "working",
                "message": {
                    "role": "agent",
                    "parts": [{"text": "Parsing your cost question…"}],
                },
            },
        }
    )
    await asyncio.sleep(0.05)

    yield sse_pack(
        {
            "id": task_id,
            "status": {
                "state": "working",
                "message": {
                    "role": "agent",
                    "parts": [{"text": f"Running query ({progress_hint})…"}],
                },
            },
        }
    )
    await asyncio.sleep(0.05)

    try:
        result_text, source_hint = await asyncio.to_thread(query_cost_data, message)
    except Exception as e:
        yield sse_pack(
            {
                "id": task_id,
                "status": {
                    "state": "working",
                    "message": {
                        "role": "agent",
                        "parts": [{"text": f"Database error: {e}"}],
                    },
                },
            }
        )
        yield sse_pack({"id": task_id, "status": {"state": "completed"}, "artifact": {"parts": [{"text": ""}]}})
        return

    # Stream result in chunks (dummy chunking for SSE demo)
    chunk_size = 180
    summary = f"Source: {source_hint}\n\nResult:\n{result_text}"
    for i in range(0, len(summary), chunk_size):
        part = summary[i : i + chunk_size]
        yield sse_pack(
            {
                "id": task_id,
                "status": {
                    "state": "working",
                    "message": {
                        "role": "agent",
                        "parts": [{"text": part}],
                    },
                },
            }
        )
        await asyncio.sleep(0.02)

    yield sse_pack(
        {
            "id": task_id,
            "status": {"state": "completed"},
            "artifact": {
                "parts": [{"text": f"\n\n✓ Completed ({source_hint})."}],
            },
        }
    )


@app.post("/tasks/send")
async def tasks_send(body: TaskSendBody):
    task_id = body.id or f"task-{uuid.uuid4().hex[:12]}"
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    return StreamingResponse(
        task_stream(body.message, task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    tproj = BQ_BILLING_PROJECT or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    table = (
        f"{tproj}.{BQ_BILLING_DATASET}.{BQ_BILLING_TABLE}"
        if _bigquery_ready() and tproj
        else None
    )
    return {
        "status": "ok",
        "adk": _ADK_AVAILABLE,
        "source_mode": SOURCE_MODE,
        "bigquery_configured": _bigquery_ready(),
        "bigquery_table": table,
    }


setup_observability(app, "cost-agent")
