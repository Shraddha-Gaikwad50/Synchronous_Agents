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

from billing_llm_sql import (
    google_ai_configured,
    llm_sql_usable,
    question_needs_llm_sql,
    run_llm_billing_query,
    vertex_available,
)

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

    m_days = re.search(
        r"\b(?:(?:over|in|during|for)\s+the\s+)?(?:last|past)\s+(\d{1,2})\s+days?\b",
        q_lower,
    )
    if m_days:
        n = min(max(int(m_days.group(1)), 1), 366)
        end = today
        start = today - timedelta(days=n - 1)
        notes.append(f"last {n} days ({start} to {end})")
        return start, end, notes

    if re.search(r"\byesterday\b", q_lower):
        y = today - timedelta(days=1)
        notes.append(f"yesterday ({y})")
        return y, y, notes

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

    if re.search(r"\bthis\s+week\b", q_lower):
        start_this_week = today - timedelta(days=today.weekday())
        notes.append("this week (week-to-date)")
        return start_this_week, today, notes

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
    billing_region: str | None
    period_start: date | None
    period_end: date | None
    wants_total: bool
    wants_top: bool
    hint: str

    @property
    def has_period(self) -> bool:
        return self.period_start is not None and self.period_end is not None


def _mentions_prod(q: str) -> bool:
    # Avoid matching "prod" inside project slugs like "my-prod-app" (hyphen before token).
    return bool(re.search(r"(?<![-])\b(prod|production|prd)\b", q, re.I))


def _mentions_dev(q: str) -> bool:
    # Avoid matching "dev" inside "jaybel-dev" (hyphen before token).
    return bool(re.search(r"(?<![-])\b(dev|development)\b", q, re.I))


def _dev_mention_is_project_slug(q: str) -> bool:
    """e.g. 'jaybel- dev project' — the word dev is part of a project id, not environment=dev."""
    return bool(re.search(r"[a-z0-9][a-z0-9-]*\s*-\s*dev\s+project", q, re.I))


def _normalize_project_id_slug(raw: str) -> str:
    return re.sub(r"\s+", "", raw.strip().lower())


def _looks_like_gcp_region(token: str) -> bool:
    sl = token.lower().strip()
    if not sl or len(sl) > 32:
        return False
    return bool(re.match(r"^[a-z]{2,}-[a-z0-9-]+\d$", sl)) or bool(
        re.match(r"^[a-z]{2,}-[a-z0-9]+-[a-z0-9]+\d$", sl)
    )


def _extract_billing_region(question: str) -> str | None:
    """location.region value, e.g. asia-south1 from 'asia-south1' region."""
    q = (
        question.strip()
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    patterns = (
        r"(?i)['\"]([a-z0-9-]+)['\"]\s+region\b",
        r"(?i)\bin\s+the\s+['\"]([a-z0-9-]+)['\"]\s+region\b",
        r"(?i)\bregion\s+['\"]([a-z0-9-]+)['\"]",
    )
    for p in patterns:
        m = re.search(p, q)
        if m and _looks_like_gcp_region(m.group(1)):
            return m.group(1).lower()
    return None


def _extract_gcp_project_id(question: str) -> str | None:
    """Billing export project.id (e.g. jaybel-dev), not necessarily GCP project number."""
    ql = question.strip()
    # Hyphenated ids with optional spaces around hyphens: "jaybel- dev project", "gls-training-486405 project"
    m_slug = re.search(
        r"(?i)\b([a-z][a-z0-9]*(?:\s*-\s*[a-z0-9]+)+)\s+project\b",
        ql,
    )
    if m_slug:
        return _normalize_project_id_slug(m_slug.group(1))
    patterns = (
        r"(?i)in\s+the\s+([a-z][a-z0-9-]{1,62})\s+project\b",
        r"(?i)\bproject\s+([a-z][a-z0-9-]{1,62})\b",
        r"(?i)\b([a-z][a-z0-9-]{1,62})\s+project\b",
    )
    for p in patterns:
        m = re.search(p, ql)
        if m:
            return _normalize_project_id_slug(m.group(1))
    return None


def _sku_rank_template_eligible(question: str, f: CostQueryFilters) -> bool:
    return bool(
        re.search(r"\bskus?\b", question, re.I)
        and _extract_gcp_project_id(question)
        and f.has_period
    )


def _try_sku_top_by_project_json(
    question: str,
    f: CostQueryFilters,
    table_ref: str,
    table_project: str,
) -> str | None:
    """Fast template: top N SKUs by cost for one project.id over parsed date range (no env label filter)."""
    if not re.search(r"\bskus?\b", question, re.I):
        return None
    pid = _extract_gcp_project_id(question)
    if not pid or not f.has_period or f.period_start is None or f.period_end is None:
        return None
    m = re.search(r"\btop\s+(\d+)\b", question, re.I)
    top_n = int(m.group(1)) if m else 5
    top_n = min(max(top_n, 1), 40)

    sql = f"""
    SELECT
      sku.description AS sku_description,
      ANY_VALUE(service.description) AS service_name,
      ANY_VALUE(project.id) AS project_id,
      SUM(cost) AS cost_inr
    FROM `{table_ref}`
    WHERE DATE(usage_start_time) BETWEEN @period_start AND @period_end
      AND project.id = @project_id
    GROUP BY sku.description
    ORDER BY cost_inr DESC
    LIMIT @top_limit
    """
    params = [
        bigquery.ScalarQueryParameter("period_start", "DATE", f.period_start.isoformat()),
        bigquery.ScalarQueryParameter("period_end", "DATE", f.period_end.isoformat()),
        bigquery.ScalarQueryParameter("project_id", "STRING", pid),
        bigquery.ScalarQueryParameter("top_limit", "INT64", top_n),
    ]
    client = bigquery.Client(project=table_project)
    rows = list(
        client.query(
            sql,
            job_config=bigquery.QueryJobConfig(query_parameters=params),
        ).result()
    )
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "sku_description": str(row["sku_description"] or ""),
                "service_name": str(row["service_name"] or ""),
                "project_id": str(row["project_id"] or ""),
                "cost_inr": str(row["cost_inr"]),
                "currency": "INR",
            }
        )
    return json.dumps(out, indent=2)


def _spend_by_project_template_eligible(question: str, f: CostQueryFilters) -> bool:
    if not f.has_period:
        return False
    ql = question.lower()
    if not (
        re.search(r"\b(breakdown|broken down)\b", ql)
        or re.search(r"\bby\s+gcp\s+project\b", ql)
        or re.search(r"\b(per|each)\s+project\b", ql)
        or re.search(r"\bby\s+project\b", ql)
    ):
        return False
    return bool(
        re.search(r"\b(spend|cost|billing|charged|amount|total|inr|money)\b", ql)
    )


def _try_spend_by_project_json(
    question: str,
    f: CostQueryFilters,
    table_ref: str,
    table_project: str,
) -> str | None:
    """Aggregate cost by billing project.id over the parsed window (no env label filter)."""
    if not _spend_by_project_template_eligible(question, f):
        return None
    if not f.has_period or f.period_start is None or f.period_end is None:
        return None
    sql = f"""
    SELECT
      project.id AS project_id,
      SUM(cost) AS cost_inr
    FROM `{table_ref}`
    WHERE DATE(usage_start_time) BETWEEN @period_start AND @period_end
    GROUP BY project.id
    ORDER BY cost_inr DESC
    LIMIT 100
    """
    params = [
        bigquery.ScalarQueryParameter("period_start", "DATE", f.period_start.isoformat()),
        bigquery.ScalarQueryParameter("period_end", "DATE", f.period_end.isoformat()),
    ]
    client = bigquery.Client(project=table_project)
    rows = list(
        client.query(
            sql,
            job_config=bigquery.QueryJobConfig(query_parameters=params),
        ).result()
    )
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "project_id": str(row["project_id"] or ""),
                "cost_inr": str(row["cost_inr"]),
                "currency": "INR",
            }
        )
    return json.dumps(out, indent=2)


def _regions_cost_template_eligible(question: str, f: CostQueryFilters) -> bool:
    """Cost grouped by location.region (no LLM)."""
    if not f.has_period:
        return False
    ql = question.lower()
    if not re.search(
        r"\b(regions|availability\s+zones?|data\s+centers?|geographic)\b",
        ql,
    ):
        return False
    return bool(
        re.search(
            r"\b(cost|spend|money|expensive|most|highest|paid|bill|price)\b",
            ql,
        )
    )


def _try_regions_cost_json(
    question: str,
    f: CostQueryFilters,
    table_ref: str,
    table_project: str,
) -> str | None:
    if not _regions_cost_template_eligible(question, f):
        return None
    if not f.has_period or f.period_start is None or f.period_end is None:
        return None
    sql = f"""
    SELECT
      COALESCE(
        NULLIF(TRIM(COALESCE(location.region, '')), ''),
        '(unset / global)'
      ) AS region,
      SUM(cost) AS cost_inr
    FROM `{table_ref}`
    WHERE DATE(usage_start_time) BETWEEN @period_start AND @period_end
    GROUP BY region
    ORDER BY cost_inr DESC
    LIMIT 30
    """
    params = [
        bigquery.ScalarQueryParameter("period_start", "DATE", f.period_start.isoformat()),
        bigquery.ScalarQueryParameter("period_end", "DATE", f.period_end.isoformat()),
    ]
    client = bigquery.Client(project=table_project)
    rows = list(
        client.query(
            sql,
            job_config=bigquery.QueryJobConfig(query_parameters=params),
        ).result()
    )
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "region": str(row["region"] or ""),
                "cost_inr": str(row["cost_inr"]),
                "currency": "INR",
            }
        )
    return json.dumps(out, indent=2)


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
    elif _mentions_dev(q) and not _dev_mention_is_project_slug(question):
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

    br = _extract_billing_region(question)
    if br:
        notes.append(f"filtering location.region={br}")

    breakdown = bool(
        re.search(
            r"\b(breakdown|broken down|by\s+gcp\s+project|by\s+project|per\s+project|each\s+project)\b",
            q,
        )
    )
    wants_total = (
        bool(re.search(r"\b(total|sum|aggregate)\b", q))
        or (
            bool(re.search(r"\bhow\s+much\b", q))
            and bool(re.search(r"\b(spend|cost|pay|paid)\b", q))
        )
    ) and not breakdown
    wants_top = bool(re.search(r"\b(top|highest|largest|biggest|most\s+expensive)\b", q))

    hint = "; ".join(notes) if notes else "no explicit filters"
    return CostQueryFilters(
        env=env,
        svc=svc,
        billing_region=br,
        period_start=ps,
        period_end=pe,
        wants_total=wants_total,
        wants_top=wants_top,
        hint=hint,
    )


def _skip_billing_llm_sql(question: str, f: CostQueryFilters) -> bool:
    """Answer with parameterized BigQuery (query_bigquery) — no generated SQL / Vertex predict."""
    if not f.has_period or f.period_start is None or f.period_end is None:
        return False
    ql = question.lower()
    if any(
        re.search(p, ql)
        for p in (
            r"\bcredits?\b",
            r"\bpromotional\b",
            r"\bpromo\b",
            r"\bcommitment\b",
            r"\breservation\b",
            r"\bsustained use\b",
        )
    ):
        return False
    if re.search(r"\bskus?\b", ql) and not _extract_gcp_project_id(question):
        return False
    return bool(f.svc or f.billing_region or f.env)


def compute_llm_date_window(f: CostQueryFilters, today: date) -> tuple[date, date, str]:
    """Enforce max lookback for LLM-generated SQL (inclusive start/end)."""
    max_days = int(os.environ.get("BILLING_LLM_MAX_LOOKBACK_DAYS", "30"))
    allow_long = os.environ.get("BILLING_LLM_ALLOW_LONG_RANGE", "").lower() in ("1", "true", "yes")
    notes: list[str] = []
    if f.has_period and f.period_start is not None and f.period_end is not None:
        start, end = f.period_start, f.period_end
        span = (end - start).days + 1
        if span > max_days:
            if allow_long:
                notes.append(
                    f"Using your full requested window ({span} days) because BILLING_LLM_ALLOW_LONG_RANGE is enabled; "
                    "dry-run still enforces the byte cap."
                )
            else:
                start = end - timedelta(days=max_days - 1)
                notes.append(
                    f"Requested window exceeded {max_days} days; clamped to {start} through {end}. "
                    "Narrow the range or set BILLING_LLM_ALLOW_LONG_RANGE=1."
                )
        return start, end, " ".join(notes) if notes else ""
    end = today
    start = today - timedelta(days=max_days - 1)
    return (
        start,
        end,
        f"No explicit period in your question — using the last {max_days} days through {end} (cost control).",
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
    sku_early = _try_sku_top_by_project_json(question, f, table_ref, table_project)
    if sku_early is not None:
        return sku_early
    spend_early = _try_spend_by_project_json(question, f, table_ref, table_project)
    if spend_early is not None:
        return spend_early
    regions_early = _try_regions_cost_json(question, f, table_ref, table_project)
    if regions_early is not None:
        return regions_early

    filters: list[str] = []
    params: list[bigquery.ScalarQueryParameter] = []
    if f.svc:
        filters.append(
            "STRPOS(LOWER(IFNULL(service.description, '')), LOWER(@service_needle)) > 0"
        )
        params.append(bigquery.ScalarQueryParameter("service_needle", "STRING", f.svc))
    if f.billing_region:
        filters.append(
            "LOWER(TRIM(COALESCE(location.region, ''))) = LOWER(@billing_region)"
        )
        params.append(
            bigquery.ScalarQueryParameter("billing_region", "STRING", f.billing_region)
        )
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
        sql = f"SELECT COALESCE(SUM(cost), 0) AS total_inr FROM `{table_ref}` {where_sql}"
    elif f.wants_top:
        sql = f"""
        SELECT
          service.description AS service_name,
          {label_sql},
          SUM(cost) AS cost_inr
        FROM `{table_ref}`
        {where_sql}
        GROUP BY 1, 2
        ORDER BY cost_inr DESC
        LIMIT 40
        """
    else:
        sql = f"""
        SELECT
          DATE(usage_start_time) AS usage_date,
          service.description AS service_name,
          {label_sql},
          SUM(cost) AS cost_inr
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
        total = rows[0]["total_inr"] if rows else Decimal("0")
        return json.dumps([{"total_inr": str(total), "currency": "INR"}], indent=2)

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
                    "cost_inr": str(row["cost_inr"]),
                    "currency": "INR",
                }
            )
        else:
            usage_date = row["usage_date"]
            out.append(
                {
                    "date": usage_date.isoformat() if isinstance(usage_date, date) else str(usage_date),
                    "service_name": str(row["service_name"]),
                    "environment": row_env,
                    "cost_inr": str(row["cost_inr"]),
                    "currency": "INR",
                }
            )
    return json.dumps(out[:100], indent=2)


def query_cost_data(question: str) -> tuple[str, str]:
    """Run query against configured source, with fallback to Postgres in auto mode."""
    mode = SOURCE_MODE if SOURCE_MODE in {"auto", "bigquery", "postgres"} else "auto"
    f = parse_cost_query(question)
    hint = f.hint
    table_project = BQ_BILLING_PROJECT or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    llm_on = os.environ.get("BILLING_AGENT_LLM_SQL", "1").lower() not in ("0", "false", "no")

    if mode in {"auto", "bigquery"} and _bigquery_ready() and table_project:
        table_ref = f"{table_project}.{BQ_BILLING_DATASET}.{BQ_BILLING_TABLE}"
        sku_fast = _try_sku_top_by_project_json(question, f, table_ref, table_project)
        if sku_fast is not None:
            return (
                sku_fast,
                f"{hint}; source=bigquery; template=sku_by_project; currency=INR",
            )
        spend_fast = _try_spend_by_project_json(question, f, table_ref, table_project)
        if spend_fast is not None:
            return (
                spend_fast,
                f"{hint}; source=bigquery; template=spend_by_project; currency=INR",
            )
        regions_fast = _try_regions_cost_json(question, f, table_ref, table_project)
        if regions_fast is not None:
            return (
                regions_fast,
                f"{hint}; source=bigquery; template=cost_by_region; currency=INR",
            )

    if (
        mode in {"auto", "bigquery"}
        and _bigquery_ready()
        and table_project
        and llm_on
        and question_needs_llm_sql(question)
        and not _skip_billing_llm_sql(question, f)
    ):
        table_ref = f"{table_project}.{BQ_BILLING_DATASET}.{BQ_BILLING_TABLE}"
        ws, we, wnote = compute_llm_date_window(f, date.today())
        try:
            body, sh = run_llm_billing_query(
                question,
                table_ref,
                table_project,
                ws,
                we,
                wnote,
            )
            return body, f"{hint}; {sh}; currency=INR"
        except Exception as e:
            hint = f"{hint}; llm-sql failed ({e}); fallback=template"

    if mode in {"auto", "bigquery"} and _bigquery_ready():
        try:
            return query_bigquery(question), f"{hint}; source=bigquery; currency=INR"
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
        "description": "GCP billing export analytics (INR): fast templates plus guarded Vertex SQL for SKU, project, credits, regions.",
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
    f = parse_cost_query(message)
    progress_hint = f.hint
    llm_on = os.environ.get("BILLING_AGENT_LLM_SQL", "1").lower() not in ("0", "false", "no")
    use_llm = (
        llm_on
        and question_needs_llm_sql(message)
        and not _skip_billing_llm_sql(message, f)
        and not _sku_rank_template_eligible(message, f)
        and not _spend_by_project_template_eligible(message, f)
        and not _regions_cost_template_eligible(message, f)
    )
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

    if _sku_rank_template_eligible(message, f):
        run_msg = "Running fast template (top SKUs by project and date range)…"
    elif _spend_by_project_template_eligible(message, f):
        run_msg = "Running fast template (spend by GCP project for date range)…"
    elif _regions_cost_template_eligible(message, f):
        run_msg = "Running fast template (cost by GCP region for date range)…"
    elif use_llm:
        run_msg = "Running guarded analytics (Vertex / Google AI → BigQuery, dry-run byte cap)…"
    else:
        run_msg = f"Running query ({progress_hint})…"
    yield sse_pack(
        {
            "id": task_id,
            "status": {
                "state": "working",
                "message": {
                    "role": "agent",
                    "parts": [{"text": run_msg}],
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
    summary = f"Source: {source_hint}\n\nResult (amounts in INR ₹ where applicable):\n{result_text}"
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
        "billing_llm_sql_enabled": (
            os.environ.get("BILLING_AGENT_LLM_SQL", "1").lower() not in ("0", "false", "no")
        ),
        "vertex_generative_available": vertex_available(),
        "google_ai_api_configured": google_ai_configured(),
        "llm_sql_usable": llm_sql_usable(),
    }


setup_observability(app, "cost-agent")
