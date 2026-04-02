"""Read-only cost query helpers with dual backend support.

Backends:
- BigQuery Billing Export table (preferred when configured)
- PostgreSQL cloud_costs table (fallback)
"""

from __future__ import annotations

import calendar
import json
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from google.cloud import bigquery
import psycopg

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5433/postgres",
)
SOURCE_MODE = os.environ.get("COST_DATA_SOURCE", "auto").strip().lower()

# BigQuery Billing Export source (optional)
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


def _mentions_prod(q: str) -> bool:
    return bool(re.search(r"(?<![-])\b(prod|production|prd)\b", q, re.I))


def _mentions_dev(q: str) -> bool:
    return bool(re.search(r"(?<![-])\b(dev|development)\b", q, re.I))


def _dev_mention_is_project_slug(q: str) -> bool:
    return bool(re.search(r"[a-z0-9][a-z0-9-]*\s*-\s*dev\s+project", q, re.I))


def _normalize_project_id_slug(raw: str) -> str:
    return re.sub(r"\s+", "", raw.strip().lower())


def _extract_gcp_project_id(question: str) -> str | None:
    ql = question.strip()
    m_slug = re.search(
        r"(?i)\b([a-z][a-z0-9]*(?:\s*-\s*[a-z0-9]+)+)\s+project\b",
        ql,
    )
    if m_slug:
        return _normalize_project_id_slug(m_slug.group(1))
    m_for = re.search(
        r"(?i)\b(?:for|in)\s+([a-z][a-z0-9]*(?:\s*-\s*[a-z0-9]+)+)\b",
        ql,
    )
    if m_for:
        return _normalize_project_id_slug(m_for.group(1))
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


def _parse_time_period(question: str, q_lower: str, today: date) -> tuple[date | None, date | None, list[str]]:
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

    _mo = (
        r"january|february|march|april|may|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
    )
    m_dom = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_mo})\s+(\d{{4}})\b", q_lower)
    if m_dom:
        dom = int(m_dom.group(1))
        month = _MONTH_NAMES[m_dom.group(2)]
        year = int(m_dom.group(3))
        if 1 <= dom <= 31:
            try:
                d_only = date(year, month, dom)
                notes.append(f"filtering date={d_only}")
                return d_only, d_only, notes
            except ValueError:
                pass
    m_mdy = re.search(
        rf"\b({_mo})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
        q_lower,
    )
    if m_mdy:
        month = _MONTH_NAMES[m_mdy.group(1)]
        dom = int(m_mdy.group(2))
        year = int(m_mdy.group(3))
        if 1 <= dom <= 31:
            try:
                d_only = date(year, month, dom)
                notes.append(f"filtering date={d_only}")
                return d_only, d_only, notes
            except ValueError:
                pass

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


def _looks_like_gcp_region(token: str) -> bool:
    sl = token.lower().strip()
    if not sl or len(sl) > 32:
        return False
    return bool(re.match(r"^[a-z]{2,}-[a-z0-9-]+\d$", sl)) or bool(
        re.match(r"^[a-z]{2,}-[a-z0-9]+-[a-z0-9]+\d$", sl)
    )


def _extract_billing_region(question: str) -> str | None:
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


@dataclass(frozen=True)
class CostQueryFilters:
    env: str | None
    svc: str | None
    billing_project_id: str | None
    billing_region: str | None
    period_start: date | None
    period_end: date | None
    wants_total: bool
    wants_top: bool
    hint: str

    @property
    def has_period(self) -> bool:
        return self.period_start is not None and self.period_end is not None


def parse_cost_query(question: str, *, today: date | None = None) -> CostQueryFilters:
    q = question.strip().lower()
    notes: list[str] = []
    env: str | None = None
    svc: str | None = None
    ref = today or date.today()

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

    bproj = _extract_gcp_project_id(question)
    if bproj:
        notes.append(f"filtering project.id={bproj}")

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

    hint = "; ".join(notes) if notes else "no explicit filters, using full available data"
    return CostQueryFilters(
        env=env,
        svc=svc,
        billing_project_id=bproj,
        billing_region=br,
        period_start=ps,
        period_end=pe,
        wants_total=wants_total,
        wants_top=wants_top,
        hint=hint,
    )


def nl_to_sql(question: str) -> tuple[str, str]:
    """Minimal, safe SELECT-only SQL for cloud_costs. Returns (sql, reasoning_snippet)."""
    f = parse_cost_query(question)
    where: list[str] = []
    if f.env:
        where.append("environment = %s")
    if f.svc:
        where.append("LOWER(service_name) LIKE LOWER(%s)")
    if f.has_period:
        where.append("date BETWEEN %s::date AND %s::date")

    if f.wants_total:
        wh = " AND ".join(where) if where else "TRUE"
        return (
            "SELECT COALESCE(SUM(cost_usd), 0) AS total_usd FROM cloud_costs WHERE " + wh,
            f.hint,
        )

    wh = " AND ".join(where) if where else "TRUE"
    order = "cost_usd DESC, date DESC" if f.wants_top else "date DESC, id DESC"
    return (
        "SELECT id, date, service_name, environment, cost_usd FROM cloud_costs WHERE "
        f"{wh} ORDER BY {order} LIMIT 100",
        f.hint,
    )


def params_for_sql(sql: str, question: str) -> tuple:
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


def _query_bigquery(question: str) -> str:
    f = parse_cost_query(question)
    table_project = BQ_BILLING_PROJECT or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not table_project:
        raise RuntimeError("Set BQ_BILLING_PROJECT or GOOGLE_CLOUD_PROJECT for BigQuery source.")
    table_ref = f"{table_project}.{BQ_BILLING_DATASET}.{BQ_BILLING_TABLE}"

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
    if f.billing_project_id:
        filters.append("project.id = @billing_project_id")
        params.append(
            bigquery.ScalarQueryParameter("billing_project_id", "STRING", f.billing_project_id)
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

    period_label = f"{f.period_start} to {f.period_end}" if f.has_period else ""
    normalized: list[dict[str, str]] = []
    for row in rows:
        row_env = _normalize_env(row["raw_environment"])
        if f.env and row_env != f.env:
            continue
        if f.wants_top:
            normalized.append(
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
            usage_date_val = usage_date.isoformat() if isinstance(usage_date, date) else str(usage_date)
            normalized.append(
                {
                    "date": usage_date_val,
                    "service_name": str(row["service_name"]),
                    "environment": row_env,
                    "cost_inr": str(row["cost_inr"]),
                    "currency": "INR",
                }
            )
    return json.dumps(normalized[:100], indent=2)


def _error_payload(kind: str, detail: str, hint: str | None = None) -> str:
    payload: dict[str, str] = {"error": kind, "detail": detail}
    if hint:
        payload["hint"] = hint
    return json.dumps(payload, indent=2)


def query_costs(question: str) -> str:
    """Query cloud costs from configured backend.

    Returns JSON rows on success, or a JSON object with ``error`` and ``detail``
    when the configured backend is unreachable or misconfigured (never raises).
    """
    mode = SOURCE_MODE if SOURCE_MODE in {"auto", "bigquery", "postgres"} else "auto"
    if mode in {"auto", "bigquery"} and _bigquery_ready():
        try:
            return _query_bigquery(question)
        except Exception as e:
            if mode == "bigquery":
                return _error_payload(
                    "bigquery_failed",
                    str(e),
                    "Check BQ_BILLING_* and IAM; table must exist and be readable.",
                )
            # auto: fall through to Postgres
    try:
        sql, _ = nl_to_sql(question)
        params = params_for_sql(sql, question)
        return run_query(sql, params)
    except Exception as e:
        return _error_payload(
            "postgres_query_failed",
            str(e),
            "Set DATABASE_URL to a reachable Postgres instance with cloud_costs. "
            "127.0.0.1 only works when a DB is bound locally (not in Agent Engine).",
        )
