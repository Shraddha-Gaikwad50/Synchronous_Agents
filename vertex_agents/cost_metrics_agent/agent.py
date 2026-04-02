"""Cost metrics specialist for Vertex AI Agent Engine (Gemini + read-only SQL tools)."""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from . import db_logic


def query_cloud_costs(question: str) -> str:
    """Answer questions about cloud spend via BigQuery export or PostgreSQL.

    Pass the user's question in natural language; filters for environment, service,
    date, totals vs detail rows are inferred automatically.
    """
    return db_logic.query_costs(question)


root_agent = LlmAgent(
    name="cost_metrics_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a cloud cost analyst. Use query_cloud_costs for all cost questions. "
        "The tool may fetch from BigQuery Billing Export (preferred) or PostgreSQL "
        "cloud_costs fallback. Handle spend, services, environments (prod/dev), and "
        "dates. Summarize results clearly; if JSON rows are returned, explain totals "
        "or key highlights. If the tool returns JSON with an \"error\" field, quote "
        "the detail and hint for the user. If cost data is unavailable, say so and "
        "suggest checking billing export or DATABASE_URL."
    ),
    tools=[FunctionTool(query_cloud_costs)],
)
