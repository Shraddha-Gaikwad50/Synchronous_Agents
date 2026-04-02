#!/usr/bin/env bash
# Postgres in Docker; agents + Next on host. Or use: docker compose up --build
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/logs"

# Vertex Agent Engine chat: set ORCHESTRATOR_AGENT_ENGINE_RESOURCE in config/gcp.env (gitignored).
if [[ -f "$ROOT/config/gcp.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/config/gcp.env"
  set +a
  echo ">>> Loaded config/gcp.env (GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-})"
fi

PYTHON_BIN="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo ">>> Creating .venv..."
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -q -r "$ROOT/agents/cost_agent/requirements.txt" -r "$ROOT/agents/orchestrator/requirements.txt"
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi

echo ">>> Starting postgres..."
if docker compose version >/dev/null 2>&1; then
  DC=(docker compose)
else
  DC=(docker-compose)
fi
"${DC[@]}" up -d postgres

for i in $(seq 1 90); do
  if "${DC[@]}" exec -T postgres pg_isready -U postgres >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! "${DC[@]}" exec -T postgres psql -U postgres -d postgres -tAc "SELECT to_regclass('public.cloud_costs');" | grep -q cloud_costs; then
  echo ">>> Applying database/schema.sql..."
  "${DC[@]}" exec -T postgres psql -U postgres -d postgres < "$ROOT/database/schema.sql"
fi

export DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@127.0.0.1:5433/postgres}"
export COST_AGENT_CARD_URL="${COST_AGENT_CARD_URL:-http://127.0.0.1:8001/.well-known/agent.json}"
export COST_AGENT_TASKS_URL="${COST_AGENT_TASKS_URL:-http://127.0.0.1:8001/tasks/send}"

# BigQuery Cloud Billing export (cost agent). Override in config/gcp.env.
export BQ_BILLING_PROJECT="${BQ_BILLING_PROJECT:-${GOOGLE_CLOUD_PROJECT:-}}"
export BQ_BILLING_DATASET="${BQ_BILLING_DATASET:-gcp_billing_data}"
export BQ_BILLING_TABLE="${BQ_BILLING_TABLE:-gcp_billing_export_resource_v1_01B40E_943432_338729}"
if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" && -f "$ROOT/frontend/.secrets/speech-sa.json" ]]; then
  export GOOGLE_APPLICATION_CREDENTIALS="$ROOT/frontend/.secrets/speech-sa.json"
fi

echo ">>> Starting cost-agent :8001..."
(cd "$ROOT/agents/cost_agent" && nohup "$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port 8001 \
  >"$ROOT/logs/cost-agent.log" 2>"$ROOT/logs/cost-agent.err.log" &)
sleep 2

echo ">>> Starting orchestrator :8000..."
(cd "$ROOT/agents/orchestrator" && nohup "$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port 8000 \
  >"$ROOT/logs/orchestrator.log" 2>"$ROOT/logs/orchestrator.err.log" &)
sleep 2

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo ">>> npm ci..."
  (cd "$ROOT/frontend" && npm ci)
fi

echo ">>> Starting frontend :3000..."
(cd "$ROOT/frontend" && nohup npm run dev -- --hostname 127.0.0.1 --port 3000 \
  >"$ROOT/logs/frontend.log" 2>"$ROOT/logs/frontend.err.log" &)

echo ""
echo "Done. Open http://127.0.0.1:3000"
echo "Logs: $ROOT/logs/"
echo "Stop: pkill -f 'uvicorn main:app' ; pkill -f 'next dev'  (careful on shared machines)"
