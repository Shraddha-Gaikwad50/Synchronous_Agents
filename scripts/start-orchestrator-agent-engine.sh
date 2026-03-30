#!/usr/bin/env bash
# Start only the FastAPI orchestrator with config/gcp.env (Vertex Agent Engine chat).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/logs"

ENV_FILE="$ROOT/config/gcp.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE — create it from config/gcp.env.example" >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

PYTHON_BIN="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Creating .venv and installing orchestrator deps..." >&2
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -q -r "$ROOT/agents/orchestrator/requirements.txt"
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi

if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
  echo "Application Default Credentials missing. Run:" >&2
  echo "  gcloud auth application-default login" >&2
  exit 1
fi

echo ">>> Orchestrator :8000 (agent_engine_chat_enabled when resource + project are set)"
echo "    ORCHESTRATOR_AGENT_ENGINE_RESOURCE=${ORCHESTRATOR_AGENT_ENGINE_RESOURCE:-}"
cd "$ROOT/agents/orchestrator"
exec "$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port 8000
