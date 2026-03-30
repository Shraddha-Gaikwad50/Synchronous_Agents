# Frontend (Next.js 14)

This UI streams assistant responses from the orchestrator SSE endpoint.

## Environment

Create `frontend/.env.local`:

```bash
NEXT_PUBLIC_ORCHESTRATOR_URL=http://127.0.0.1:8000
```

The UI always talks to this HTTP server. It does **not** call Vertex directly.

### Invoking Vertex AI Agent Engine on every message

To have each chat go through your deployed **`pa_orchestrator_agent`** (so GCP metrics and sessions update), configure the **Python orchestrator** (`agents/orchestrator`), not the frontend URL:

1. Authenticate for local ADC: `gcloud auth application-default login`
2. Set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` (e.g. `us-central1`)
3. Set **`ORCHESTRATOR_AGENT_ENGINE_RESOURCE`** to the full resource from the console (**Copy identity**), e.g.  
   `projects/your-project/locations/us-central1/reasoningEngines/8296018091465244672`  
   Alternatively set **`ORCHESTRATOR_AGENT_ENGINE_QUERY_URL`** to the **Copy query URL** value.
4. Reinstall orchestrator deps (`google-cloud-aiplatform` is in `agents/orchestrator/requirements.txt`) and restart uvicorn on port 8000.

When this is active, `GET http://127.0.0.1:8000/health` includes `"agent_engine_chat_enabled": true` and each message uses `stream_query` on that engine.

See `vertex_agents/AGENT-ENGINE-DEPLOY.txt` and `config/gcp.env.example`.

## Local Run

From `frontend/`:

```bash
npm ci
npm run dev
```

Open `http://127.0.0.1:3000`.

## Full Local Stack

From repo root (Windows PowerShell):

```powershell
.\scripts\start-all.ps1
```

This starts:
- Postgres on `5433` (host)
- Cost agent on `8001`
- Orchestrator on `8000`
- Frontend on `3000`

Stop services:

```powershell
.\scripts\stop-all.ps1
docker compose down
```
