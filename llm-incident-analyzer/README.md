# LLM Incident Analyzer

An AI-powered incident analysis system that uses an LLM with **function calling** to investigate system incidents — fetching logs, metrics, and service status — then returns a structured root cause analysis via a FastAPI REST API.

---

## What It Does

Given an incident description, the system:
1. Sends the incident to an LLM (llama3.2:3b via Ollama)
2. LLM calls tools to gather context (logs, metrics, service status)
3. LLM returns a structured JSON analysis

```json
POST /analyze

{
  "root_cause": "Database connection pool exhausted due to long-running queries",
  "severity": "high",
  "affected_services": ["api-server", "database"],
  "recommended_fix": "Restart the database service and increase connection pool size in config",
  "estimated_resolution_time": "15 minutes"
}
```

---

## Architecture

```
User sends incident description
        ↓
FastAPI  →  analyze_incident()
        ↓
LLM (llama3.2:3b via Ollama)
  ↓ tool_call: get_service_logs(service_name)
  ↓ tool_call: get_system_metrics(service_name)
  ↓ tool_call: get_service_status(service_name)
        ↓
LLM produces structured JSON analysis
        ↓
MLflow logs incident + severity score
        ↓
FastAPI returns IncidentAnalysis response
```

---

## Key Concepts

**Function Calling** — the LLM does not run tools itself. It decides *which* tool to call and with *what arguments*. The application runs the tool and passes the result back. This loop repeats until the LLM has enough context to answer.

**Agentic Loop** — the LLM iterates (up to 5 times): call tool → observe result → call another tool → form final answer. This is the foundation of AI agents.

**Ollama** — runs LLMs locally. Exposes an OpenAI-compatible API at `http://localhost:11434/v1`, so standard OpenAI SDK works with no changes except `base_url`.

---

## Model Results

| Metric | Value |
|---|---|
| Model | llama3.2:3b (local via Ollama) |
| Avg response time | ~15–30s (CPU inference) |
| Tool calls per request | 1–3 |
| Severity classification | low / medium / high / critical |

> In production, swap Ollama for OpenAI GPT-4 by changing `OLLAMA_BASE_URL` and `api_key`. Code is identical.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | llama3.2:3b (Ollama) |
| LLM Client | OpenAI Python SDK (Ollama-compatible) |
| Function Calling | OpenAI tool_calls API |
| API | FastAPI + Pydantic |
| Server | Uvicorn |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Testing | Pytest |

---

## Project Structure

```
llm-incident-analyzer/
├── src/
│   ├── tools.py        # tool definitions + simulated tool functions
│   └── analyzer.py     # LLM agentic loop + function calling logic
├── api/
│   ├── main.py         # FastAPI application
│   └── schemas.py      # Pydantic input/output schemas
├── tests/
│   └── test_api.py
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Setup and Run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

> Requires [Ollama](https://ollama.com) running locally with `llama3.2:3b` pulled:
> ```bash
> ollama pull llama3.2:3b
> ollama serve
> ```

```bash
# Start the API
make serve
# Open http://localhost:8000/docs

# View MLflow experiments
make mlflow
# Open http://localhost:5000
```

**Run with Docker:**
```bash
make docker-build
make docker-run
```

**Run tests:**
```bash
make test
```

---

## Sample API Call

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"description": "API server returning 500 errors. Database connections timing out after 30s. CPU at 95%."}'
```
