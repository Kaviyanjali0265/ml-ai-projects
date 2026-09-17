import sys
import os

from fastapi import FastAPI, HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api.schemas import IncidentInput, IncidentAnalysis
from src.analyzer import analyze_incident

app = FastAPI(
    title="LLM Incident Analyzer",
    description="Analyzes system incidents using an LLM with function calling to fetch logs and metrics.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/analyze", response_model=IncidentAnalysis)
def analyze(input: IncidentInput):
    result = analyze_incident(input.description)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return IncidentAnalysis(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
