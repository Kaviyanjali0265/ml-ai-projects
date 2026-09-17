import json
import os

import mlflow
from openai import OpenAI

from src.tools import TOOL_DEFINITIONS, call_tool

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) and incident responder.

When given an incident description, use the available tools to gather more information,
then provide a structured analysis.

Always respond with valid JSON in this exact format:
{
  "root_cause": "brief description of what caused the incident",
  "severity": "low | medium | high | critical",
  "affected_services": ["list", "of", "services"],
  "recommended_fix": "step-by-step fix instructions",
  "estimated_resolution_time": "e.g. 15 minutes"
}

Use tools to investigate before forming your conclusion. Be concise and actionable."""


def analyze_incident(incident_description: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Incident reported:\n\n{incident_description}"}
    ]

    with mlflow.start_run():
        mlflow.log_param("model", MODEL_NAME)
        mlflow.log_param("incident", incident_description[:200])

        tool_calls_made = []

        # Agentic loop — keep going until LLM stops calling tools
        for _ in range(5):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                temperature=0.1,
            )

            choice = response.choices[0]

            # LLM wants to call a tool
            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)

                for tool_call in choice.message.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    tool_calls_made.append({"tool": name, "args": args})

                    result = call_tool(name, args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

            # LLM is done — has final answer
            else:
                final_text = choice.message.content.strip()
                mlflow.log_param("tool_calls_count", len(tool_calls_made))

                # Extract JSON from response
                try:
                    # Handle case where model wraps JSON in markdown code block
                    if "```" in final_text:
                        final_text = final_text.split("```")[1]
                        if final_text.startswith("json"):
                            final_text = final_text[4:]
                    result = json.loads(final_text)
                except json.JSONDecodeError:
                    result = {
                        "root_cause": final_text,
                        "severity": "unknown",
                        "affected_services": [],
                        "recommended_fix": "Manual investigation required",
                        "estimated_resolution_time": "unknown"
                    }

                mlflow.log_metric("severity_score", _severity_to_score(result.get("severity", "unknown")))
                return result

    return {"error": "LLM did not produce a final answer after 5 iterations"}


def _severity_to_score(severity: str) -> int:
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(severity.lower(), 0)
