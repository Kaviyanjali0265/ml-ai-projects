import json
import random
from datetime import datetime, timedelta, timezone

# Tool definitions sent to the LLM so it knows what tools are available
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_service_logs",
            "description": "Fetch recent logs for a given service. Returns log lines with timestamps and log levels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Name of the service (e.g. api-server, auth-service, database)"
                    },
                    "last_minutes": {
                        "type": "integer",
                        "description": "How many minutes back to fetch logs. Default 10."
                    }
                },
                "required": ["service_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_metrics",
            "description": "Fetch current CPU, memory, and disk usage for a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Name of the service to get metrics for"
                    }
                },
                "required": ["service_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_status",
            "description": "Check whether a service is running, degraded, or down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Name of the service to check"
                    }
                },
                "required": ["service_name"]
            }
        }
    }
]


def get_service_logs(service_name: str, last_minutes: int = 10) -> str:
    now = datetime.now(timezone.utc)
    logs = []
    templates = [
        ("ERROR", f"Connection timeout to {service_name} after 30s"),
        ("ERROR", f"Failed to process request: NullPointerException in {service_name}"),
        ("WARN",  f"High memory usage detected in {service_name}: 87%"),
        ("ERROR", f"Database connection pool exhausted in {service_name}"),
        ("INFO",  f"{service_name} health check passed"),
        ("ERROR", f"Out of memory: kill process {service_name}"),
        ("WARN",  f"Response time degraded: {service_name} p99=4200ms"),
        ("INFO",  f"Request handled by {service_name} in 45ms"),
    ]
    for i in range(8):
        ts = (now - timedelta(minutes=random.randint(0, last_minutes))).strftime("%Y-%m-%dT%H:%M:%SZ")
        level, msg = templates[i % len(templates)]
        logs.append(f"[{ts}] {level} {msg}")

    return json.dumps({"service": service_name, "logs": logs})


def get_system_metrics(service_name: str) -> str:
    metrics = {
        "service": service_name,
        "cpu_percent": round(random.uniform(60, 98), 1),
        "memory_percent": round(random.uniform(70, 95), 1),
        "disk_percent": round(random.uniform(40, 85), 1),
        "open_connections": random.randint(80, 500),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    return json.dumps(metrics)


def get_service_status(service_name: str) -> str:
    statuses = ["running", "running", "running", "degraded", "down"]
    status = random.choice(statuses)
    result = {
        "service": service_name,
        "status": status,
        "uptime_seconds": random.randint(0, 86400) if status != "down" else 0,
        "last_restart": (datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 48))).strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    return json.dumps(result)


# Dispatcher — maps tool name → actual function call
def call_tool(tool_name: str, tool_args: dict) -> str:
    service_name = tool_args.get("service_name") or tool_args.get("name") or "unknown-service"

    if tool_name == "get_service_logs":
        return get_service_logs(
            service_name=service_name,
            last_minutes=tool_args.get("last_minutes", 10)
        )
    elif tool_name == "get_system_metrics":
        return get_system_metrics(service_name=service_name)
    elif tool_name == "get_service_status":
        return get_service_status(service_name=service_name)
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
