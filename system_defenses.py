import json
from typing import Dict, Any

# ---------- D1: Tool Authorization ----------
ALLOWED_TOOLS = {
    "agent_star": ["lookup_account", "search_flights"],
    "agent_b0": ["search_flights"],
    "agent_b1": ["search_hotels"],
    "agent_b2": ["plan_itinerary"],
}

def authorize_tool(agent: str, tool: str) -> bool:
    return tool in ALLOWED_TOOLS.get(agent, [])


# ---------- D2: Least Privilege ----------
def least_privilege_segment(segment: str) -> str:
    return segment.replace("[ACCOUNT]", "")


# ---------- D3: Typed I/O Boundary ----------
def enforce_json_schema(output: str) -> str:
    try:
        data = json.loads(output)
        if "result" in data:
            return json.dumps({"result": str(data["result"])})
    except Exception:
        pass

    # fallback: sanitize to safe JSON
    return json.dumps({"result": "sanitized"})


# ---------- D4: Output Sanitization ----------
FORBIDDEN_MARKERS = ["[[A2A_TRIGGERED]]"]

def sanitize_output(output: str) -> str:
    for m in FORBIDDEN_MARKERS:
        output = output.replace(m, "")
    return output
