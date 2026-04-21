"""ClearTrace cassette bridge.

Loads ClearTrace cassette files and converts recorded agent traces
into the trajectory format for explainability analysis.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from clear_trace.explain.core.types import ToolDecision, TrajectoryExplanation


def load_cleartrace_cassette(path: str | Path) -> Dict[str, Any]:
    """Load a ClearTrace cassette YAML file.

    Args:
        path: Path to the .yaml cassette file.

    Returns:
        The parsed cassette data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cassette not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def cassette_to_decisions(cassette: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a ClearTrace cassette into a list of decision dicts.

    Maps ClearTrace trace events to the format expected by
    TrajectoryAttributor.explain_trajectory().

    Expected cassette structure:
        trace:
          events:
            - type: llm_call | tool_call | agent_decision
              ...

    Returns:
        List of dicts with keys: tool, context, output, alternatives
    """
    trace = cassette.get("trace", {})
    events = trace.get("events", [])
    decisions: List[Dict[str, Any]] = []

    for event in events:
        event_type = event.get("type", "")

        if event_type == "tool_call":
            decisions.append({
                "tool": event.get("tool_name", event.get("name", "unknown_tool")),
                "context": _extract_context(event),
                "output": str(event.get("result", event.get("output", ""))),
            })
        elif event_type == "llm_call":
            decisions.append({
                "tool": f"llm:{event.get('model', 'unknown')}",
                "context": _extract_llm_context(event),
                "output": event.get("response", {}).get("content", ""),
            })
        elif event_type == "agent_decision":
            decisions.append({
                "tool": event.get("action", "decide"),
                "context": event.get("reasoning", ""),
                "output": event.get("result", ""),
                "alternatives": event.get("alternatives", []),
            })

    return decisions


def cassette_to_trajectory(cassette: Dict[str, Any]) -> TrajectoryExplanation:
    """Convert a ClearTrace cassette directly into a TrajectoryExplanation."""
    raw_decisions = cassette_to_decisions(cassette)

    tool_decisions = [
        ToolDecision(
            step=i,
            tool_name=d["tool"],
            context_snippet=d.get("context", "")[:500],
            alternatives=d.get("alternatives", []),
        )
        for i, d in enumerate(raw_decisions)
    ]

    path = " → ".join(d.tool_name for d in tool_decisions)
    summary = f"Imported from ClearTrace cassette: {len(tool_decisions)} steps\nPath: {path}"

    return TrajectoryExplanation(
        decisions=tool_decisions,
        trajectory_summary=summary,
    )


def _extract_context(event: Dict[str, Any]) -> str:
    """Extract context string from a tool call event."""
    parts = []
    if "arguments" in event:
        args = event["arguments"]
        if isinstance(args, dict):
            parts.append(str(args))
        else:
            parts.append(str(args))
    if "description" in event:
        parts.append(event["description"])
    return " | ".join(parts) if parts else ""


def _extract_llm_context(event: Dict[str, Any]) -> str:
    """Extract context from an LLM call event."""
    messages = event.get("request", {}).get("messages", [])
    if messages:
        # Use the last user message as context
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))[:500]
    return str(event.get("request", ""))[:500]
