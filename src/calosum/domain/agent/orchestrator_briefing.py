from __future__ import annotations

import json
from typing import Any


def build_session_briefing(
    agent: Any,
    session_id: str,
    *,
    right_state: Any | None = None,
    last_n: int = 10,
) -> str:
    """
    Gera um resumo técnico do estado atual da sessão para o Left Hemisphere.
    """
    dashboard = agent.cognitive_dashboard(session_id)
    decisions = list(dashboard.get("decision", []))[-last_n:]
    felt = list(dashboard.get("felt", []))[-last_n:]
    executions = list(dashboard.get("execution", []))[-last_n:]
    turn_number = len(dashboard.get("decision", [])) + 1
    
    def _avg(rows: list[dict[str, Any]], key: str, default: float) -> float:
        if not rows:
            return default
        return sum(float(item.get(key, default)) for item in rows) / len(rows)
    
    tool_success_rate = _avg(decisions, "tool_success_rate", 1.0)
    avg_retries = _avg(decisions, "runtime_retry_count", 0.0)
    avg_surprise = _avg(felt, "surprise_score", 0.0)
    
    uncertainty = None
    if right_state is not None:
        uncertainty = right_state.telemetry.get("jepa_uncertainty")
    if uncertainty is None and felt:
        telemetry_values = [entry.get("telemetry", {}) for entry in felt]
        uncertainty_values = [
            float(item.get("jepa_uncertainty"))
            for item in telemetry_values
            if isinstance(item, dict) and item.get("jepa_uncertainty") is not None
        ]
        if uncertainty_values:
            uncertainty = sum(uncertainty_values) / len(uncertainty_values)
            
    uncertainty = float(uncertainty or 0.0)
    
    failures: dict[tuple[str, str], int] = {}
    for event in executions:
        for result in event.get("results", []):
            if result.get("status") != "rejected":
                continue
            action_type = str(result.get("action_type", "unknown_tool"))
            output = result.get("output", {})
            if isinstance(output, dict):
                error_type = str(output.get("error_type", "runtime_rejection")).upper()
            else:
                error_type = "RUNTIME_REJECTION"
            key = (error_type, action_type)
            failures[key] = failures.get(key, 0) + 1
            
    dominant_failure = "none"
    if failures:
        (error_type, action_type), count = max(failures.items(), key=lambda item: item[1])
        dominant_failure = f"{error_type} in {action_type} ({count}x)"
        
    pending = [item for item in agent.pending_directives if item.status == "pending"]
    pending_summary = ", ".join(
        f"{directive.target_component}:{directive.reasoning[:50].strip()}"
        for directive in pending[:2]
    ) if pending else "none"
    
    threshold_note = "below 80% threshold" if tool_success_rate < 0.8 else "within healthy band"
    
    return (
        f"[SESSION BRIEFING - Turn {turn_number} | session: {session_id}]\n"
        f"Tool success rate (last {last_n}): {tool_success_rate:.0%} ({threshold_note})\n"
        f"Avg retries: {avg_retries:.2f}\n"
        f"Dominant failure: {dominant_failure}\n"
        f"Avg surprise: {avg_surprise:.2f} | JEPA uncertainty: {uncertainty:.2f}\n"
        f"Active evolution directives: {pending_summary}"
    )
