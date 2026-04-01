from __future__ import annotations
from typing import TYPE_CHECKING
from calosum.shared.types import DirectiveType, SessionDiagnostic

if TYPE_CHECKING:
    from calosum.domain.orchestrator import CalosumAgent

async def process_awareness_loop(agent: CalosumAgent, session_id: str) -> None:
    agent._awareness_turn_counts[session_id] = agent._awareness_turn_counts.get(session_id, 0) + 1
    if agent._awareness_turn_counts[session_id] % max(1, agent.config.awareness_interval_turns) != 0:
        return

    diagnostic = analyze_session(agent, session_id, persist=True)
    if not diagnostic.bottlenecks:
        return

    for directive in agent.evolution_manager.proposer.propose(diagnostic):
        if directive.directive_type == DirectiveType.PARAMETER:
            agent.evolution_manager.apply_directive(directive, agent)
            if agent.evolution_manager.archive:
                agent.evolution_manager.archive.record_directive(directive, event="auto_applied")
            continue
        agent.evolution_manager.queue_directive(directive)

def analyze_session(agent: CalosumAgent, session_id: str, *, persist: bool = False) -> SessionDiagnostic:
    dashboard = agent.cognitive_dashboard(session_id)
    if not dashboard.get("decision"):
        diagnostic = SessionDiagnostic(
            session_id=session_id, analyzed_turns=0, tool_success_rate=1.0,
            average_retries=0.0, average_surprise=0.0, bottlenecks=[],
            pending_approval_backlog=0,
            pending_directive_count=len(agent.evolution_manager.pending_directives))
    else:
        from calosum.domain.introspection import IntrospectionEngine
        diagnostic = IntrospectionEngine().analyze(
            session_id, dashboard,
            pending_directive_count=len(agent.evolution_manager.pending_directives))
    agent.latest_awareness_by_session[session_id] = diagnostic
    if persist:
        if hasattr(agent.telemetry_bus, "record_awareness"):
            agent.telemetry_bus.record_awareness(session_id, diagnostic)
        if agent.evolution_archive:
            agent.evolution_archive.record_diagnostic(diagnostic)
    return diagnostic
