from __future__ import annotations

import hashlib
import logging
from typing import Any
from calosum.shared.types import ComponentHealth

logger = logging.getLogger(__name__)

class IntrospectionTool:
    """
    Ferramenta para analisar saúde do sistema, arquitetura ou gargalos de awareness.
    Extraída do ActionRuntime para desacoplar Bootstrap de Adapters.
    """
    
    def __init__(self, agent_accessor: Any = None) -> None:
        # agent_accessor deve disparar o acesso ao agente/builder sem importar bootstrap.api
        self.agent_accessor = agent_accessor

    async def execute(self, payload: dict) -> str:
        query = payload.get("query", "").lower()
        session_id = payload.get("session_id")
        
        if self.agent_accessor is None:
            return "Acesso ao self-model não configurado para este runtime."

        try:
            agent, builder = self.agent_accessor()
            info = builder.describe(agent)
            workspace = agent.workspace_for_session(session_id)
            awareness = agent.latest_awareness_for_session(session_id)
            
            resolved_session_id = (
                session_id
                or (workspace.task_frame.get("session_id") if workspace else None)
                or "introspect-session"
            )
            if awareness is None:
                awareness = agent.analyze_session(resolved_session_id, persist=False)

            components = [f"{c.component_id}={c.health}" for c in agent.self_model.components]
            tools = [f"{t.name}" for t in agent.self_model.capabilities.tools]
            routing = info.get("routing_resolution", {})

            if "arquitetura" in query or "funcionamento" in query or "backend" in query:
                return (
                    f"Status Sessão {resolved_session_id}: turns={awareness.analyzed_turns}. "
                    f"Backends: reason={info.get('left_hemisphere_backend')}, "
                    f"memory={info.get('memory_backend')}, "
                    f"telemetry={info.get('telemetry_backend')}. "
                    f"Routing: {routing}"
                )

            if "falha" in query or "gargalo" in query:
                issues = "; ".join(f"{b.description}" for b in awareness.bottlenecks)
                return f"Gargalos: {issues or 'nenhum'}. Surprise trend: {awareness.surprise_trend:+.3f}"

            return f"Status Sessão {resolved_session_id}: turns={awareness.analyzed_turns}, tools={len(tools)}"
        except Exception as e:
            logger.error(f"Introspection tool crash: {e}")
            return f"Não foi possível acessar dados internos: {e}"
