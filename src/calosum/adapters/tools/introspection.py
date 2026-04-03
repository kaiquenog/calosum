from __future__ import annotations

import ast
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from calosum.shared.models.types import DirectiveType, EvolutionDirective

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
            
            # Workspace and awareness are now async in the stateless orchestrator
            from calosum.shared.utils.async_utils import maybe_await
            workspace = await maybe_await(agent.aload_workspace_for_session(session_id))
            awareness = await maybe_await(agent.alatest_awareness_for_session(session_id))
            
            resolved_session_id = (
                session_id
                or (workspace.task_frame.get("session_id") if workspace else None)
                or "introspect-session"
            )
            if awareness is None:
                awareness = await maybe_await(agent.aanalyze_session(resolved_session_id, persist=False))

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

    async def query_session_stats(self, payload: dict[str, Any]) -> str:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            return "session_id é obrigatório."
        last_n = max(1, int(payload.get("last_n", 10)))
        agent, _ = self._resolve_agent()
        dashboard = agent.cognitive_dashboard(session_id)
        decisions = list(dashboard.get("decision", []))[-last_n:]
        executions = list(dashboard.get("execution", []))[-last_n:]
        felt = list(dashboard.get("felt", []))[-last_n:]

        def _avg(rows: list[dict[str, Any]], key: str, default: float) -> float:
            if not rows:
                return default
            return sum(float(item.get(key, default)) for item in rows) / len(rows)

        failures: dict[str, int] = {}
        for event in executions:
            for result in event.get("results", []):
                if result.get("status") != "rejected":
                    continue
                output = result.get("output", {})
                failure_type = str(output.get("error_type", "runtime_rejection")) if isinstance(output, dict) else "runtime_rejection"
                failures[failure_type] = failures.get(failure_type, 0) + 1

        dominant_failure = "none"
        if failures:
            name, count = max(failures.items(), key=lambda item: item[1])
            dominant_failure = f"{name} ({count}x)"

        payload_out = {
            "session_id": session_id,
            "last_n": last_n,
            "tool_success_rate": round(_avg(decisions, "tool_success_rate", 1.0), 3),
            "avg_retries": round(_avg(decisions, "runtime_retry_count", 0.0), 3),
            "avg_surprise": round(_avg(felt, "surprise_score", 0.0), 3),
            "dominant_failure": dominant_failure,
        }
        return json.dumps(payload_out, ensure_ascii=False)

    async def explain_last_decision(self, payload: dict[str, Any]) -> str:
        turn_id = str(payload.get("turn_id", "")).strip()
        session_id = str(payload.get("session_id", "")).strip() or None
        agent, _ = self._resolve_agent()
        dashboard = agent.cognitive_dashboard(session_id)

        if not turn_id:
            recent_turns = [item.get("_turn_id", "") for item in dashboard.get("decision", [])]
            recent_turns = [item for item in recent_turns if item]
            if not recent_turns:
                return "Nenhuma decisão encontrada para explicar."
            turn_id = recent_turns[-1]

        def _by_turn(channel: str) -> dict[str, Any]:
            for item in dashboard.get(channel, []):
                if str(item.get("_turn_id", "")) == turn_id:
                    return item
            return {}

        thought = _by_turn("thought")
        felt = _by_turn("felt")
        output = {
            "turn_id": turn_id,
            "right_state": {
                "surprise_score": felt.get("surprise_score"),
                "salience": felt.get("salience"),
                "emotional_labels": felt.get("emotional_labels"),
                "world_hypotheses": felt.get("world_hypotheses"),
            },
            "bridge_directives": thought.get("system_directives", []),
            "reasoning": thought.get("reasoning_summary", []),
            "critique_verdict": thought.get("critique_verdict"),
        }
        return json.dumps(output, ensure_ascii=False)

    async def read_architecture(self, payload: dict[str, Any]) -> str:
        component_name = str(payload.get("component_name", "")).strip()
        if not component_name:
            return "component_name é obrigatório."
        root = Path(__file__).resolve().parents[2]
        needle = component_name.lower()
        candidates: list[Path] = []
        for path in root.rglob("*.py"):
            if ".venv" in path.parts:
                continue
            if needle in path.stem.lower() or needle in str(path).lower():
                candidates.append(path)
        if not candidates:
            for path in root.rglob("*.py"):
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for node in tree.body:
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.lower() == needle:
                        candidates.append(path)
                        break
                if candidates:
                    break
        if not candidates:
            return f"Componente '{component_name}' não encontrado no código-fonte."

        target = candidates[0]
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source)
        dependencies: list[str] = []
        docstrings: list[str] = []
        module_doc = ast.get_docstring(tree)
        if module_doc:
            docstrings.append(module_doc.strip())
        for node in tree.body:
            if isinstance(node, ast.Import):
                dependencies.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                dependencies.append(module)
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.lower() == needle:
                    doc = ast.get_docstring(node)
                    if doc:
                        docstrings.append(doc.strip())
        payload_out = {
            "component_name": component_name,
            "path": str(target),
            "dependencies": sorted(set(item for item in dependencies if item)),
            "docstrings": docstrings,
            "source_code": source,
        }
        return json.dumps(payload_out, ensure_ascii=False)

    async def propose_config_change(self, payload: dict[str, Any]) -> str:
        parameter = str(payload.get("parameter", "")).strip()
        reason = str(payload.get("reason", "")).strip()
        raw_new_value = payload.get("new_value")
        if not parameter or not reason:
            return "parameter e reason são obrigatórios."

        parsed_new_value = raw_new_value
        if isinstance(raw_new_value, str):
            try:
                parsed_new_value = json.loads(raw_new_value)
            except json.JSONDecodeError:
                parsed_new_value = raw_new_value

        if "." in parameter:
            target_component, key = parameter.split(".", 1)
        else:
            target_component, key = "orchestrator", parameter
        directive = EvolutionDirective(
            directive_id=f"manual-{uuid.uuid4()}",
            directive_type=DirectiveType.PARAMETER,
            target_component=target_component,
            proposed_change={key: parsed_new_value},
            reasoning=reason,
            status="pending",
        )
        agent, _ = self._resolve_agent()
        agent.evolution_manager.queue_directive(directive)
        response = {
            "status": "queued",
            "directive_id": directive.directive_id,
            "target_component": directive.target_component,
            "proposed_change": directive.proposed_change,
        }
        return json.dumps(response, ensure_ascii=False)

    def _resolve_agent(self) -> tuple[Any, Any]:
        if self.agent_accessor is None:
            raise RuntimeError("Acesso ao self-model não configurado para este runtime.")
        agent, builder = self.agent_accessor()
        return agent, builder
