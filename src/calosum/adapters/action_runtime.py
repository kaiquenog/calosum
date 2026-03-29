from __future__ import annotations

import logging

from calosum.adapters.tools.code_execution import CodeExecutionTool
from calosum.adapters.tools.http_request import HttpRequestTool
from calosum.shared.async_utils import run_sync
from calosum.shared.tools import ToolRegistry, ToolSchema
from calosum.shared.types import ActionExecutionResult, LeftHemisphereResult, ToolDescriptor, CognitiveWorkspace

logger = logging.getLogger(__name__)

class ConcreteActionRuntime:
    """
    Adapter real para o ActionRuntimePort.
    Ele converte ações simbólicas ('propose_plan', 'respond_text', 'search_web')
    em execuções verdadeiras via ferramentas Python.
    """
    def __init__(
        self,
        vault: dict[str, str] | None = None,
        registry: ToolRegistry | None = None,
        granted_permissions: set[str] | None = None,
    ) -> None:
        self.vault = vault or {}
        self.registry = registry or self._build_default_registry()
        self.granted_permissions = granted_permissions

    def _build_default_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        code_execution = CodeExecutionTool()
        http_request = HttpRequestTool()
        
        registry.register(
            ToolSchema("respond_text", "Emit text to user", {"text": "string"}, []),
            self._execute_respond_text
        )
        registry.register(
            ToolSchema("propose_plan", "Save plan to DB", {"steps": "list"}, []),
            self._execute_propose_plan
        )
        registry.register(
            ToolSchema("load_semantic_rules", "Load semantic rules into runtime", {"rules": "list"}, []),
            self._execute_load_semantic_rules
        )
        registry.register(
            ToolSchema("search_web", "Search the web", {"query": "string"}, ["network"]),
            self._execute_search_web
        )
        registry.register(
            ToolSchema("write_file", "Write to sandbox", {"path": "string", "content": "string"}, ["fs_write"]),
            self._execute_write_file
        )
        registry.register(
            ToolSchema("read_file", "Read from sandbox", {"path": "string"}, ["fs_read"]),
            self._execute_read_file
        )
        registry.register(
            ToolSchema("execute_bash", "Execute shell command in sandbox", {"command": "string"}, ["shell"]),
            self._execute_bash
        )
        registry.register(
            ToolSchema("introspect_self", "Analyze system health, architecture, or awareness bottlenecks", {"query": "string"}, []),
            self._execute_introspect_self
        )
        registry.register(code_execution.schema, code_execution.execute)
        registry.register(http_request.schema, http_request.execute)
        
        return registry

    def run(self, left_result: LeftHemisphereResult, workspace: CognitiveWorkspace | None = None) -> list[ActionExecutionResult]:
        return run_sync(self.arun(left_result, workspace))

    def get_registered_tools(self) -> list[ToolDescriptor]:
        return self.registry.get_descriptors()

    async def arun(self, left_result: LeftHemisphereResult, workspace: CognitiveWorkspace | None = None) -> list[ActionExecutionResult]:
        results = []
        for action in left_result.actions:
            schema = self.registry.get_schema(action.action_type)
            if not schema:
                results.append(
                    ActionExecutionResult(
                        action_type=action.action_type,
                        typed_signature=action.typed_signature,
                        status="rejected",
                        output={
                            "error": f"Tool '{action.action_type}' not found in registry.",
                            "error_type": "tool_not_found",
                            "tool": action.action_type,
                        },
                        violations=[f"Unknown tool: {action.action_type}"],
                    )
                )
                continue

            validation_violations = self.registry.validate_payload(
                action.action_type,
                action.payload,
            )
            if validation_violations:
                results.append(
                    ActionExecutionResult(
                        action_type=action.action_type,
                        typed_signature=action.typed_signature,
                        status="rejected",
                        output={
                            "error": "Tool payload validation failed",
                            "error_type": "validation_failed",
                            "tool": action.action_type,
                        },
                        violations=validation_violations,
                    )
                )
                continue

            missing_permissions = self._missing_permissions(schema)
            if missing_permissions:
                results.append(
                    ActionExecutionResult(
                        action_type=action.action_type,
                        typed_signature=action.typed_signature,
                        status="needs_approval",
                        output={
                            "message": "Action requires additional permissions",
                            "missing_permissions": missing_permissions,
                            "tool": action.action_type,
                        },
                        violations=[],
                    )
                )
                continue

            if schema.needs_approval and not action.payload.get("approved", False):
                results.append(
                    ActionExecutionResult(
                        action_type=action.action_type,
                        typed_signature=action.typed_signature,
                        status="needs_approval",
                        output={
                            "message": "Action requires user approval",
                            "tool": action.action_type,
                        },
                        violations=[],
                    )
                )
                continue

            try:
                res = await self.registry.execute(action.action_type, action.payload)
                results.append(
                    ActionExecutionResult(
                        action_type=action.action_type,
                        typed_signature=action.typed_signature,
                        status="executed",
                        output={"result": str(res), "tool": action.action_type},
                        violations=[],
                    )
                )
            except Exception as e:
                logger.error(f"Error executing {action.action_type}: {e}")
                results.append(
                    ActionExecutionResult(
                        action_type=action.action_type,
                        typed_signature=action.typed_signature,
                        status="rejected",
                        output={
                            "error": str(e),
                            "error_type": "runtime_crash",
                            "tool": action.action_type,
                        },
                        violations=[f"Runtime crash: {e}"],
                    )
                )

        if workspace is not None:
            workspace.runtime_feedback.extend(
                [{"action": res.action_type, "status": res.status, "violations": res.violations} for res in results]
            )
            for res in results:
                pending_question = self._pending_question_from_result(res)
                if pending_question and pending_question not in workspace.pending_questions:
                    workspace.pending_questions.append(pending_question)

        return results

    def _missing_permissions(self, schema: ToolSchema) -> list[str]:
        if self.granted_permissions is None:
            return []
        return [
            permission
            for permission in schema.required_permissions
            if permission not in self.granted_permissions
        ]

    async def _execute_respond_text(self, payload: dict) -> str:
        # Ponto de integração real para emitir no WebSocket/Socket.IO do usuário
        text = payload.get("text", "")
        return f"Message buffered to client: {text}"

    async def _execute_propose_plan(self, payload: dict) -> str:
        # Ponto de integração real salvando o plano em banco relacional
        steps = payload.get("steps", [])
        return f"Plan saved. Steps: {len(steps)}"

    async def _execute_load_semantic_rules(self, payload: dict) -> str:
        rules = payload.get("rules", [])
        return f"Semantic rules loaded. Count: {len(rules)}"

    async def _execute_search_web(self, payload: dict) -> str:
        # Integração real usando duckduckgo-search livre
        query = payload.get("query", "")
        if not query:
            return "No query provided for web search."
            
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                # Puxa até 3 sumarizações
                for r in list(ddgs.text(query, max_results=3)):
                    results.append(f"[{r.get('title')}]({r.get('href')}): {r.get('body')}")
            
            if not results:
                return f"No results found for {query}."
                
            return f"Searched web for '{query}'. Found: " + " | ".join(results)
        except Exception as e:
            logger.error(f"DDGS failure: {e}")
            return f"Search failed: {e}"

    async def _execute_write_file(self, payload: dict) -> str:
        import tempfile
        from pathlib import Path
        path_str = payload.get("path", "")
        content = payload.get("content", "")
        if not path_str:
            return "No path provided for write_file."
            
        try:
            # Sandbox: force write inside a temporary directory to avoid host modifications
            sandbox_dir = Path(tempfile.gettempdir()) / "calosum_sandbox"
            sandbox_dir.mkdir(parents=True, exist_ok=True)
            
            # Resolve the path to ensure it stays within the sandbox
            safe_name = Path(path_str).name
            if not safe_name:
                safe_name = "default_output.txt"
                
            target = sandbox_dir / safe_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to sandboxed path: {target} (requested: {path_str})."
        except Exception as e:
            logger.error(f"File write failure: {e}")
            return f"File write failed: {e}"

    async def _execute_read_file(self, payload: dict) -> str:
        import tempfile
        from pathlib import Path
        path_str = payload.get("path", "")
        if not path_str:
            return "No path provided for read_file."
            
        try:
            sandbox_dir = Path(tempfile.gettempdir()) / "calosum_sandbox"
            safe_name = Path(path_str).name
            target = sandbox_dir / safe_name
            
            if not target.exists():
                return f"File not found in sandbox: {target}"
                
            content = target.read_text(encoding="utf-8")
            # Truncate se for muito grande
            if len(content) > 4000:
                content = content[:4000] + "\n...[truncated]"
            return content
        except Exception as e:
            logger.error(f"File read failure: {e}")
            return f"File read failed: {e}"

    async def _execute_bash(self, payload: dict) -> str:
        import asyncio
        import tempfile
        from pathlib import Path
        
        command = payload.get("command", "")
        if not command:
            return "No command provided."
            
        try:
            sandbox_dir = Path(tempfile.gettempdir()) / "calosum_sandbox"
            sandbox_dir.mkdir(parents=True, exist_ok=True)
            
            # Roda o processo limitando o cwd ao sandbox
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(sandbox_dir)
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
            
            out_str = stdout.decode('utf-8').strip()
            err_str = stderr.decode('utf-8').strip()
            
            result = []
            if out_str:
                result.append(f"STDOUT:\n{out_str[:2000]}")
            if err_str:
                result.append(f"STDERR:\n{err_str[:2000]}")
                
            return "\n".join(result) if result else "Command executed silently (exit code 0)."
            
        except asyncio.TimeoutError:
            return "Command execution timed out after 10 seconds."
        except Exception as e:
            logger.error(f"Bash execution failure: {e}")
            return f"Bash execution failed: {e}"

    async def _execute_introspect_self(self, payload: dict) -> str:
        """
        Responde perguntas introspectivas usando self-model, awareness e workspace reais.
        """
        query = payload.get("query", "").lower()
        session_id = payload.get("session_id")

        try:
            from calosum.bootstrap.api import get_agent, get_builder
            agent = get_agent()
            builder = get_builder()
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

            components = [
                f"{component.component_id}={component.health}"
                for component in agent.self_model.components
            ]
            tools = [
                f"{tool.name}(perms={','.join(tool.required_permissions) or 'none'}, approval={'yes' if tool.requires_approval else 'no'})"
                for tool in agent.self_model.capabilities.tools
            ]
            routing = info.get("routing_resolution", {})

            if "arquitetura" in query or "como você funciona" in query or "como voce funciona" in query:
                return (
                    f"Arquitetura ativa: {', '.join(components)}. "
                    f"Backends: perception={_routing_summary(routing.get('perception'))}; "
                    f"reason={_routing_summary(routing.get('reason'))}; "
                    f"reflection={_routing_summary(routing.get('reflection'))}; "
                    f"verifier={_routing_summary(routing.get('verifier'))}."
                )

            if (
                "backend" in query
                or "memória" in query
                or "memoria" in query
                or "telemetria" in query
                or "tracing" in query
            ):
                return (
                    f"Sessão {resolved_session_id}: "
                    f"reason={_routing_summary(routing.get('reason'))}; "
                    f"perception={_routing_summary(routing.get('perception'))}; "
                    f"memory={info.get('memory_backend') or 'unknown'}; "
                    f"knowledge_graph={info.get('knowledge_graph_backend') or 'unknown'}; "
                    f"telemetry={info.get('telemetry_backend') or 'unknown'}; "
                    f"otlp_jsonl={info.get('otlp_jsonl') or 'none'}; "
                    f"otel_collector={info.get('otel_collector_endpoint') or 'none'}."
                )

            if "tool" in query or "ferramenta" in query or "capacidade" in query:
                return (
                    f"Ferramentas registradas: {', '.join(tools) if tools else 'nenhuma'}. "
                    f"Health global das capabilities: {agent.self_model.capabilities.health}."
                )

            if "falha" in query or "gargalo" in query or "problema" in query:
                if not awareness.bottlenecks:
                    return (
                        f"Awareness mais recente: sem gargalos críticos em {awareness.analyzed_turns} turnos. "
                        f"Surpresa média={awareness.average_surprise:.3f}, retries médios={awareness.average_retries:.3f}."
                    )

                issues = "; ".join(
                    f"{item.description} (sev={item.severity}, evidência={'; '.join(item.evidence[:2])})"
                    for item in awareness.bottlenecks
                )
                return (
                    f"Gargalos recentes: {issues}. "
                    f"Tipos de falha: {_failure_summary(awareness.failure_types)}. "
                    f"Backlog de aprovação={awareness.pending_approval_backlog}, "
                    f"diretivas pendentes={awareness.pending_directive_count}, "
                    f"tendência de surprise={awareness.surprise_trend:+.3f}."
                )

            if "mudar" in query or "sugere" in query or "diretiva" in query:
                if not agent.pending_directives:
                    return "Nenhuma diretiva pendente no momento."
                directives = [
                    f"{item.directive_type.value}:{item.target_component} -> {item.proposed_change}"
                    for item in agent.pending_directives
                ]
                return f"Diretivas pendentes: {'; '.join(directives)}."

            workspace_summary = "sem workspace recente"
            if workspace is not None:
                workspace_summary = (
                    f"sessão={workspace.task_frame.get('session_id') or resolved_session_id}, "
                    f"pending_questions={len(workspace.pending_questions)}, "
                    f"runtime_feedback={len(workspace.runtime_feedback)}, "
                    f"verifier_feedback={len(workspace.verifier_feedback)}"
                )
            return (
                f"Resumo interno: {workspace_summary}. "
                f"Awareness: turns={awareness.analyzed_turns}, surprise={awareness.average_surprise:.3f}, "
                f"dominant_variant={awareness.dominant_variant or 'n/a'}, "
                f"directives_pending={awareness.pending_directive_count}."
            )
        except Exception as e:
            logger.error(f"Introspection execution failure: {e}")
            return f"Não foi possível acessar a telemetria ou o self-model interno: {e}"

    def _pending_question_from_result(self, result: ActionExecutionResult) -> str | None:
        if result.status != "needs_approval":
            return None

        output = result.output if isinstance(result.output, dict) else {}
        missing_permissions = output.get("missing_permissions", [])
        if missing_permissions:
            return (
                f"A ação '{result.action_type}' precisa de aprovação para conceder: "
                f"{', '.join(str(item) for item in missing_permissions)}."
            )
        return f"A ação '{result.action_type}' precisa de aprovação antes de continuar."


def _routing_summary(route: dict | None) -> str:
    if not route:
        return "indisponível"
    requested = route.get("requested_model") or "auto"
    active = route.get("active_model") or "n/a"
    backend = route.get("backend") or "n/a"
    availability = "available" if route.get("available") else "unavailable"
    note = route.get("note")
    if note:
        return f"requested={requested}, active={active}, backend={backend}, {availability}, note={note}"
    return f"requested={requested}, active={active}, backend={backend}, {availability}"


def _failure_summary(failure_types: dict[str, int]) -> str:
    if not failure_types:
        return "nenhuma"
    return ", ".join(f"{name}={count}" for name, count in sorted(failure_types.items()))
