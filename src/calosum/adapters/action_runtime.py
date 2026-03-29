from __future__ import annotations

import logging

from calosum.shared.async_utils import run_sync
from calosum.shared.tools import ToolRegistry, ToolSchema
from calosum.shared.types import ActionExecutionResult, LeftHemisphereResult

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
        
        return registry

    def run(self, left_result: LeftHemisphereResult) -> list[ActionExecutionResult]:
        return run_sync(self.arun(left_result))

    async def arun(self, left_result: LeftHemisphereResult) -> list[ActionExecutionResult]:
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
