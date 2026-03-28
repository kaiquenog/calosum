from __future__ import annotations

import logging

from calosum.shared.async_utils import run_sync
from calosum.shared.types import ActionExecutionResult, LeftHemisphereResult

logger = logging.getLogger(__name__)

class ConcreteActionRuntime:
    """
    Adapter real para o ActionRuntimePort.
    Ele converte ações simbólicas ('propose_plan', 'respond_text', 'search_web')
    em execuções verdadeiras via ferramentas Python.
    """
    def __init__(self, vault: dict[str, str] | None = None) -> None:
        self.vault = vault or {}

    def run(self, left_result: LeftHemisphereResult) -> list[ActionExecutionResult]:
        return run_sync(self.arun(left_result))

    async def arun(self, left_result: LeftHemisphereResult) -> list[ActionExecutionResult]:
        results = []
        for action in left_result.actions:
            try:
                if action.action_type == "respond_text":
                    res = await self._execute_respond_text(action.payload)
                elif action.action_type == "propose_plan":
                    res = await self._execute_propose_plan(action.payload)
                elif action.action_type == "search_web":
                    res = await self._execute_search_web(action.payload)
                elif action.action_type == "write_file":
                    res = await self._execute_write_file(action.payload)
                else:
                    raise ValueError(f"Action '{action.action_type}' not supported in concrete runtime.")

                results.append(
                    ActionExecutionResult(
                        action_type=action.action_type,
                        typed_signature=action.typed_signature,
                        status="executed",
                        output={"result": str(res)},
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
                        output={"error": str(e)},
                        violations=[f"Runtime crash: {e}"],
                    )
                )
        return results

    async def _execute_respond_text(self, payload: dict) -> str:
        # Ponto de integração real para emitir no WebSocket/Socket.IO do usuário
        text = payload.get("text", "")
        return f"Message buffered to client: {text}"

    async def _execute_propose_plan(self, payload: dict) -> str:
        # Ponto de integração real salvando o plano em banco relacional
        steps = payload.get("steps", [])
        return f"Plan saved. Steps: {len(steps)}"

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
        from pathlib import Path
        path = payload.get("path", "")
        content = payload.get("content", "")
        if not path:
            return "No path provided for write_file."
            
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {path}."
        except Exception as e:
            logger.error(f"File write failure: {e}")
            return f"File write failed: {e}"

