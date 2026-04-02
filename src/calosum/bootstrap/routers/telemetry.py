import logging
from collections import Counter

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from calosum.bootstrap.entry.context import get_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/telemetry")

@router.get("/dashboard/{session_id}")
async def get_dashboard(session_id: str) -> JSONResponse:
    try:
        agent = get_agent()
        dashboard = agent.cognitive_dashboard(session_id)
        return JSONResponse({"status": "ok", "dashboard": dashboard})
    except Exception as e:
        logger.error("Error retrieving dashboard", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@router.get("/dashboard")
async def get_global_dashboard() -> JSONResponse:
    try:
        agent = get_agent()
        dashboard = agent.cognitive_dashboard(None)
        return JSONResponse({"status": "ok", "dashboard": dashboard})
    except Exception as e:
        logger.error("Error retrieving dashboard", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@router.post("/query")
async def query_telemetry(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        session_id = str(data.get("session_id", "")).strip()
        question = str(data.get("question", "")).strip().lower()
        if not session_id or not question:
            return JSONResponse(
                {"status": "error", "error": "session_id and question are required"},
                status_code=400,
            )

        agent = get_agent()
        dashboard = agent.cognitive_dashboard(session_id)

        if "tool" in question and any(term in question for term in ("falh", "erro", "failure")):
            by_tool = Counter()
            by_error = Counter()
            for event in dashboard.get("execution", []):
                for result in event.get("results", []):
                    if result.get("status") != "rejected":
                        continue
                    tool_name = str(result.get("action_type", "unknown_tool"))
                    by_tool[tool_name] += 1
                    output = result.get("output", {})
                    error_name = str(output.get("error_type", "runtime_rejection")) if isinstance(output, dict) else "runtime_rejection"
                    by_error[error_name] += 1

            if not by_tool:
                answer = "Nao encontrei falhas de tool na janela de telemetria desta sessao."
            else:
                tool_text = ", ".join(f"{name}: {count}" for name, count in by_tool.most_common(5))
                error_text = ", ".join(f"{name}: {count}" for name, count in by_error.most_common(5))
                answer = f"Falhas por tool: {tool_text}. Tipos de erro: {error_text}."
            return JSONResponse({"status": "ok", "answer": answer})

        diagnostic = agent.analyze_session(session_id, persist=False)
        answer = (
            f"Sessao {session_id}: tool_success_rate={diagnostic.tool_success_rate:.0%}, "
            f"avg_retries={diagnostic.average_retries:.2f}, avg_surprise={diagnostic.average_surprise:.2f}, "
            f"pending_directives={diagnostic.pending_directive_count}."
        )
        return JSONResponse({"status": "ok", "answer": answer})
    except Exception as e:
        logger.error("Error querying telemetry", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)
