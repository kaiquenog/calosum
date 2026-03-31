import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from calosum.shared.serialization import to_primitive
from calosum.bootstrap.context import get_agent, get_builder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/system")

@router.get("/info")
async def system_info() -> JSONResponse:
    try:
        builder, agent = get_builder(), get_agent()
        info = builder.describe(agent)
        return JSONResponse({"status": "ok", "info": info})
    except Exception as e:
        logger.error("Error retrieving system info", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@router.get("/architecture")
async def system_architecture() -> JSONResponse:
    try:
        agent = get_agent()
        return JSONResponse({"status": "ok", "architecture": to_primitive(agent.self_model)})
    except Exception as e:
        logger.error("Error retrieving system architecture", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@router.get("/capabilities")
async def system_capabilities() -> JSONResponse:
    try:
        agent = get_agent()
        return JSONResponse({"status": "ok", "capabilities": to_primitive(agent.self_model.capabilities)})
    except Exception as e:
        logger.error("Error retrieving system capabilities", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@router.get("/state")
async def system_state(session_id: str | None = None) -> JSONResponse:
    try:
        agent = get_agent()
        target_session = session_id or "api-session"
        workspace = agent.workspace_for_session(target_session)
        if workspace is None and session_id is None:
            workspace = agent.workspace_for_session()
        if not workspace:
            return JSONResponse(
                {
                    "status": "ok",
                    "state": None,
                    "session_id": target_session,
                    "note": "No cognitive workspace found yet",
                }
            )
        return JSONResponse({"status": "ok", "state": to_primitive(workspace)})
    except Exception as e:
        logger.error("Error retrieving system state", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@router.get("/awareness")
async def system_awareness(session_id: str | None = None) -> JSONResponse:
    try:
        agent = get_agent()
        target_session = session_id or "api-session"
        diagnostic = agent.analyze_session(target_session, persist=False)
        return JSONResponse({"status": "ok", "diagnostic": to_primitive(diagnostic)})
    except Exception as e:
        logger.error("Error generating system awareness", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@router.get("/directives")
async def system_directives() -> JSONResponse:
    try:
        agent = get_agent()
        return JSONResponse({"status": "ok", "directives": to_primitive(agent.pending_directives)})
    except Exception as e:
        logger.error("Error retrieving system directives", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@router.post("/directives/apply")
async def apply_directive(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        directive_id = data.get("directive_id")
        if not directive_id:
            return JSONResponse({"status": "error", "error": "directive_id is required"}, status_code=400)
        agent = get_agent()
        target_directive = agent.apply_pending_directive(directive_id)
        if target_directive is None:
            return JSONResponse({"status": "error", "error": "Directive not found"}, status_code=404)
        return JSONResponse({"status": "ok", "directive": to_primitive(target_directive)})
    except Exception as e:
        logger.error("Error applying directive", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@router.post("/introspect")
async def system_introspect(request: Request) -> JSONResponse:
    try:
        data = await request.json()
        query = data.get("query", "")
        session_id = data.get("session_id", "introspect-session")
        if not query:
            return JSONResponse({"status": "error", "error": "Query is required"}, status_code=400)
        agent = get_agent()
        tool_payload = {"query": query, "session_id": session_id}
        result_text = await agent.action_runtime.registry.execute("introspect_self", tool_payload)
        return JSONResponse({"status": "ok", "response": result_text})
    except Exception as e:
        logger.error("Error running system introspection", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@router.post("/idle")
async def trigger_idle_foraging() -> JSONResponse:
    try:
        agent = get_agent()
        result = await agent.aidle_foraging()
        return JSONResponse({"status": "ok", "result": to_primitive(result)})
    except Exception as e:
        logger.error("Error running idle foraging", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)
