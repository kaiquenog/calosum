import logging
from fastapi import APIRouter
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
