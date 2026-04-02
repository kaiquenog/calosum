import logging
from typing import Any
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from calosum.shared.utils.serialization import to_primitive
from calosum.shared.models.types import UserTurn
from calosum.bootstrap.entry.context import get_agent, _run_in_session_lane

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/chat")

@router.post("/completions")
async def chat_completions(request: Request) -> JSONResponse:
    try:
        data: dict[str, Any] = await request.json()
    except Exception:
        data = {}
    from calosum.shared.models.types import MultimodalSignal, Modality
    signals = []
    if "signals" in data:
        for s in data["signals"]:
            signals.append(MultimodalSignal(
                modality=Modality(s["modality"]),
                source=s.get("source", "api"),
                payload=s.get("payload", b"").encode() if isinstance(s.get("payload"), str) else s.get("payload", b"")
            ))
    session_id = data.get("session_id", "api-session")
    text = data.get("text", "")
    user_turn = UserTurn(session_id=session_id, user_text=text, signals=signals)
    agent = get_agent()
    try:
        result = await _run_in_session_lane(
            user_turn.session_id,
            lambda: agent.aprocess_turn(user_turn),
        )
        return JSONResponse({"status": "ok", "result": to_primitive(result)})
    except Exception as e:
        logger.error("Error processing turn", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@router.get("/sse")
async def chat_sse(request: Request, text: str, session_id: str = "api-session"):
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)
    async def event_generator():
        yield {"event": "status", "data": "processing"}
        agent = get_agent()
        user_turn = UserTurn(session_id=session_id, user_text=text, signals=[])
        try:
            result = await _run_in_session_lane(
                user_turn.session_id,
                lambda: agent.aprocess_turn(user_turn),
            )
            turn_result = result.selected_result if hasattr(result, "selected_result") else result
            yield {"event": "reasoning", "data": turn_result.left_result.response_text or "..."}
            for exec_res in turn_result.execution_results:
                yield {"event": "action", "data": f"[{exec_res.action_type}] {exec_res.status}"}
            yield {"event": "done", "data": "success"}
        except Exception as e:
            logger.error("SSE Error", exc_info=True)
            yield {"event": "error", "data": str(e)}
    return EventSourceResponse(event_generator())
