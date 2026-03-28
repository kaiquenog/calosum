import asyncio
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from calosum.bootstrap.factory import CalosumAgentBuilder
from calosum.bootstrap.settings import InfrastructureSettings
from calosum.shared.serialization import to_primitive
from calosum.shared.types import UserTurn

logger = logging.getLogger(__name__)

app = FastAPI(title="Calosum API")

# Initialize isolated dependencies during startup
settings = InfrastructureSettings.from_sources()
builder = CalosumAgentBuilder(settings)
agent = builder.build()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    try:
        data: dict[str, Any] = await request.json()
    except Exception:
        data = {}

    session_id = data.get("session_id", "api-session")
    text = data.get("text", "")
    
    if not text:
        return JSONResponse(
            {"status": "error", "error": "Request must contain text"}, 
            status_code=400
        )
    
    user_turn = UserTurn(session_id=session_id, user_text=text, signals=[])
    
    try:
        # Offload logic execution to prevent blocking the async loop
        def run() -> Any:
            return agent.process_turn(user_turn)
            
        result = await asyncio.to_thread(run)
        return JSONResponse({"status": "ok", "result": to_primitive(result)})
    except Exception as e:
        logger.error("Error processing turn", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port)
