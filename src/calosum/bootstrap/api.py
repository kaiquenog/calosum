import asyncio
import logging
import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from calosum.bootstrap.factory import CalosumAgentBuilder
from calosum.bootstrap.settings import (
    InfrastructureSettings,
    should_enable_local_persistence_defaults,
    with_local_persistence_defaults,
)
from calosum.shared.serialization import to_primitive
from calosum.shared.types import UserTurn

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Calosum API",
    docs_url="/docs",
    redoc_url="/redoc"
)


@lru_cache(maxsize=1)
def get_settings() -> InfrastructureSettings:
    return resolve_api_settings(os.environ)


def resolve_api_settings(environ: dict[str, str] | None = None) -> InfrastructureSettings:
    env = environ or os.environ
    settings = InfrastructureSettings.from_sources(environ=env)
    if should_enable_local_persistence_defaults(settings, environ=env):
        return with_local_persistence_defaults(settings)
    return settings


@lru_cache(maxsize=1)
def get_builder() -> CalosumAgentBuilder:
    return CalosumAgentBuilder(get_settings())


@lru_cache(maxsize=1)
def get_agent():
    return get_builder().build()

# Inicialização global dos adaptadores de canais
_active_channels = []

@app.on_event("startup")
async def startup_event():
    settings = get_settings()
    if settings.telegram_bot_token:
        from calosum.adapters.channel_telegram import TelegramChannelAdapter
        import asyncio
        
        logger.info("Inicializando Telegram Channel Adapter...")
        telegram_adapter = TelegramChannelAdapter(settings.telegram_bot_token)
        
        async def on_telegram_message(user_turn: UserTurn):
            agent = get_agent()
            try:
                # Dispara processamento em background para não bloquear o polling do Telegram
                asyncio.create_task(_process_and_reply(agent, telegram_adapter, user_turn))
            except Exception as e:
                logger.error(f"Erro engatilhando mensagem do Telegram: {e}")

        async def _process_and_reply(agent, adapter, user_turn):
            try:
                result = await agent.aprocess_turn(user_turn)
                if hasattr(result, "selected_result"):
                    response_text = result.selected_result.left_result.response_text
                else:
                    response_text = result.left_result.response_text
                await adapter.send(user_turn.session_id, response_text)
            except Exception as e:
                logger.error(f"Erro processando mensagem do Telegram: {e}")
                await adapter.send(user_turn.session_id, "Ocorreu um erro ao processar sua mensagem.")

        _active_channels.append(telegram_adapter)
        # O listen() agora é agendado corretamente no event loop
        asyncio.create_task(telegram_adapter.listen(on_telegram_message))

@app.on_event("shutdown")
async def shutdown_event():
    for adapter in _active_channels:
        if hasattr(adapter, 'app'):
            logger.info("Encerrando Telegram Channel Adapter...")
            await adapter.app.updater.stop() # type: ignore
            await adapter.app.stop()
            await adapter.app.shutdown()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check() -> JSONResponse:
    """Basic health check to ensure the API is running."""
    return JSONResponse({"status": "ok"})


@app.get("/v1/system/info")
async def system_info() -> JSONResponse:
    """
    Retorna o snapshot das capacidades e configuracoes da arquitetura atual.
    (Baseline de Capability State - Sprint 0)
    """
    try:
        builder = get_builder()
        # O agent ja pode ter sido instanciado, forçamos um descritivo
        # baseado no builder. O describe() inclui o CapabilityDescriptor
        info = builder.describe()
        return JSONResponse({"status": "ok", "info": info})
    except Exception as e:
        logger.error("Error retrieving system info", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get("/v1/system/architecture")
async def system_architecture() -> JSONResponse:
    """
    Retorna o Self-Model da arquitetura (componentes, conexões, health).
    """
    try:
        agent = get_agent()
        return JSONResponse({"status": "ok", "architecture": to_primitive(agent.self_model)})
    except Exception as e:
        logger.error("Error retrieving system architecture", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get("/v1/system/capabilities")
async def system_capabilities() -> JSONResponse:
    """
    Retorna as capacidades ativas extraídas do Self-Model.
    """
    try:
        agent = get_agent()
        return JSONResponse({"status": "ok", "capabilities": to_primitive(agent.self_model.capabilities)})
    except Exception as e:
        logger.error("Error retrieving system capabilities", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get("/v1/system/state")
async def system_state(session_id: str | None = None) -> JSONResponse:
    """
    Retorna o workspace cognitivo do último turno da sessão fornecida.
    Se não houver session_id, retorna o da sessão padrão (api-session) ou o mais recente disponível.
    """
    try:
        agent = get_agent()
        target_session = session_id or "api-session"
        workspace = agent.last_workspace_by_session.get(target_session)
        
        if not workspace:
            # Fallback se a sessao específica não tiver, pega a última registrada
            if agent.last_workspace_by_session:
                last_key = list(agent.last_workspace_by_session.keys())[-1]
                workspace = agent.last_workspace_by_session[last_key]
                
        if not workspace:
            return JSONResponse({"status": "error", "error": "No cognitive workspace found"}, status_code=404)
            
        return JSONResponse({"status": "ok", "state": to_primitive(workspace)})
    except Exception as e:
        logger.error("Error retrieving system state", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get("/ready")
async def readiness_check() -> JSONResponse:
    """
    Readiness check to verify that dependencies (LLM, Qdrant) are reachable.
    In a real scenario, you'd ping the actual services. Here we just ensure 
    the builder can instantiate the agent without crashing.
    """
    try:
        get_agent()
        return JSONResponse({"status": "ready"})
    except Exception as e:
        logger.error("Readiness check failed", exc_info=True)
        return JSONResponse({"status": "unready", "error": str(e)}, status_code=503)

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
    agent = get_agent()
    
    try:
        result = await agent.aprocess_turn(user_turn)
        return JSONResponse({"status": "ok", "result": to_primitive(result)})
    except Exception as e:
        logger.error("Error processing turn", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get("/v1/telemetry/dashboard/{session_id}")
async def get_dashboard(session_id: str) -> JSONResponse:
    """
    Retorna o dashboard cognitivo da sessão contendo os eventos 
    separados por 'felt', 'thought' e 'decision'.
    """
    try:
        agent = get_agent()
        dashboard = agent.cognitive_dashboard(session_id)
        return JSONResponse({"status": "ok", "dashboard": dashboard})
    except Exception as e:
        logger.error("Error retrieving dashboard", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get("/v1/telemetry/dashboard")
async def get_global_dashboard() -> JSONResponse:
    """
    Retorna o dashboard cognitivo global contendo os eventos de todas as sessões.
    """
    try:
        agent = get_agent()
        dashboard = agent.cognitive_dashboard(None)
        return JSONResponse({"status": "ok", "dashboard": dashboard})
    except Exception as e:
        logger.error("Error retrieving dashboard", exc_info=True)
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get("/v1/chat/sse")
async def chat_sse(request: Request, text: str, session_id: str = "api-session"):
    """
    Endpoint SSE simplificado.
    Em um cenário real de streaming de tokens, o adapter do LLM 
    deveria usar um AsyncGenerator. Aqui simulamos as etapas da orquestração.
    """
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    async def event_generator():
        yield {
            "event": "status",
            "data": "processing"
        }
        
        agent = get_agent()
        user_turn = UserTurn(session_id=session_id, user_text=text, signals=[])
        
        try:
            # Processa o turno de forma assíncrona
            result = await agent.aprocess_turn(user_turn)
            
            if hasattr(result, "selected_result"):
                turn_result = result.selected_result
            else:
                turn_result = result
            
            # Envia o resultado principal (raciocínio/resposta)
            yield {
                "event": "reasoning",
                "data": turn_result.left_result.response_text or "..."
            }
            
            # Envia as ações executadas
            for exec_res in turn_result.execution_results:
                yield {
                    "event": "action",
                    "data": f"[{exec_res.action_type}] {exec_res.status}"
                }
                
            yield {
                "event": "done",
                "data": "success"
            }
        except Exception as e:
            logger.error("SSE Error", exc_info=True)
            yield {
                "event": "error",
                "data": str(e)
            }
            
    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=get_settings().api_port)
