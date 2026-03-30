import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from calosum.bootstrap.context import (
    get_settings,
    get_builder,
    get_agent,
    _run_in_session_lane,
    _session_locks,
)
from calosum.bootstrap.settings import (
    InfrastructureSettings,
    should_enable_local_persistence_defaults,
    with_local_persistence_defaults,
)
from calosum.shared.types import UserTurn

logger = logging.getLogger(__name__)

# Inicialização global dos adaptadores de canais
_active_channels = []


def resolve_api_settings(environ: dict[str, str] | None = None) -> InfrastructureSettings:
    env = environ or os.environ
    settings = InfrastructureSettings.from_sources(environ=env)
    if should_enable_local_persistence_defaults(settings, environ=env):
        return with_local_persistence_defaults(settings)
    return settings

@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if settings.telegram_bot_token:
        try:
            from calosum.adapters.channel_telegram import TelegramChannelAdapter
            logger.info("Inicializando Telegram Channel Adapter...")
            telegram_adapter = TelegramChannelAdapter(
                settings.telegram_bot_token,
                dm_policy=settings.telegram_dm_policy,
                allowlist_ids=settings.telegram_allowlist_ids,
            )
        except Exception as exc:
            logger.warning("Telegram channel disabled: %s", exc)
            telegram_adapter = None

        if telegram_adapter is not None:
            async def on_telegram_message(user_turn: UserTurn):
                agent = get_agent()
                try:
                    # Dispara processamento em background para não bloquear o polling do Telegram
                    asyncio.create_task(_process_and_reply(agent, telegram_adapter, user_turn))
                except Exception as e:
                    logger.error(f"Erro engatilhando mensagem do Telegram: {e}")

            async def _process_and_reply(agent, adapter, user_turn):
                try:
                    result = await _run_in_session_lane(
                        user_turn.session_id,
                        lambda: agent.aprocess_turn(user_turn),
                    )
                    if hasattr(result, "selected_result"):
                        response_text = result.selected_result.left_result.response_text
                    else:
                        response_text = result.left_result.response_text
                    await adapter.send(user_turn.session_id, response_text)
                except Exception as e:
                    logger.error(f"Erro processando mensagem do Telegram: {e}")
                    await adapter.send(user_turn.session_id, "Ocorreu um erro ao processar sua mensagem.")

            _active_channels.append(telegram_adapter)
            asyncio.create_task(telegram_adapter.listen(on_telegram_message))

    try:
        yield
    finally:
        for adapter in _active_channels:
            if hasattr(adapter, "app"):
                logger.info("Encerrando Telegram Channel Adapter...")
                await adapter.app.updater.stop()  # type: ignore
                await adapter.app.stop()
                await adapter.app.shutdown()
        _active_channels.clear()
        _session_locks.clear()


app = FastAPI(
    title="Calosum API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from calosum.bootstrap.routers import system, chat, telemetry
app.include_router(system.router)
app.include_router(chat.router)
app.include_router(telemetry.router)

@app.get("/health")
async def health_check() -> JSONResponse:
    """Basic health check to ensure the API is running."""
    return JSONResponse({"status": "ok"})

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=get_settings().api_port)
