from __future__ import annotations

import logging
from typing import Awaitable, Callable

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from calosum.shared.types import UserTurn

logger = logging.getLogger(__name__)

class TelegramChannelAdapter:
    """
    Adaptador de mensageria para o Telegram.
    
    Implementa ChannelPort para abstrair a comunicação com o mundo externo.
    Utiliza polling para receber mensagens.
    """

    def __init__(
        self,
        token: str,
        *,
        dm_policy: str = "open",
        allowlist_ids: list[str] | None = None,
    ) -> None:
        self.token = token
        self.app = ApplicationBuilder().token(self.token).build()
        self._on_message_callback: Callable[[UserTurn], Awaitable[None]] | None = None
        self.dm_policy = (dm_policy or "open").strip().lower()
        self.allowlist_ids = set(allowlist_ids or [])

    async def listen(self, on_message: Callable[[UserTurn], Awaitable[None]]) -> None:
        self._on_message_callback = on_message
        
        # Registra o handler para mensagens de texto
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_telegram_message))
        
        logger.info("Iniciando escuta no Telegram (polling)...")
        import asyncio
        
        async def _start_polling():
            try:
                await self.app.initialize()
                await self.app.start()
                # Start polling requires an active event loop but can't block the main FastAPI thread
                # Se usarmos apenas start_polling, ele trava. Se usarmos drop_pending_updates ele pode falhar por timeout silencioso.
                # A abordagem correta no python-telegram-bot v20+ rodando junto com fastapi é rodar o run_polling
                # mas como não queremos bloquear, usamos updater.start_polling com tratamento de erro
                await self.app.updater.start_polling(drop_pending_updates=True) # type: ignore
                logger.info("Telegram polling rodando em background com sucesso.")
            except Exception as e:
                logger.error(f"Erro na thread de inicialização do Telegram: {e}")
                
        # Inicia a rotina sem bloquear a inicialização do FastAPI e lidando com Timeouts de rede (ex: docker bridge)
        asyncio.create_task(_start_polling())

    async def send(self, session_id: str, text: str) -> None:
        try:
            chat_id = int(session_id)
            await self.app.bot.send_message(chat_id=chat_id, text=text)
        except Exception as e:
            logger.error(f"Erro ao enviar mensagem para Telegram (session {session_id}): {e}")

    async def _handle_telegram_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._on_message_callback or not update.message or not update.message.text:
            return

        chat_id = str(update.message.chat_id)
        sender_id = str(update.effective_user.id) if update.effective_user else chat_id
        text = update.message.text

        if not self._is_sender_allowed(sender_id):
            logger.warning(
                "Telegram message rejected by dm_policy policy=%s sender_id=%s chat_id=%s",
                self.dm_policy,
                sender_id,
                chat_id,
            )
            return

        user_turn = UserTurn(
            session_id=chat_id,
            user_text=text,
            signals=[], # No futuro podemos extrair reações ou áudios aqui
        )

        try:
            await self._on_message_callback(user_turn)
        except Exception as e:
            logger.error(f"Erro no callback de processamento do Telegram: {e}")
            await self.send(chat_id, "Desculpe, ocorreu um erro interno ao processar sua mensagem.")

    def _is_sender_allowed(self, sender_id: str) -> bool:
        if self.dm_policy == "allowlist":
            return sender_id in self.allowlist_ids
        return True
