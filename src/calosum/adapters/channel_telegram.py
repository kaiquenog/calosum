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

    def __init__(self, token: str) -> None:
        self.token = token
        self.app = ApplicationBuilder().token(self.token).build()
        self._on_message_callback: Callable[[UserTurn], Awaitable[None]] | None = None

    async def listen(self, on_message: Callable[[UserTurn], Awaitable[None]]) -> None:
        self._on_message_callback = on_message
        
        # Registra o handler para mensagens de texto
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_telegram_message))
        
        logger.info("Iniciando escuta no Telegram (polling)...")
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling() # type: ignore

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
        text = update.message.text

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
