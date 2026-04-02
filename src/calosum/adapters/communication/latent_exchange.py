from __future__ import annotations

import logging
from calosum.shared.models.ports import LatentExchangePort
from calosum.domain.infrastructure.event_bus import InternalEventBus, CognitiveEvent

logger = logging.getLogger(__name__)

class InternalLatentExchangeAdapter(LatentExchangePort):
    """
    Implementa a troca de latentes (Soft Prompts) usando o barramento de eventos interno.
    Simula uma rede de agentes Calosum sincronizando seus modelos de mundo.
    """
    
    def __init__(self, event_bus: InternalEventBus):
        self.event_bus = event_bus
        self.peer_latents: list[list[float]] = []
        # Subscreve para ouvir transmissões de outros "peers" (ou de si mesmo em modo loopback)
        self.event_bus.subscribe("PeerLatentBroadcast", self._on_peer_latent)

    async def _on_peer_latent(self, event: CognitiveEvent):
        """Recebe um vetor latente de um peer e armazena no buffer de contexto social."""
        if not isinstance(event.payload, list):
            return
            
        self.peer_latents.append(event.payload)
        # Mantém apenas os 10 latentes mais recentes para evitar saturação de memória
        if len(self.peer_latents) > 10:
            self.peer_latents.pop(0)
            
        logger.debug(f"V3: Latente recebido de peer. Buffer size: {len(self.peer_latents)}")

    async def broadcast_latent(self, session_id: str, latent_vector: list[float]) -> None:
        """Transmite o estado latente atual para a rede de peers."""
        await self.event_bus.publish(
            CognitiveEvent("PeerLatentBroadcast", latent_vector, f"sync-{session_id}")
        )

    async def get_peer_latents(self, session_id: str) -> list[list[float]]:
        """Retorna o buffer de latentes recebidos de outros agentes."""
        return list(self.peer_latents)
