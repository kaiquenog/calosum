from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

@dataclass
class CognitiveEvent:
    event_type: str
    payload: Any
    turn_id: str

class InternalEventBus:
    def __init__(self):
        self.subscribers: dict[str, list[Callable[[CognitiveEvent], Awaitable[None]]]] = {}
        self.queue: asyncio.Queue[CognitiveEvent] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def start(self):
        if self._worker_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._worker_task = loop.create_task(self._process_events())
            except RuntimeError:
                pass # No loop running yet

    async def _process_events(self):
        while True:
            event = await self.queue.get()
            try:
                handlers = self.subscribers.get(event.event_type, [])
                for handler in handlers:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Error handling event {event.event_type}: {e}", exc_info=True)
            finally:
                self.queue.task_done()

    def subscribe(self, event_type: str, handler: Callable[[CognitiveEvent], Awaitable[None]]):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    async def publish(self, event: CognitiveEvent):
        # Fire and forget into the queue
        await self.queue.put(event)
        
        # Ensure worker is started
        if self._worker_task is None:
            self.start()