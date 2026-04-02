import os
import asyncio
from functools import lru_cache
from typing import Any
from calosum.bootstrap.wiring.factory import CalosumAgentBuilder
from calosum.bootstrap.infrastructure.settings import (
    InfrastructureSettings,
    should_enable_local_persistence_defaults,
    with_local_persistence_defaults,
)

_session_locks: dict[str, asyncio.Lock] = {}

def _session_lock(session_id: str) -> asyncio.Lock:
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[session_id] = lock
    return lock

async def _run_in_session_lane(session_id: str, operation):
    async with _session_lock(session_id):
        return await operation()

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

def get_agent_and_builder():
    return get_agent(), get_builder()

@lru_cache(maxsize=1)
def get_agent():
    return get_builder().build(agent_accessor=get_agent_and_builder)
