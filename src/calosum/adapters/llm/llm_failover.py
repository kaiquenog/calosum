from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from calosum.shared.utils.async_utils import maybe_await, run_sync
from calosum.shared.models.ports import ActionPlannerPort
from calosum.shared.models.types import (
    ActionExecutionResult,
    PerceptionSummary,
    ActionPlannerResult,
    MemoryContext,
    TypedLambdaProgram,
    UserTurn,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResilientLeftHemisphereConfig:
    cooldown_seconds: float = 30.0


class ResilientLeftHemisphereAdapter:
    def __init__(
        self,
        providers: list[ActionPlannerPort],
        config: ResilientLeftHemisphereConfig | None = None,
    ) -> None:
        if not providers:
            raise ValueError("ResilientLeftHemisphereAdapter requires at least one provider")
        self.providers = providers
        self.config = config or ResilientLeftHemisphereConfig()
        self._cooldowns: dict[str, float] = {}
        self._provider_names = {
            id(provider): self._derive_provider_name(provider, index)
            for index, provider in enumerate(self.providers)
        }

    def reason(
        self,
        user_turn: UserTurn,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
    ) -> ActionPlannerResult:
        return run_sync(
            self.areason(
                user_turn=user_turn,
                bridge_packet=bridge_packet,
                memory_context=memory_context,
                runtime_feedback=runtime_feedback,
                attempt=attempt,
            )
        )

    async def areason(
        self,
        user_turn: UserTurn,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
    ) -> ActionPlannerResult:
        return await self._invoke_with_failover(
            "areason",
            "reason",
            user_turn,
            bridge_packet,
            memory_context,
            runtime_feedback,
            attempt,
        )

    def repair(
        self,
        user_turn: UserTurn,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        previous_result: ActionPlannerResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
    ) -> ActionPlannerResult:
        return run_sync(
            self.arepair(
                user_turn=user_turn,
                bridge_packet=bridge_packet,
                memory_context=memory_context,
                previous_result=previous_result,
                rejected_results=rejected_results,
                attempt=attempt,
                critique_feedback=critique_feedback,
            )
        )

    async def arepair(
        self,
        user_turn: UserTurn,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        previous_result: ActionPlannerResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
    ) -> ActionPlannerResult:
        return await self._invoke_with_failover(
            "arepair",
            "repair",
            user_turn,
            bridge_packet,
            memory_context,
            previous_result,
            rejected_results,
            attempt,
            critique_feedback,
        )

    async def _invoke_with_failover(
        self,
        async_method_name: str,
        sync_method_name: str,
        *args: object,
    ) -> ActionPlannerResult:
        attempted: list[str] = []
        last_result: ActionPlannerResult | None = None
        last_error: str | None = None

        for provider in self._ordered_providers():
            provider_name = self._provider_name(provider)
            attempted.append(provider_name)
            try:
                method = getattr(provider, async_method_name, None)
                if method is None:
                    method = getattr(provider, sync_method_name)
                result = await maybe_await(method(*args))
            except Exception as exc:  # pragma: no cover - exercised by integration path
                last_error = repr(exc)
                self._mark_failure(provider_name)
                logger.warning("Left hemisphere provider %s failed: %s", provider_name, exc)
                continue

            reason = self._failover_reason(result)
            if reason is None:
                self._clear_failure(provider_name)
                return self._annotate_result(result, provider_name, attempted, exhausted=False)

            last_error = reason
            last_result = result
            self._mark_failure(provider_name)
            logger.warning(
                "Left hemisphere provider %s returned unusable result, failing over: %s",
                provider_name,
                reason,
            )

        if last_result is not None:
            return self._fallback_result(last_error or "all providers returned unusable results", attempted)

        return self._fallback_result(last_error or "all providers failed", attempted)

    def _ordered_providers(self) -> list[ActionPlannerPort]:
        now = time.time()
        healthy: list[ActionPlannerPort] = []
        cooling: list[ActionPlannerPort] = []
        for provider in self.providers:
            provider_name = self._provider_name(provider)
            if self._cooldowns.get(provider_name, 0.0) > now:
                cooling.append(provider)
            else:
                healthy.append(provider)
        return healthy or cooling or list(self.providers)

    def _mark_failure(self, provider_name: str) -> None:
        self._cooldowns[provider_name] = time.time() + self.config.cooldown_seconds

    def _clear_failure(self, provider_name: str) -> None:
        self._cooldowns.pop(provider_name, None)

    def _failover_reason(self, result: ActionPlannerResult) -> str | None:
        error = result.telemetry.get("error")
        if isinstance(error, str) and error:
            return error
        if not result.response_text.strip() and not result.actions:
            return "empty_response_and_no_actions"
        if any(
            isinstance(item, str) and item.lower().startswith("structured output parse failed")
            for item in result.reasoning_summary
        ):
            return "structured_output_parse_failed"
        return None

    def _annotate_result(
        self,
        result: ActionPlannerResult,
        selected_provider: str,
        attempted: list[str],
        *,
        exhausted: bool,
    ) -> ActionPlannerResult:
        result.telemetry["provider_name"] = selected_provider
        result.telemetry["failover_attempt_count"] = len(attempted)
        result.telemetry["failover_attempts"] = attempted
        result.telemetry["failover_exhausted"] = exhausted
        return result

    def _fallback_result(self, error: str, attempted: list[str]) -> ActionPlannerResult:
        return ActionPlannerResult(
            response_text="Desculpe, todos os provedores do hemisferio esquerdo falharam temporariamente.",
            lambda_program=TypedLambdaProgram("Fallback", "()", "None"),
            actions=[],
            reasoning_summary=[f"Erro de failover: {error}"],
            telemetry={
                "adapter": "ResilientLeftHemisphereAdapter",
                "error": error,
                "failover_attempt_count": len(attempted),
                "failover_attempts": attempted,
                "failover_exhausted": True,
            },
        )

    def _provider_name(self, provider: ActionPlannerPort) -> str:
        return self._provider_names[id(provider)]

    def _derive_provider_name(self, provider: ActionPlannerPort, index: int) -> str:
        config = getattr(provider, "config", None)
        model_name = getattr(config, "model_name", None)
        api_url = getattr(config, "api_url", None)
        if isinstance(api_url, str) and api_url:
            host = urlparse(api_url).hostname or "local"
        else:
            host = "local"
        if isinstance(model_name, str) and model_name:
            return f"{model_name}@{host}"
        return f"{provider.__class__.__name__}#{index + 1}"
