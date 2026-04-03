from __future__ import annotations
from dataclasses import asdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from calosum.domain.agent.orchestrator import CalosumAgent
    from calosum.bootstrap.wiring.factory import CalosumAgentFactory

from calosum.shared.models.types import (
    CapabilityDescriptor,
    ComponentHealth,
    ModelDescriptor,
    RoutingPolicy,
)
from calosum.bootstrap.wiring.operational_budget import operational_budget_snapshot


class CalosumSystemIntrospector:
    """
    Serviço de domínio para introspecção de capacidades e estado do sistema.
    """

    @staticmethod
    def build_capability_snapshot(factory: CalosumAgentFactory, action_runtime: Any | None = None) -> CapabilityDescriptor:
        right_health = factory._right_hemisphere_health()
        budgets = operational_budget_snapshot(
            right_backend=factory._right_hemisphere_backend_name(),
            left_backend=factory._left_hemisphere_backend_name(),
            bridge_backend="cross_attention" if (factory.settings.bridge_backend or "").strip().lower() == "cross_attention" else "heuristic_projection",
            requested_right_backend=factory.settings.right_hemisphere_backend or factory._right_hemisphere_backend_name(),
        )
        right_model = ModelDescriptor(
            provider="huggingface_local" if "huggingface" in factory._right_hemisphere_backend_name() else "local",
            model_name=factory._right_hemisphere_model_name(),
            backend=factory._right_hemisphere_backend_name(),
            health=right_health,
        )
        left_model = ModelDescriptor(
            provider=factory.settings.left_hemisphere_provider or "auto",
            model_name=factory._reason_model_name(),
            backend=factory._left_hemisphere_backend_name(),
            health=ComponentHealth.HEALTHY,
        )
        embedding_model = None
        if factory._embedding_backend_name():
            embedding_model = ModelDescriptor(
                provider=factory._derived_embedding_provider() or "auto",
                model_name=factory._derived_embedding_model() or "auto",
                backend=factory._embedding_backend_name() or "auto",
                health=ComponentHealth.HEALTHY,
            )
        kg_model = ModelDescriptor(
            provider="local", model_name="nanorag",
            backend=factory._knowledge_graph_backend_name(),
            health=factory._knowledge_graph_health(),
        )
        tools = action_runtime.get_registered_tools() if action_runtime else []
        routing_policy = RoutingPolicy(
            perception_model=factory.settings.perception_model or right_model.model_name,
            reason_model=factory.settings.reason_model or left_model.model_name,
            reflection_model=factory.settings.reflection_model or left_model.model_name,
            verifier_model=factory.settings.verifier_model,
        )
        healths = {right_health, left_model.health, kg_model.health}
        overall_health = ComponentHealth.HEALTHY
        if ComponentHealth.UNAVAILABLE in healths: overall_health = ComponentHealth.UNAVAILABLE
        elif ComponentHealth.DEGRADED in healths: overall_health = ComponentHealth.DEGRADED

        return CapabilityDescriptor(
            right_hemisphere=right_model, left_hemisphere=left_model,
            embeddings=embedding_model, knowledge_graph=kg_model,
            tools=tools, routing_policy=routing_policy, health=overall_health,
            operational_constraints={
                "budgets": budgets,
                "turn_contract": {
                    "single_candidate": "AgentTurnResult",
                    "multi_candidate": "GroupTurnResult",
                    "selection_accessor": "selected_result",
                    "compatibility_method": "as_agent_turn_result",
                },
            },
        )

    @staticmethod
    def describe(factory: CalosumAgentFactory, agent: Any | None = None) -> dict[str, Any]:
        action_runtime = getattr(agent, "action_runtime", None) if agent else None
        snapshot = agent.capability_snapshot if agent and hasattr(agent, "capability_snapshot") else CalosumSystemIntrospector.build_capability_snapshot(factory, action_runtime)
        s = factory.settings
        right_runtime = CalosumSystemIntrospector._right_runtime_state(agent, factory)
        return {
            "pattern": "v3_dual_hemisphere_factory", "profile": s.profile.value,
            "capabilities": asdict(snapshot), "memory_backend": factory._memory_backend_name(),
            "embedding_backend": factory._embedding_backend_name(),
            "telemetry_backend": factory._telemetry_backend_name(),
            "right_hemisphere_backend": factory._right_hemisphere_backend_name(),
            "left_hemisphere_backend": factory._left_hemisphere_backend_name(),
            "memory_dir": str(s.memory_dir) if s.memory_dir else None,
            "vector_db_url": s.vector_db_url,
            "bridge_state_dir": str(s.bridge_state_dir) if s.bridge_state_dir else None,
            "evolution_path": str(s.evolution_archive_path) if s.evolution_archive_path else None,
            "awareness_turns": s.awareness_interval_turns,
            "otel_endpoint": s.otel_collector_endpoint,
            "left_hemisphere_model": factory._reason_model_name(),
            "left_hemisphere_provider": s.left_hemisphere_provider,
            "left_hemisphere_failover": bool(s.left_hemisphere_fallback_endpoint),
            "knowledge_graph_backend": factory._knowledge_graph_backend_name(),
            "routing_resolution": factory._routing_resolution(snapshot),
            "operational_budgets": snapshot.operational_constraints.get("budgets", {}),
            "turn_contract": snapshot.operational_constraints.get("turn_contract", {}),
            "degradations": {"right_hemisphere": right_runtime},
        }

    @staticmethod
    def _right_runtime_state(agent: Any | None, factory: CalosumAgentFactory) -> dict[str, Any]:
        adapter = getattr(agent, "right_hemisphere", None) if agent is not None else None
        provider = getattr(adapter, "provider", adapter)
        checkpoint_loaded = getattr(provider, "is_available", None)
        if checkpoint_loaded is None and hasattr(provider, "_health"):
            checkpoint_loaded = getattr(provider, "_health", None) == ComponentHealth.HEALTHY
        return {
            "health": factory._right_hemisphere_health(),
            "degraded_reason": getattr(provider, "degraded_reason", None),
            "checkpoint_loaded": checkpoint_loaded,
            "multimodal_capable": factory._right_hemisphere_backend_name() == "vljepa_local",
            "contract_version": getattr(provider, "CONTRACT_VERSION", None),
            "budget": getattr(provider, "operational_budget", None),
        }
