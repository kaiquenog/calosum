from __future__ import annotations

from calosum.shared.models.types import (
    AdaptationSurface,
    ArchitectureComponent,
    CapabilityDescriptor,
    CognitiveArchitectureMap,
    ComponentConnection,
    ComponentHealth,
)

def build_self_model(agent: "CalosumAgent") -> CognitiveArchitectureMap:
    from calosum.domain.agent.orchestrator import CalosumAgent
    
    capabilities = agent.capability_snapshot or CapabilityDescriptor(
        right_hemisphere=None,
        left_hemisphere=None,
        embeddings=None,
        knowledge_graph=None,
        tools=[],
        health=ComponentHealth.UNAVAILABLE,
    )

    right_health = (
        capabilities.right_hemisphere.health
        if capabilities.right_hemisphere is not None
        else ComponentHealth.UNAVAILABLE
    )
    left_health = (
        capabilities.left_hemisphere.health
        if capabilities.left_hemisphere is not None
        else ComponentHealth.UNAVAILABLE
    )
    memory_health = (
        capabilities.knowledge_graph.health
        if capabilities.knowledge_graph is not None
        else capabilities.health
    )
    runtime_health = _tool_runtime_health(capabilities)

    # Safely extract class names
    right_hemisphere_class = agent.right_hemisphere.__class__.__name__
    left_hemisphere_class = agent.left_hemisphere.__class__.__name__
    tokenizer_class = agent.tokenizer.__class__.__name__
    memory_system_class = agent.memory_system.__class__.__name__
    action_runtime_class = agent.action_runtime.__class__.__name__
    reflection_class = agent.reflection_controller.__class__.__name__
    verifier_class = agent.verifier.__class__.__name__ if agent.verifier else "None"
    
    components = [
        ArchitectureComponent(
            component_id="right_hemisphere",
            role="perception",
            adapter_class=right_hemisphere_class,
            health=right_health,
        ),
        ArchitectureComponent(
            component_id="tokenizer",
            role="bridge",
            adapter_class=tokenizer_class,
            health=ComponentHealth.HEALTHY,
        ),
        ArchitectureComponent(
            component_id="left_hemisphere",
            role="reasoning",
            adapter_class=left_hemisphere_class,
            health=left_health,
        ),
        ArchitectureComponent(
            component_id="memory_system",
            role="memory",
            adapter_class=memory_system_class,
            health=memory_health,
        ),
        ArchitectureComponent(
            component_id="action_runtime",
            role="execution",
            adapter_class=action_runtime_class,
            health=runtime_health,
        ),
        ArchitectureComponent(
            component_id="reflection_controller",
            role="metacognition",
            adapter_class=reflection_class,
            health=ComponentHealth.HEALTHY,
        ),
        ArchitectureComponent(
            component_id="verifier",
            role="verification",
            adapter_class=verifier_class,
            health=ComponentHealth.HEALTHY if agent.verifier else ComponentHealth.UNAVAILABLE,
        ),
    ]

    connections = [
        ComponentConnection("user_input", "right_hemisphere", "UserTurn"),
        ComponentConnection("user_input", "memory_system", "UserTurn -> MemoryContext"),
        ComponentConnection("right_hemisphere", "tokenizer", "InputPerceptionState"),
        ComponentConnection("tokenizer", "left_hemisphere", "PerceptionSummary"),
        ComponentConnection("memory_system", "left_hemisphere", "MemoryContext"),
        ComponentConnection("left_hemisphere", "action_runtime", "ActionPlannerResult"),
        ComponentConnection("action_runtime", "verifier", "ActionExecutionResult"),
        ComponentConnection("verifier", "left_hemisphere", "CritiqueVerdict"),
        ComponentConnection("left_hemisphere", "reflection_controller", "GroupTurnResult -> Reflection"),
    ]

    adaptation_surface = AdaptationSurface(
        tunable_parameters=[
            "bridge.target_temperature",
            "bridge.empathy_priority",
            "agent.surprise_threshold",
            "agent.max_runtime_retries",
            "agent.branching_budget.max_width",
            "agent.branching_budget.max_depth",
            "agent.awareness_interval_turns",
        ],
        supported_directives=["PARAMETER", "PROMPT", "TOPOLOGY", "ARCHITECTURE"],
    )

    return CognitiveArchitectureMap(
        components=components,
        connections=connections,
        adaptation_surface=adaptation_surface,
        capabilities=capabilities,
    )


def _tool_runtime_health(capabilities: CapabilityDescriptor) -> ComponentHealth:
    if not capabilities.tools:
        return ComponentHealth.HEALTHY

    statuses = {tool.health for tool in capabilities.tools}
    if ComponentHealth.UNAVAILABLE in statuses:
        return ComponentHealth.UNAVAILABLE
    if ComponentHealth.DEGRADED in statuses:
        return ComponentHealth.DEGRADED
    return ComponentHealth.HEALTHY
