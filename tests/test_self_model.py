from __future__ import annotations

import unittest
from calosum.domain.orchestrator import CalosumAgent
from calosum.domain.self_model import build_self_model
from calosum.shared.types import (
    CapabilityDescriptor,
    CognitiveArchitectureMap,
    ComponentHealth,
    ModelDescriptor,
)


class SelfModelTests(unittest.TestCase):
    def test_build_self_model_generates_correct_architecture_map(self) -> None:
        agent = CalosumAgent()
        self_model = build_self_model(agent)

        self.assertIsInstance(self_model, CognitiveArchitectureMap)
        self.assertTrue(len(self_model.components) > 0)
        
        roles = [c.role for c in self_model.components]
        self.assertIn("perception", roles)
        self.assertIn("reasoning", roles)
        self.assertIn("bridge", roles)
        self.assertIn("memory", roles)
        self.assertIn("execution", roles)

        self.assertTrue(len(self_model.connections) > 0)
        self.assertIn("bridge.target_temperature", self_model.adaptation_surface.tunable_parameters)
        self.assertIn("PARAMETER", self_model.adaptation_surface.supported_directives)

    def test_calosum_agent_persists_self_model_on_boot(self) -> None:
        agent = CalosumAgent()
        self.assertIsNotNone(agent.self_model)
        self.assertIsInstance(agent.self_model, CognitiveArchitectureMap)
        
        # Verify that capabilities snapshot is propagated if available
        self.assertIsNotNone(agent.self_model.capabilities)

    def test_self_model_propagates_component_health_from_capability_snapshot(self) -> None:
        snapshot = CapabilityDescriptor(
            right_hemisphere=ModelDescriptor(
                provider="local",
                model_name="jepa",
                backend="active_inference_heuristic_fallback",
                health=ComponentHealth.DEGRADED,
            ),
            left_hemisphere=ModelDescriptor(
                provider="openai",
                model_name="gpt-4.1-mini",
                backend="openai_responses_adapter",
                health=ComponentHealth.HEALTHY,
            ),
            embeddings=None,
            knowledge_graph=ModelDescriptor(
                provider="local",
                model_name="nanorag",
                backend="in_memory_graph_fallback",
                health=ComponentHealth.DEGRADED,
            ),
            tools=[],
            health=ComponentHealth.DEGRADED,
        )
        agent = CalosumAgent(capability_snapshot=snapshot)

        self_model = build_self_model(agent)
        health_by_component = {
            component.component_id: component.health
            for component in self_model.components
        }

        self.assertEqual(health_by_component["right_hemisphere"], ComponentHealth.DEGRADED)
        self.assertEqual(health_by_component["memory_system"], ComponentHealth.DEGRADED)
        self.assertEqual(self_model.capabilities.health, ComponentHealth.DEGRADED)


if __name__ == "__main__":
    unittest.main()
