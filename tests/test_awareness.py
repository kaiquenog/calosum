from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from calosum.domain.evolution import JsonlEvolutionArchive
from calosum.domain.introspection import IntrospectionEngine
from calosum.domain.orchestrator import CalosumAgent
from calosum.adapters.active_inference import ActiveInferenceRightHemisphereAdapter
from calosum.domain.right_hemisphere import RightHemisphereJEPA
from calosum.shared.types import DirectiveType, EvolutionDirective


class AwarenessTests(unittest.TestCase):
    def test_introspection_engine_reports_failure_types_backlog_and_surprise_trend(self) -> None:
        dashboard = {
            "felt": [
                {"surprise_score": 0.2},
                {"surprise_score": 0.35},
                {"surprise_score": 0.65},
            ],
            "thought": [
                {"active_variant": None},
                {"active_variant": None},
                {"active_variant": None},
            ],
            "decision": [
                {"tool_success_rate": 0.5, "runtime_retry_count": 2},
                {"tool_success_rate": 0.4, "runtime_retry_count": 2},
                {"tool_success_rate": 0.6, "runtime_retry_count": 1},
            ],
            "execution": [
                {
                    "results": [
                        {"status": "rejected", "output": {"error_type": "runtime_crash"}},
                        {"status": "needs_approval", "output": {"missing_permissions": ["shell"]}},
                    ]
                }
            ],
        }

        diagnostic = IntrospectionEngine().analyze(
            "session-1",
            dashboard,
            pending_directive_count=2,
        )

        self.assertEqual(diagnostic.failure_types["runtime_crash"], 1)
        self.assertEqual(diagnostic.failure_types["approval_required"], 1)
        self.assertEqual(diagnostic.pending_approval_backlog, 1)
        self.assertEqual(diagnostic.pending_directive_count, 2)
        self.assertGreater(diagnostic.surprise_trend, 0.0)
        self.assertIsNone(diagnostic.dominant_variant)
        self.assertGreater(len(diagnostic.bottlenecks), 0)

    def test_evolution_archive_reload_only_pending_directives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = JsonlEvolutionArchive(Path(temp_dir) / "evolution" / "archive.jsonl")
            pending = EvolutionDirective(
                directive_id="directive-pending",
                directive_type=DirectiveType.PROMPT,
                target_component="left_hemisphere",
                proposed_change={"instruction": "ask for clarification"},
                reasoning="Need safer clarification",
            )
            applied = EvolutionDirective(
                directive_id="directive-applied",
                directive_type=DirectiveType.PARAMETER,
                target_component="orchestrator",
                proposed_change={"max_runtime_retries": 3},
                reasoning="Raise retries",
                status="applied",
            )

            archive.record_directive(pending, event="queued")
            archive.record_directive(applied, event="auto_applied")

            pending_directives = archive.load_pending_directives()

            self.assertEqual(len(pending_directives), 1)
            self.assertEqual(pending_directives[0].directive_id, "directive-pending")

    def test_manual_topology_directive_is_blocked_by_guardrail(self) -> None:
        agent = CalosumAgent()
        directive = EvolutionDirective(
            directive_id="directive-topology",
            directive_type=DirectiveType.TOPOLOGY,
            target_component="right_hemisphere",
            proposed_change={"model": "vjepa2"},
            reasoning="Attempt topology switch",
        )
        agent.pending_directives.append(directive)

        applied = agent.apply_pending_directive("directive-topology")

        self.assertIsNotNone(applied)
        assert applied is not None
        self.assertEqual(applied.status, "rejected_guardrail_topology_locked")

    def test_right_hemisphere_parameter_directive_is_clamped_and_audited(self) -> None:
        agent = CalosumAgent(
            right_hemisphere=ActiveInferenceRightHemisphereAdapter(RightHemisphereJEPA()),
        )
        wrapper_config = getattr(agent.right_hemisphere, "config")
        base_config = getattr(getattr(agent.right_hemisphere, "base_adapter"), "config")
        initial_step = base_config.salience_max_step
        initial_alpha = base_config.salience_smoothing_alpha
        initial_novelty = wrapper_config.novelty_weight

        directive = EvolutionDirective(
            directive_id="directive-right-params",
            directive_type=DirectiveType.PARAMETER,
            target_component="right_hemisphere",
            proposed_change={
                "salience_max_step": 0.9,
                "salience_smoothing_alpha": 0.99,
                "salience_window_size": 99,
                "novelty_weight": 0.99,
                "model_name": "forbidden-topology-like-change",
            },
            reasoning="Tune right hemisphere safely",
        )

        agent._apply_directive(directive)

        self.assertEqual(directive.status, "applied")
        self.assertIn("_audit", directive.proposed_change)
        audit = directive.proposed_change["_audit"]
        self.assertIn("model_name", audit["rejected"])
        self.assertEqual(audit["rejected"]["model_name"], "param_not_allowed")

        # Guardrail: bounded and small deltas only.
        self.assertLessEqual(base_config.salience_max_step - initial_step, 0.1 + 1e-9)
        self.assertLessEqual(base_config.salience_smoothing_alpha - initial_alpha, 0.2 + 1e-9)
        self.assertLessEqual(wrapper_config.novelty_weight - initial_novelty, 0.2 + 1e-9)
        self.assertLessEqual(base_config.salience_window_size, 12)

    def test_evolution_proposer_prefers_parameter_tuning_for_surprise_trend(self) -> None:
        diagnostic = IntrospectionEngine().analyze(
            "session-2",
            {
                "felt": [{"surprise_score": 0.1}, {"surprise_score": 0.9}],
                "thought": [{"active_variant": "default"}, {"active_variant": "default"}],
                "decision": [{"tool_success_rate": 1.0, "runtime_retry_count": 0}] * 2,
                "execution": [{"results": []}],
            },
            pending_directive_count=0,
        )
        from calosum.domain.evolution import EvolutionProposer

        directives = EvolutionProposer().propose(diagnostic)
        right_param = [
            item for item in directives
            if item.directive_type == DirectiveType.PARAMETER and item.target_component == "right_hemisphere"
        ]
        self.assertTrue(right_param)


if __name__ == "__main__":
    unittest.main()
