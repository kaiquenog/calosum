from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from calosum.domain.evolution import JsonlEvolutionArchive
from calosum.domain.introspection import IntrospectionEngine
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
                {"active_variant": "default"},
                {"active_variant": "default"},
                {"active_variant": "default"},
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
        self.assertEqual(diagnostic.dominant_variant, "default")
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


if __name__ == "__main__":
    unittest.main()
