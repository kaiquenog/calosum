from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from calosum import CalosumAgentBuilder, InfrastructureSettings, UserTurn


class DualHemisphereE2ETests(unittest.TestCase):
    def test_dual_hemisphere_stack_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = InfrastructureSettings(
                right_hemisphere_backend="vjepa21",
                right_model_path=Path(temp_dir),
                left_hemisphere_backend="rlm",
                bridge_backend="cross_attention",
                gea_sharing_enabled=True,
                gea_experience_store_path=Path(temp_dir) / "gea.sqlite",
            ).with_profile_defaults()
            builder = CalosumAgentBuilder(settings)
            agent = builder.build()

            base = getattr(agent.right_hemisphere, "base_adapter", None)
            if base is not None and hasattr(base, "_embedder"):
                base._embedder = False

            result = agent.process_turn(
                UserTurn(
                    session_id="dual-e2e",
                    user_text="Estou ansioso e preciso reorganizar o projeto com passos claros.",
                )
            )

            self.assertGreaterEqual(result.right_state.surprise_score, 0.0)
            self.assertLessEqual(result.right_state.surprise_score, 1.0)
            self.assertTrue(result.left_result.actions)
            self.assertTrue(any(item.status in {"executed", "planned"} for item in result.execution_results))


if __name__ == "__main__":
    unittest.main()
