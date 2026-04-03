from __future__ import annotations

import asyncio
import unittest

from calosum.domain.agent.multiagent import MultiAgentWorkflow


class MultiAgentWorkflowTests(unittest.TestCase):
    def test_multiagent_workflow_executes_real_event_chain(self) -> None:
        result = asyncio.run(MultiAgentWorkflow().aorchestrate("organize rollout", timeout_seconds=2.0))
        self.assertTrue(result["is_valid"])
        self.assertIn("executed_steps", result)


if __name__ == "__main__":
    unittest.main()
