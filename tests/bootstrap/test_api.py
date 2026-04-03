from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from calosum.bootstrap.entry import api as api_module
from calosum.shared.models.types import DirectiveType, EvolutionDirective


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        base = Path(self.temp_dir.name)
        env = {
            "CALOSUM_IGNORE_DOTENV": "true",
            "CALOSUM_VECTOR_QUANTIZATION": "none",
            "CALOSUM_INFRA_PROFILE": "persistent",
            "CALOSUM_MEMORY_DIR": str(base / "memory"),
            "CALOSUM_OTLP_JSONL": str(base / "telemetry" / "events.jsonl"),
            "CALOSUM_BRIDGE_STATE_DIR": str(base / "state"),
            "CALOSUM_EVOLUTION_ARCHIVE_PATH": str(base / "evolution" / "archive.jsonl"),
            "CALOSUM_PERCEPTION_MODEL": "jepa",
            "CALOSUM_LEFT_ENDPOINT": "http://127.0.0.1:9/v1/chat/completions",
            "CALOSUM_LEFT_MODEL": "test-left-model",
            "CALOSUM_REASON_MODEL": "test-reason-model",
            "CALOSUM_LEFT_PROVIDER": "openai_compatible_chat",
        }
        self.env_patcher = patch.dict(os.environ, env, clear=False)
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

        from unittest.mock import AsyncMock
        from calosum import ActionPlannerResult, TypedLambdaProgram, PrimitiveAction
        
        self.mock_reason_result = ActionPlannerResult(
            response_text="Mocked API response",
            lambda_program=TypedLambdaProgram("C->R", '{"plan": ["introspect_self"]}', "R"),
            actions=[
                PrimitiveAction(
                    action_type="introspect_self",
                    typed_signature="T->S",
                    payload={"query": "Who am I?"},
                    safety_invariants=["safe_output"]
                )
            ],
            reasoning_summary=["mocked"],
        )

        
        # Patch resolve_left_hemisphere to return a mock
        self.left_patcher = patch("calosum.bootstrap.wiring.backend_resolvers.resolve_left_hemisphere")
        self.mock_resolve = self.left_patcher.start()
        
        mock_left = AsyncMock()
        mock_left.areason.return_value = self.mock_reason_result
        mock_left.reason.return_value = self.mock_reason_result
        # For contract enforced wrapper
        mock_left.provider = AsyncMock()
        mock_left.provider.areason.return_value = self.mock_reason_result
        
        from calosum.adapters.infrastructure.contract_wrappers import ContractEnforcedLeftHemisphereAdapter
        self.mock_resolve.return_value = (ContractEnforcedLeftHemisphereAdapter(mock_left), "mock_adapter")
        
        self.addCleanup(self.left_patcher.stop)

        self._clear_caches()
        self.addCleanup(self._clear_caches)
        self.client = TestClient(api_module.app)

    def _clear_caches(self) -> None:
        api_module.get_settings.cache_clear()
        api_module.get_builder.cache_clear()
        api_module.get_agent.cache_clear()

    def test_health_check(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_ready_check_returns_component_health_details(self) -> None:
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ready")
        self.assertIn("health", payload)
        self.assertIn("components", payload)
        self.assertIn("right_hemisphere", payload["components"])
        self.assertIn("operational_budgets", payload)
        self.assertIn("turn_contract", payload)

    def test_system_info_returns_runtime_capabilities_and_routing_resolution(self) -> None:
        response = self.client.get("/v1/system/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("capabilities", data["info"])
        self.assertGreater(len(data["info"]["capabilities"]["tools"]), 0)
        self.assertIn("routing_resolution", data["info"])
        self.assertEqual(data["info"]["routing_resolution"]["perception"]["active"], "jepa")

    def test_system_architecture_returns_self_model(self) -> None:
        response = self.client.get("/v1/system/architecture")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("components", data["architecture"])
        self.assertIn("connections", data["architecture"])
        self.assertIn("adaptation_surface", data["architecture"])
        self.assertTrue(len(data["architecture"]["components"]) > 0)

    def test_system_capabilities_returns_capabilities_from_self_model(self) -> None:
        response = self.client.get("/v1/system/capabilities")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("right_hemisphere", data["capabilities"])
        self.assertIn("left_hemisphere", data["capabilities"])
        self.assertIn("tools", data["capabilities"])
        self.assertEqual(data["capabilities"]["routing_policy"]["reason_model"], "test-reason-model")

    def test_system_state_returns_workspace_not_found_initially(self) -> None:
        response = self.client.get("/v1/system/state", params={"session_id": "fresh-session"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIsNone(data["state"])

    def test_system_state_returns_workspace_after_turn(self) -> None:
        post_response = self.client.post("/v1/chat/completions", json={"text": "Hello workspace"})
        self.assertEqual(post_response.status_code, 200)

        response = self.client.get("/v1/system/state")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        workspace = data["state"]

        self.assertIn("task_frame", workspace)
        self.assertIn("right_notes", workspace)
        self.assertIn("left_notes", workspace)
        self.assertIn("self_model_ref", workspace)
        self.assertIn("capability_snapshot", workspace)
        self.assertIn("pending_questions", workspace)
        self.assertIn("session_briefing", workspace["task_frame"])
        self.assertEqual(workspace["task_frame"]["user_text"], "Hello workspace")

    def test_system_state_does_not_leak_other_session_workspace(self) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={"text": "Hello from session A", "session_id": "session-a"},
        )
        self.assertEqual(response.status_code, 200)

        missing = self.client.get("/v1/system/state", params={"session_id": "session-b"})
        self.assertEqual(missing.status_code, 200)
        self.assertEqual(missing.json()["status"], "ok")
        self.assertIsNone(missing.json()["state"])

    def test_workspace_carries_previous_runtime_feedback_across_turns(self) -> None:
        session_id = "runtime-feedback-session"
        first = self.client.post(
            "/v1/chat/completions",
            json={"text": "Primeiro turno para gerar execucao.", "session_id": session_id},
        )
        self.assertEqual(first.status_code, 200)

        second = self.client.post(
            "/v1/chat/completions",
            json={"text": "Segundo turno para reutilizar contexto operacional.", "session_id": session_id},
        )
        self.assertEqual(second.status_code, 200)

        state_response = self.client.get("/v1/system/state", params={"session_id": session_id})
        self.assertEqual(state_response.status_code, 200)
        state = state_response.json()["state"]

        self.assertIn("previous_runtime_feedback", state["task_frame"])
        self.assertGreater(len(state["task_frame"]["previous_runtime_feedback"]), 0)
        self.assertIn("runtime_feedback_bias", state["right_notes"])

    def test_dashboard_felt_exposes_right_hemisphere_runtime_telemetry_contract(self) -> None:
        session_id = "telemetry-contract-session"
        post_response = self.client.post(
            "/v1/chat/completions",
            json={"text": "Preciso de ajuda urgente", "session_id": session_id},
        )
        self.assertEqual(post_response.status_code, 200)

        dashboard_response = self.client.get(f"/v1/telemetry/dashboard/{session_id}")
        self.assertEqual(dashboard_response.status_code, 200)
        dashboard = dashboard_response.json()["dashboard"]
        self.assertGreater(len(dashboard["felt"]), 0)

        latest_felt = dashboard["felt"][-1]
        telemetry = latest_felt["telemetry"]
        self.assertIn("right_backend", telemetry)
        self.assertIn("right_model_name", telemetry)
        self.assertIn("right_mode", telemetry)
        self.assertIn("degraded_reason", telemetry)
        self.assertIn("runtime_feedback_bias", telemetry)
        self.assertEqual(telemetry["right_mode"], "literal_embedding")

    def test_system_awareness_generates_extended_diagnostic(self) -> None:
        self.client.post("/v1/chat/completions", json={"text": "Quero uma resposta com ação externa e aprovação."})

        response = self.client.get("/v1/system/awareness", params={"session_id": "api-session"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("diagnostic", data)
        self.assertIn("bottlenecks", data["diagnostic"])
        self.assertIn("failure_types", data["diagnostic"])
        self.assertIn("pending_approval_backlog", data["diagnostic"])
        self.assertIn("pending_directive_count", data["diagnostic"])
        self.assertIn("surprise_trend", data["diagnostic"])

    def test_system_directives_returns_pending_directives(self) -> None:
        response = self.client.get("/v1/system/directives")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("directives", data)
        self.assertIsInstance(data["directives"], list)

    def test_apply_runtime_contract_audit_directive_via_api(self) -> None:
        directive = EvolutionDirective(
            directive_id="directive-api-runtime-audit",
            directive_type=DirectiveType.TOPOLOGY,
            target_component="action_runtime",
            proposed_change={
                "action": "audit_runtime_contracts",
                "failure_types": {"validation_failed": 1},
            },
            reasoning="Runtime contract audit requested by API",
        )
        api_module.get_agent().pending_directives.append(directive)

        pending_response = self.client.get("/v1/system/directives")
        self.assertEqual(pending_response.status_code, 200)
        pending = pending_response.json()["directives"]
        pending_ids = [item["directive_id"] for item in pending]
        self.assertIn("directive-api-runtime-audit", pending_ids)

        apply_response = self.client.post(
            "/v1/system/directives/apply",
            json={"directive_id": "directive-api-runtime-audit"},
        )
        self.assertEqual(apply_response.status_code, 200)
        data = apply_response.json()
        self.assertEqual(data["status"], "ok")
        applied = data["directive"]
        self.assertEqual(applied["status"], "applied")
        self.assertIn("_audit", applied["proposed_change"])
        self.assertEqual(
            applied["proposed_change"]["_audit"]["validation_failed_recent_count"],
            1,
        )

    def test_system_introspect_returns_grounded_self_awareness(self) -> None:
        self.client.post(
            "/v1/chat/completions",
            json={"text": "Preciso de uma resposta curta", "session_id": "focused-session"},
        )

        response = self.client.post(
            "/v1/system/introspect",
            json={
                "query": "qual backend de raciocinio memoria e telemetria estao ativos nesta sessao?",
                "session_id": "focused-session",
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("response", data)
        self.assertIn("Sessão focused-session", data["response"])
        self.assertIn("reason=", data["response"])
        self.assertIn("memory=", data["response"])
        self.assertIn("telemetry=", data["response"])

    def test_telemetry_query_answers_tool_failure_question(self) -> None:
        session_id = "telemetry-query-session"
        self.client.post(
            "/v1/chat/completions",
            json={"text": "Primeira tentativa para gerar telemetria", "session_id": session_id},
        )
        response = self.client.post(
            "/v1/telemetry/query",
            json={
                "session_id": session_id,
                "question": "Em quais types de tools estou falhando mais?",
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("answer", data)

    def test_chat_completions_serializes_turns_with_same_session_lane(self) -> None:
        active = 0
        max_active = 0

        async def tracked_operation():
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.05)
            finally:
                active -= 1

        async def run_pair():
            await asyncio.gather(
                api_module._run_in_session_lane("lane-session", tracked_operation),
                api_module._run_in_session_lane("lane-session", tracked_operation),
            )

        asyncio.run(run_pair())
        self.assertEqual(max_active, 1)


if __name__ == "__main__":
    unittest.main()
