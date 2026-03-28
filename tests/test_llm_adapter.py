from __future__ import annotations

import json
import unittest

import httpx

from calosum.adapters.llm_qwen import QwenAdapterConfig, QwenLeftHemisphereAdapter
from calosum.shared.types import (
    BridgeControlSignal,
    CognitiveBridgePacket,
    MemoryContext,
    UserTurn,
)


RESULT_PAYLOAD = {
    "response_text": "Resposta estruturada",
    "lambda_program": {
        "signature": "Context -> Response",
        "expression": "lambda ctx: respond_text()",
        "expected_effect": "Emit a safe answer",
    },
    "actions": [
        {
            "action_type": "respond_text",
            "typed_signature": "ResponsePlan -> SafeTextMessage",
            "payload": {"text": "Resposta estruturada"},
            "safety_invariants": ["text only"],
        }
    ],
    "reasoning_summary": ["structured_output_ok"],
}


def _memory_context() -> MemoryContext:
    return MemoryContext(recent_episodes=[], semantic_rules=[], knowledge_triples=[])


def _bridge_packet() -> CognitiveBridgePacket:
    return CognitiveBridgePacket(
        context_id="ctx-1",
        soft_prompts=[],
        control=BridgeControlSignal(
            target_temperature=0.25,
            empathy_priority=False,
            system_directives=["be precise"],
            annotations={"salience_threshold": 0.7},
        ),
        salience=0.2,
        bridge_metadata={},
    )


class LlmAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_base_url_uses_responses_api_and_normalizes_common_model_alias(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(200, json={"output_text": json.dumps(RESULT_PAYLOAD)})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)

        adapter = QwenLeftHemisphereAdapter(
            QwenAdapterConfig(
                api_url="https://api.openai.com/v1",
                api_key="sk-test",
                model_name="gpt-5.4-mini",
                reasoning_effort="low",
            ),
            client=client,
        )

        result = await adapter.areason(
            UserTurn(session_id="openai-session", user_text="OI"),
            _bridge_packet(),
            _memory_context(),
        )

        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(payload["model"], "gpt-5-mini")
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertEqual(result.response_text, "Resposta estruturada")
        self.assertEqual(result.telemetry["api_mode"], "openai_responses")
        self.assertEqual(result.telemetry["model_name"], "gpt-5-mini")

    async def test_openai_compatible_endpoint_keeps_chat_completions_contract(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["payload"] = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps(RESULT_PAYLOAD)}}]},
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)

        adapter = QwenLeftHemisphereAdapter(
            QwenAdapterConfig(
                api_url="http://localhost:11434/v1/chat/completions",
                api_key="ollama",
                model_name="qwen3.5:0.8b",
            ),
            client=client,
        )

        result = await adapter.areason(
            UserTurn(session_id="local-session", user_text="OI"),
            _bridge_packet(),
            _memory_context(),
        )

        payload = captured["payload"]
        assert isinstance(payload, dict)
        self.assertEqual(captured["url"], "http://localhost:11434/v1/chat/completions")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(result.response_text, "Resposta estruturada")
        self.assertEqual(result.telemetry["api_mode"], "openai_compatible_chat")


if __name__ == "__main__":
    unittest.main()
