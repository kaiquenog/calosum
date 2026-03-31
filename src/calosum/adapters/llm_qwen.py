from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from calosum.adapters.llm_payloads import (
    augment_prompt_with_compiled_artifact,
    build_compatible_chat_payload,
    build_left_hemisphere_prompt,
    build_openai_chat_payload,
    build_openai_responses_payload,
    extract_chat_content,
    extract_responses_content,
    load_compiled_examples,
    load_compiled_prompt_artifact,
)
from calosum.shared.async_utils import run_sync
from calosum.shared.types import (
    ActionExecutionResult,
    CognitiveBridgePacket,
    LeftHemisphereResult,
    MemoryContext,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
    CognitiveWorkspace,
)

logger = logging.getLogger(__name__)
@dataclass(slots=True)
class QwenAdapterConfig:
    api_url: str = "http://localhost:8000/v1/chat/completions"
    api_key: str = "empty"
    model_name: str = "Qwen/Qwen-3.5-9B-Instruct"
    max_tokens: int = 4096
    provider: str = "auto"
    reasoning_effort: str | None = None
    compiled_prompt_path: Path | None = Path(".calosum-runtime/dspy_artifacts/latest/compiled_prompt.json")


class QwenLeftHemisphereAdapter:
    """
    Adapter estruturado para o hemisferio esquerdo.

    Mantem compatibilidade com endpoints locais no formato OpenAI-compatible,
    mas detecta automaticamente a OpenAI oficial e usa o Responses API com
    Structured Outputs quando apropriado.
    """

    def __init__(
        self,
        config: QwenAdapterConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or QwenAdapterConfig()
        headers: dict[str, str] = {}
        if self.config.api_key and self.config.api_key != "empty":
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        self.client = client or httpx.AsyncClient(headers=headers, timeout=300.0)
        self.compiled_prompt_artifact = load_compiled_prompt_artifact(self.config.compiled_prompt_path)
        self.compiled_examples = load_compiled_examples(self.config.compiled_prompt_path)

    def reason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        return run_sync(
            self.areason(
                user_turn=user_turn,
                bridge_packet=bridge_packet,
                memory_context=memory_context,
                runtime_feedback=runtime_feedback,
                attempt=attempt,
                workspace=workspace,
            )
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
        reraise=True
    )
    async def _post_with_retry(self, url: str, json_payload: dict[str, Any]) -> httpx.Response:
        response = await self.client.post(url, json=json_payload)
        response.raise_for_status()
        return response

    async def areason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        # Detectar intenção introspectiva baseada no workspace/input
        introspective_intent = False
        text_lower = user_turn.user_text.lower()
        if "como você funciona" in text_lower or "como voce funciona" in text_lower or "gargalo" in text_lower or "arquitetura" in text_lower or "diretiva" in text_lower:
            introspective_intent = True
            
        prompt = build_left_hemisphere_prompt(
            user_turn,
            bridge_packet,
            memory_context,
            runtime_feedback,
        )
        
        if introspective_intent:
            prompt += "\n\nO usuário fez uma pergunta sobre o seu próprio estado ou arquitetura. Utilize a ferramenta 'introspect_self' passando a query correspondente para obter dados reais antes de responder."
            
        prompt = augment_prompt_with_compiled_artifact(prompt, self.compiled_prompt_artifact)
        request = self._build_request(prompt)

        try:
            response = await self._post_with_retry(request["url"], request["payload"])
            data = response.json()
            content = self._extract_content(data, request["api_mode"])

            if not content.strip():
                result = self._fallback_result(
                    f"empty_response_from_{request['api_mode']}",
                    request["api_mode"],
                    request["resolved_model"],
                )
                if workspace:
                    workspace.left_notes.update({"error": "empty_response"})
                return result

            parsed = json.loads(content)
            result = self._parse_to_result(
                parsed,
                api_mode=request["api_mode"],
                resolved_model=request["resolved_model"],
                system_directives=bridge_packet.control.system_directives,
            )
            
            if workspace:
                workspace.left_notes.update({
                    "response_text": result.response_text,
                    "reasoning_summary": result.reasoning_summary,
                    "actions": [a.action_type for a in result.actions],
                    "introspective_intent": introspective_intent
                })
            return result
            
        except json.JSONDecodeError as exc:
            result = LeftHemisphereResult(
                response_text="",
                lambda_program=TypedLambdaProgram("Any", "()", "None"),
                actions=[],
                reasoning_summary=[f"Structured output parse failed: {exc}"],
                telemetry={
                    "adapter": "QwenLeftHemisphereAdapter",
                    "api_mode": request["api_mode"],
                    "model_name": request["resolved_model"],
                },
            )
            if workspace:
                workspace.left_notes.update({"error": "json_decode_error", "details": str(exc)})
            return result
        except Exception as exc:
            result = self._fallback_result(
                repr(exc),
                request["api_mode"],
                request["resolved_model"],
                bridge_packet.control.system_directives,
            )
            if workspace:
                workspace.left_notes.update({"error": "runtime_exception", "details": repr(exc)})
            return result

    def repair(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        return run_sync(
            self.arepair(
                user_turn=user_turn,
                bridge_packet=bridge_packet,
                memory_context=memory_context,
                previous_result=previous_result,
                rejected_results=rejected_results,
                attempt=attempt,
                critique_feedback=critique_feedback,
                workspace=workspace,
            )
        )

    async def arepair(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        previous_result: LeftHemisphereResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        feedback = []
        if critique_feedback:
            feedback.extend(critique_feedback)
        else:
            feedback.extend([
                f"Ação {item.action_type} rejeitada: {', '.join(item.violations)}"
                for item in rejected_results
            ])
        return await self.areason(user_turn, bridge_packet, memory_context, feedback, attempt, workspace)

    def _build_request(self, prompt: str) -> dict[str, Any]:
        api_mode = self._resolve_api_mode()
        resolved_model = self._resolve_model_name(api_mode)

        if api_mode == "openai_responses":
            return {
                "api_mode": api_mode, "resolved_model": resolved_model,
                "url": self._responses_url(), "payload": build_openai_responses_payload(prompt, resolved_model, self.config.max_tokens, self.config.reasoning_effort),
            }

        if api_mode == "openai_chat":
            return {
                "api_mode": api_mode, "resolved_model": resolved_model,
                "url": self._chat_completions_url(), "payload": build_openai_chat_payload(prompt, resolved_model, self.config.max_tokens),
            }

        return {
            "api_mode": api_mode, "resolved_model": resolved_model,
            "url": self._chat_completions_url(), "payload": build_compatible_chat_payload(prompt, resolved_model, self.config.max_tokens),
        }

    def _resolve_api_mode(self) -> str:
        provider = self.config.provider.strip().lower()
        if provider in {"openai_responses", "openai_chat", "openai_compatible_chat"}:
            return provider
        if provider in {"openai", "responses"}:
            return "openai_responses"
        if provider in {"chat"}:
            return "openai_chat"

        parsed = urlparse(self.config.api_url)
        hostname = (parsed.hostname or "").lower()
        path = parsed.path.rstrip("/")

        if hostname == "api.openai.com":
            return "openai_responses"

        return "openai_compatible_chat"

    def _resolve_model_name(self, api_mode: str) -> str:
        model = self.config.model_name.strip()
        if not model:
            if api_mode == "openai_responses":
                return "gpt-5.4-mini"
            return "gpt-4o-mini"
        return model

    def _responses_url(self) -> str:
        base = self.config.api_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        if base.endswith("/responses"):
            return base
        if base.endswith("/v1"):
            return f"{base}/responses"
        return f"{base}/v1/responses"

    def _chat_completions_url(self) -> str:
        base = self.config.api_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _extract_content(self, data: dict[str, Any], api_mode: str) -> str:
        if api_mode == "openai_responses":
            return extract_responses_content(data)
        return extract_chat_content(data)

    def _parse_to_result(
        self,
        parsed: dict[str, Any],
        *,
        api_mode: str,
        resolved_model: str,
        system_directives: list[str] | None = None,
    ) -> LeftHemisphereResult:
        lambda_prog = parsed.get("lambda_program", {})
        response_text = str(parsed.get("response_text", "") or "")
        reasoning_summary = [str(item) for item in parsed.get("reasoning_summary", []) if str(item).strip()]

        program = TypedLambdaProgram(
            signature=str(lambda_prog.get("signature") or "Context -> Response"),
            expression=str(
                lambda_prog.get("expression")
                or "(lambda context memory (sequence (emit respond_text)))"
            ),
            expected_effect=str(lambda_prog.get("expected_effect") or "Deliver a safe response"),
        )

        actions: list[PrimitiveAction] = []
        for item in parsed.get("actions", []):
            if not isinstance(item, dict):
                continue
            actions.append(
                PrimitiveAction(
                    action_type=str(item.get("action_type", "unknown")),
                    typed_signature=str(item.get("typed_signature", "Any -> Any")),
                    payload=dict(item.get("payload", {})),
                    safety_invariants=[
                        str(invariant)
                        for invariant in item.get("safety_invariants", [])
                        if isinstance(invariant, str) and invariant.strip()
                    ],
                )
            )

        if not response_text.strip():
            for action in actions:
                if action.action_type != "respond_text":
                    continue
                candidate = action.payload.get("text")
                if isinstance(candidate, str) and candidate.strip():
                    response_text = candidate.strip()
                    break

        if not actions and response_text.strip():
            actions = [
                PrimitiveAction(
                    action_type="respond_text",
                    typed_signature="ResponsePlan -> SafeTextMessage",
                    payload={"text": response_text},
                    safety_invariants=["safe output only"],
                )
            ]

        if not reasoning_summary:
            reasoning_summary = ["structured_output_ok"]

        if not response_text.strip() and not actions:
            raise ValueError("incomplete_structured_output: empty response_text and no actions")

        return LeftHemisphereResult(
            response_text=response_text,
            lambda_program=program,
            actions=actions,
            reasoning_summary=reasoning_summary,
            telemetry={
                "adapter": "QwenLeftHemisphereAdapter",
                "api_mode": api_mode,
                "model_name": resolved_model,
                "compiled_few_shot_count": len(self.compiled_examples),
                "compiled_prompt_selected": bool(self.compiled_prompt_artifact.get("selected_prompt")),
                "system_directives": system_directives or [],
            },
        )

    def _fallback_result(
        self,
        error: str,
        api_mode: str,
        resolved_model: str,
        system_directives: list[str] | None = None,
    ) -> LeftHemisphereResult:
        return LeftHemisphereResult(
            response_text="Desculpe, meu subsistema de raciocínio falhou temporariamente.",
            lambda_program=TypedLambdaProgram("Fallback", "()", "None"),
            actions=[],
            reasoning_summary=[f"Erro LLM: {error}"],
            telemetry={
                "adapter": "QwenLeftHemisphereAdapter",
                "api_mode": api_mode,
                "model_name": resolved_model,
                "error": error,
                "system_directives": system_directives or [],
            },
        )
