from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from calosum.adapters.llm.llm_payloads import (
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
from calosum.adapters.llm.llm_payload_parser import parse_to_result, fallback_result
from calosum.shared.utils.async_utils import run_sync
from calosum.shared.models.types import (
    ActionExecutionResult,
    PerceptionSummary,
    ActionPlannerResult,
    MemoryContext,
    PrimitiveAction,
    TypedLambdaProgram,
    UserTurn,
    CognitiveWorkspace,
)

logger = logging.getLogger(__name__)
@dataclass(slots=True)
class QwenAdapterConfig:
    api_url: str | None = None
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
        
        provider = self.config.provider.strip().lower()
        if not self.config.api_url and provider not in {"openai", "responses", "openai_responses", "openai_chat", "chat", "openrouter"}:
            raise ValueError(
                "QwenLeftHemisphereAdapter requires an explicit api_url or a valid cloud provider (e.g., 'openai', 'openrouter')."
            )

        headers: dict[str, str] = {}
        if self.config.api_key and self.config.api_key != "empty":
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/kaiquenog/calosum"
            headers["X-Title"] = "Calosum Agent Framework"

        self.client = client or httpx.AsyncClient(headers=headers, timeout=300.0)
        self.compiled_prompt_artifact = load_compiled_prompt_artifact(self.config.compiled_prompt_path)
        self.compiled_examples = load_compiled_examples(self.config.compiled_prompt_path)

    def reason(
        self,
        user_turn: UserTurn,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> ActionPlannerResult:
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

    async def _post_once(self, url: str, json_payload: dict[str, Any]) -> httpx.Response:
        response = await self.client.post(url, json=json_payload)
        response.raise_for_status()
        return response

    async def areason(
        self,
        user_turn: UserTurn,
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> ActionPlannerResult:
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
            session_briefing=(
                str(workspace.task_frame.get("session_briefing", ""))
                if workspace and workspace.task_frame
                else None
            ),
        )
        
        if introspective_intent:
            prompt += "\n\nO usuário fez uma pergunta sobre o seu próprio estado ou arquitetura. Utilize a ferramenta 'introspect_self' passando a query correspondente para obter dados reais antes de responder."
            
        prompt = augment_prompt_with_compiled_artifact(prompt, self.compiled_prompt_artifact)
        temperature = self._resolve_temperature(bridge_packet)
        self._last_annotations = dict(bridge_packet.control.annotations or {})
        request = self._build_request(prompt, temperature=temperature)

        try:
            response = await self._post_once(request["url"], request["payload"])
            data = response.json()
            content = self._extract_content(data, request["api_mode"])

            if not content.strip():
                result = fallback_result(
                    f"empty_response_from_{request['api_mode']}",
                    request["api_mode"],
                    request["resolved_model"],
                )
                if workspace:
                    workspace.left_notes.update({"error": "empty_response"})
                return result

            parsed = json.loads(content)
            result = parse_to_result(
                parsed,
                api_mode=request["api_mode"],
                resolved_model=request["resolved_model"],
                compiled_few_shot_count=len(self.compiled_examples),
                compiled_prompt_selected=bool(self.compiled_prompt_artifact.get("selected_prompt")),
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
            result = ActionPlannerResult(
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
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            result = fallback_result(
                f"backtracking:http_transport_failure:{exc.__class__.__name__}",
                request["api_mode"],
                request["resolved_model"],
                bridge_packet.control.system_directives,
            )
            result.telemetry["backtracking_node"] = "http_transport_failure"
            result.telemetry["http_retries"] = 0
            result.reasoning_summary.append("Backtracking node: transporte HTTP indisponivel; replanejar politica")
            if workspace:
                workspace.left_notes.update(
                    {
                        "error": "http_transport_failure",
                        "details": repr(exc),
                        "backtracking_node": "http_transport_failure",
                    }
                )
            return result
        except Exception as exc:
            result = fallback_result(
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
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        previous_result: ActionPlannerResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> ActionPlannerResult:
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
        bridge_packet: PerceptionSummary,
        memory_context: MemoryContext,
        previous_result: ActionPlannerResult,
        rejected_results: list[ActionExecutionResult],
        attempt: int,
        critique_feedback: list[str] | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> ActionPlannerResult:
        feedback = []
        if critique_feedback:
            feedback.extend(critique_feedback)
        else:
            feedback.extend([
                f"Ação {item.action_type} rejeitada: {', '.join(item.violations)}"
                for item in rejected_results
            ])
        feedback.insert(0, "Backtracking: revise a politica para evitar acoes invalidas e convergir em um plano executavel.")
        return await self.areason(user_turn, bridge_packet, memory_context, feedback, attempt, workspace)

    def _build_request(self, prompt: str, *, temperature: float | None = None) -> dict[str, Any]:
        api_mode = self._resolve_api_mode()
        resolved_model = self._resolve_model_name(api_mode)
        top_p, logit_bias = self._resolve_sampling_controls()

        if api_mode == "openai_responses":
            return {
                "api_mode": api_mode, "resolved_model": resolved_model,
                "url": self._responses_url(),
                "payload": build_openai_responses_payload(
                    prompt,
                    resolved_model,
                    self.config.max_tokens,
                    self.config.reasoning_effort,
                    temperature,
                    top_p,
                    logit_bias,
                ),
            }

        if api_mode == "openai_chat":
            return {
                "api_mode": api_mode, "resolved_model": resolved_model,
                "url": self._chat_completions_url(),
                "payload": build_openai_chat_payload(
                    prompt,
                    resolved_model,
                    self.config.max_tokens,
                    temperature,
                    top_p,
                    logit_bias,
                ),
            }

        if api_mode == "openrouter":
            return {
                "api_mode": api_mode, "resolved_model": resolved_model,
                "url": self._chat_completions_url(),
                "payload": build_openai_chat_payload(
                    prompt,
                    resolved_model,
                    self.config.max_tokens,
                    temperature,
                    top_p,
                    logit_bias,
                ),
            }

        return {
            "api_mode": api_mode, "resolved_model": resolved_model,
            "url": self._chat_completions_url(),
            "payload": build_compatible_chat_payload(
                    prompt,
                    resolved_model,
                    self.config.max_tokens,
                    temperature,
                    top_p,
                    logit_bias,
                ),
        }

    def _resolve_temperature(self, bridge_packet: PerceptionSummary) -> float:
        candidate = float(bridge_packet.control.target_temperature)
        return round(min(1.5, max(0.0, candidate)), 3)

    def _resolve_sampling_controls(self) -> tuple[float | None, dict[str, float] | None]:
        annotations = getattr(self, "_last_annotations", None)
        if annotations is None:
            return None, None
        raw_top_p = annotations.get("target_top_p")
        top_p: float | None = None
        if raw_top_p is not None:
            try:
                top_p = round(min(1.0, max(0.0, float(raw_top_p))), 3)
            except (TypeError, ValueError):
                top_p = None
        raw_logit_bias = annotations.get("target_logit_bias")
        if not isinstance(raw_logit_bias, dict):
            return top_p, None
        mapped = {str(key): float(value) for key, value in raw_logit_bias.items() if isinstance(value, (int, float))}
        return top_p, mapped or None

    def _resolve_api_mode(self) -> str:
        provider = self.config.provider.strip().lower()
        if provider in {"openai_responses", "openai_chat", "openai_compatible_chat", "openrouter"}:
            return provider
        if provider in {"openai", "responses"}:
            return "openai_responses"
        if provider in {"chat"}:
            return "openai_chat"

        parsed = urlparse(self.config.api_url or "")
        hostname = (parsed.hostname or "").lower()
        path = parsed.path.rstrip("/")

        if hostname == "api.openai.com":
            return "openai_responses"

        return "openai_compatible_chat"

    def _resolve_model_name(self, api_mode: str) -> str:
        model = self.config.model_name.strip()
        if not model:
            if api_mode == "openai_responses":
                return "gpt-4o-mini"
            return "gpt-4o-mini"
        return model

    def _responses_url(self) -> str:
        if self.config.provider.strip().lower() == "openrouter" and not self.config.api_url:
             return "https://openrouter.ai/api/v1/responses"
        
        base = (self.config.api_url or "https://api.openai.com/v1").rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        if base.endswith("/responses"):
            return base
        if base.endswith("/v1"):
            return f"{base}/responses"
        return f"{base}/v1/responses"

    def _chat_completions_url(self) -> str:
        if self.config.provider.strip().lower() == "openrouter" and not self.config.api_url:
            return "https://openrouter.ai/api/v1/chat/completions"
            
        base = (self.config.api_url or "https://api.openai.com/v1").rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _extract_content(self, data: dict[str, Any], api_mode: str) -> str:
        if api_mode == "openai_responses":
            return extract_responses_content(data)
        return extract_chat_content(data)
