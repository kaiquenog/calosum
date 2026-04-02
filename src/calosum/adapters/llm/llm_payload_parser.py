from typing import Any
from calosum.shared.models.types import LeftHemisphereResult, TypedLambdaProgram, PrimitiveAction

def parse_to_result(
    parsed: dict[str, Any],
    *,
    api_mode: str,
    resolved_model: str,
    compiled_few_shot_count: int = 0,
    compiled_prompt_selected: bool = False,
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

    actions = [
        PrimitiveAction(
            action_type=str(item.get("action_type", "unknown")),
            typed_signature=str(item.get("typed_signature", "Any -> Any")),
            payload=dict(item.get("payload", {})),
            safety_invariants=[str(i) for i in item.get("safety_invariants", []) if isinstance(i, str) and i.strip()],
        )
        for item in parsed.get("actions", []) if isinstance(item, dict)
    ]

    if not response_text.strip():
        for action in actions:
            if action.action_type == "respond_text":
                candidate = action.payload.get("text")
                if isinstance(candidate, str) and candidate.strip():
                    response_text = candidate.strip()
                    break
    if not actions and response_text.strip():
        actions = [PrimitiveAction(action_type="respond_text", typed_signature="ResponsePlan -> SafeTextMessage", payload={"text": response_text}, safety_invariants=["safe output only"])]
    if not reasoning_summary: reasoning_summary = ["structured_output_ok"]
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
            "compiled_few_shot_count": compiled_few_shot_count,
            "compiled_prompt_selected": compiled_prompt_selected,
            "system_directives": system_directives or [],
        },
    )

def fallback_result(
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
