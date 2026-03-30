from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from calosum.shared.types import CognitiveBridgePacket, MemoryContext, UserTurn

logger = logging.getLogger(__name__)


def build_left_hemisphere_prompt(
    user_turn: UserTurn,
    bridge_packet: CognitiveBridgePacket,
    memory_context: MemoryContext,
    feedback: list[str] | None,
) -> str:
    rules = [item.statement for item in memory_context.semantic_rules]
    triples = [
        f"{item.subject} {item.predicate} {item.object}"
        for item in memory_context.knowledge_triples[:5]
    ]
    knowledge_block = "\n".join(triples) or "None"
    feedback_block = "\n".join(feedback) if feedback else "None"

    return f"""
Analyze the input and generate a JSON object.

Input: {user_turn.user_text}
Soft Prompts (Bridge): {[token.token for token in bridge_packet.soft_prompts]}
System Directives: {bridge_packet.control.system_directives}

# MEMORY & CONTEXT
Semantic Rules: {rules}
Knowledge Triples: {knowledge_block}

# RUNTIME OBSERVATIONS (Tool Outputs)
{feedback_block}

IMPORTANT - MULTI-STEP REASONING:
1. If you need more information, output actions like "execute_bash", "search_web", "read_file", or "introspect_self".
2. Leave 'response_text' empty ("") while gathering data.
3. Once the '# RUNTIME OBSERVATIONS' section contains the facts you need, you MUST PROVIDE THE FINAL ANSWER in 'response_text' and stop calling tools.
4. Do not repeat the same tool call if the observation already contains the answer.
5. Synthesize facts into a helpful response.

Available Action Types:
- "respond_text": {{ "text": "your final answer" }}
- "propose_plan": {{ "steps": ["step1", "step2"] }}
- "search_web": {{ "query": "search keywords" }}
- "read_file": {{ "path": "file/path.txt" }}
- "execute_bash": {{ "command": "ls -la" }}
- "introspect_self": {{ "query": "arquitetura" }}
""".strip()


def load_compiled_prompt_artifact(compiled_prompt_path: Path | None) -> dict[str, Any]:
    if compiled_prompt_path is None:
        return {}

    path = Path(compiled_prompt_path)
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load compiled prompt artifact from %s: %s", path, exc)
        return {}

    if not isinstance(payload, dict):
        return {}
    return payload


def load_compiled_examples(compiled_prompt_path: Path | None) -> list[dict[str, Any]]:
    payload = load_compiled_prompt_artifact(compiled_prompt_path)
    examples = payload.get("few_shot_examples", [])
    if not isinstance(examples, list):
        return []
    return [item for item in examples if isinstance(item, dict)]


def augment_prompt_with_compiled_artifact(
    prompt: str,
    compiled_prompt_artifact: dict[str, Any],
) -> str:
    if not compiled_prompt_artifact:
        return prompt

    sections: list[str] = []
    selected_prompt = compiled_prompt_artifact.get("selected_prompt")
    if isinstance(selected_prompt, str) and selected_prompt.strip():
        sections.append(
            "Optimized Prompt Directives (offline):\n"
            f"{selected_prompt.strip()}"
        )

    notes = compiled_prompt_artifact.get("optimization_notes", [])
    if isinstance(notes, list):
        rendered_notes = [f"- {note}" for note in notes if isinstance(note, str) and note.strip()]
        if rendered_notes:
            sections.append("Optimization Notes:\n" + "\n".join(rendered_notes))

    compiled_examples = compiled_prompt_artifact.get("few_shot_examples", [])
    if isinstance(compiled_examples, list):
        rendered_examples = _render_compiled_examples(compiled_examples)
        if rendered_examples:
            sections.append(
                "Few-shot Examples (optimized offline):\n"
                + "\n\n".join(rendered_examples)
            )

    if not sections:
        return prompt
    return f"{prompt}\n\n" + "\n\n".join(sections)


def augment_prompt_with_compiled_examples(
    prompt: str,
    compiled_examples: list[dict[str, Any]],
) -> str:
    return augment_prompt_with_compiled_artifact(
        prompt,
        {"few_shot_examples": compiled_examples},
    )


def _render_compiled_examples(compiled_examples: list[dict[str, Any]]) -> list[str]:
    rendered_examples: list[str] = []
    for example in compiled_examples[:3]:
        input_text = example.get("input_text")
        response_text = example.get("response_text")
        if not isinstance(input_text, str) or not isinstance(response_text, str):
            continue
        rendered_examples.append(
            f"Example Input: {input_text}\nExample Response: {response_text}"
        )
    return rendered_examples


def left_hemisphere_result_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "response_text",
            "lambda_program",
            "actions",
            "reasoning_summary",
        ],
        "properties": {
            "response_text": {
                "type": "string",
                "description": "The final synthesized text response to the user. MUST be empty (\"\") if you are still gathering data via epistemic tools (like introspect_self or execute_bash). Only fill this when you are providing the final answer based on the 'Runtime Feedback' observations."
            },
            "lambda_program": {
                "type": "object",
                "additionalProperties": False,
                "required": ["signature", "expression", "expected_effect"],
                "properties": {
                    "signature": {"type": "string"},
                    "expression": {"type": "string"},
                    "expected_effect": {"type": "string"},
                },
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "action_type",
                        "typed_signature",
                        "payload",
                        "safety_invariants",
                    ],
                    "properties": {
                        "action_type": {"type": "string"},
                        "typed_signature": {"type": "string"},
                        "payload": {"type": "object", "additionalProperties": True},
                        "safety_invariants": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "reasoning_summary": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }


def extract_responses_content(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text", "")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
    return "\n".join(chunks)


def extract_chat_content(data: dict[str, Any]) -> str:
    msg = data["choices"][0]["message"]
    content = msg.get("content", "")

    if isinstance(content, list):
        text_chunks = [item.get("text", "") for item in content if isinstance(item, dict)]
        content = "\n".join(chunk for chunk in text_chunks if chunk)

    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].strip()

    if "{" in content and "}" in content:
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        content = content[start_idx:end_idx]
    return content.strip()


def build_openai_responses_payload(prompt: str, model: str, max_tokens: int, reasoning_effort: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input_text": prompt,
        "output_format": "json_object",
        "parameters": {"max_new_tokens": max_tokens, "temperature": 0.1, "top_p": 0.9},
        "text": {
            "format": {
                "type": "json_schema",
                "strict": False,
                "json_schema": {
                    "name": "left_hemisphere_result",
                    "schema": left_hemisphere_result_schema()
                }
            }
        },
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload

def build_openai_chat_payload(prompt: str, model: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a logical neuro-symbolic agent. Output valid JSON only, corresponding to LeftHemisphereResult format."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
    }

def build_compatible_chat_payload(prompt: str, model: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a logical neuro-symbolic agent. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
