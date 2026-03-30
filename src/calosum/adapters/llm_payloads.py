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
    return f"""
Analyze the input and generate a JSON object.

Input: {user_turn.user_text}
Soft Prompts (Bridge): {[token.token for token in bridge_packet.soft_prompts]}
System Directives: {bridge_packet.control.system_directives}
Semantic Rules: {rules}
Knowledge Triples: {triples}
Runtime Feedback (Observations): {feedback or []}

IMPORTANT - MULTI-STEP REASONING: You are an autonomous agent. If you need to gather information, output actions like "execute_bash", "search_web", "read_file", or "introspect_self". The system will execute them and provide the output back to you as "Runtime Feedback" in a loop. Do NOT include a "respond_text" action if you are just exploring; wait until you have gathered all data, then use "respond_text" to give the final answer.

Available Action Types (Use exactly these action_type values):
- "respond_text": {{ "text": "your response here" }}
- "propose_plan": {{ "steps": ["step1", "step2"] }}
- "load_semantic_rules": {{ "rules": ["grounding rule 1", "grounding rule 2"] }}
- "search_web": {{ "query": "search keywords" }}
- "write_file": {{ "path": "file/path.txt", "content": "file content" }}
- "read_file": {{ "path": "file/path.txt" }}
- "execute_bash": {{ "command": "ls -la" }}
- "introspect_self": {{ "query": "status" }}
- "code_execution": {{ "code": "print(sum(range(5)))", "approved": true }}
- "http_request": {{ "method": "GET", "url": "https://example.com/api" }}
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
            "response_text": {"type": "string"},
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
