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
Runtime Feedback: {feedback or []}

Available Action Types (Use exactly these action_type values):
- "respond_text": {{ "text": "your response here" }}
- "propose_plan": {{ "steps": ["step1", "step2"] }}
- "search_web": {{ "query": "search keywords" }}
- "write_file": {{ "path": "file/path.txt", "content": "file content" }}
""".strip()


def load_compiled_examples(compiled_prompt_path: Path | None) -> list[dict[str, Any]]:
    if compiled_prompt_path is None:
        return []

    path = Path(compiled_prompt_path)
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load compiled prompt artifact from %s: %s", path, exc)
        return []

    examples = payload.get("few_shot_examples", [])
    if not isinstance(examples, list):
        return []
    return [item for item in examples if isinstance(item, dict)]


def augment_prompt_with_compiled_examples(
    prompt: str,
    compiled_examples: list[dict[str, Any]],
) -> str:
    if not compiled_examples:
        return prompt

    rendered_examples: list[str] = []
    for example in compiled_examples[:3]:
        input_text = example.get("input_text")
        response_text = example.get("response_text")
        if not isinstance(input_text, str) or not isinstance(response_text, str):
            continue
        rendered_examples.append(
            f"Example Input: {input_text}\nExample Response: {response_text}"
        )

    if not rendered_examples:
        return prompt

    return (
        f"{prompt}\n\n"
        "Few-shot Examples (optimized offline):\n"
        + "\n\n".join(rendered_examples)
    )


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
