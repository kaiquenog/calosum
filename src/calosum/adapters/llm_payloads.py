from __future__ import annotations

from typing import Any

from calosum.shared.types import CognitiveBridgePacket, MemoryContext, UserTurn


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
