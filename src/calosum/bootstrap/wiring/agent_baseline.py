from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from calosum.adapters.execution.action_runtime import ConcreteActionRuntime
from calosum.adapters.llm.llm_qwen import QwenAdapterConfig, QwenLeftHemisphereAdapter
from calosum.adapters.memory.text_embeddings import TextEmbeddingAdapter, TextEmbeddingAdapterConfig
from calosum.bootstrap.infrastructure.settings import InfrastructureSettings
from calosum.shared.models.types import (
    BridgeControlSignal,
    CognitiveBridgePacket,
    MemoryContext,
    SemanticRule,
    UserTurn,
)


@dataclass(slots=True)
class AgentBaselineConfig:
    memory_path: Path = Path(".calosum-runtime/baseline/memory.jsonl")
    memory_limit: int = 12


class AgentBaseline:
    """Baseline enxuto para comparação de sprints.

    Stack: LLM API + embeddings + memória JSONL + tool loop básico.
    Sem hemisférios, sem bridge adaptativo e sem group turns.
    """

    def __init__(
        self,
        *,
        left_hemisphere: QwenLeftHemisphereAdapter,
        embedder: TextEmbeddingAdapter,
        action_runtime: ConcreteActionRuntime,
        config: AgentBaselineConfig | None = None,
    ) -> None:
        self.left_hemisphere = left_hemisphere
        self.embedder = embedder
        self.action_runtime = action_runtime
        self.config = config or AgentBaselineConfig()

    @classmethod
    def from_settings(cls, settings: InfrastructureSettings) -> "AgentBaseline":
        left = QwenLeftHemisphereAdapter(
            QwenAdapterConfig(
                api_url=settings.left_hemisphere_endpoint or "http://localhost:8000/v1/chat/completions",
                api_key=settings.left_hemisphere_api_key or "empty",
                model_name=settings.reason_model or settings.left_hemisphere_model or "gpt-4o-mini",
                provider=settings.left_hemisphere_provider or "auto",
                reasoning_effort=settings.left_hemisphere_reasoning_effort,
            )
        )
        embedder = TextEmbeddingAdapter(
            TextEmbeddingAdapterConfig(
                provider=settings.embedding_provider or "auto",
                api_url=settings.embedding_endpoint or settings.left_hemisphere_endpoint,
                api_key=settings.embedding_api_key or settings.left_hemisphere_api_key,
                model_name=settings.embedding_model or "text-embedding-3-small",
            )
        )
        runtime = ConcreteActionRuntime(vault=settings.vault)
        memory_path = (settings.memory_dir or Path(".calosum-runtime/memory")) / "baseline_memory.jsonl"
        return cls(
            left_hemisphere=left,
            embedder=embedder,
            action_runtime=runtime,
            config=AgentBaselineConfig(memory_path=memory_path),
        )

    def process_turn(self, user_turn: UserTurn) -> dict[str, Any]:
        started_at = perf_counter()
        context = self._build_context()
        embedding = self.embedder.embed_texts([user_turn.user_text])[0]

        bridge_packet = CognitiveBridgePacket(
            context_id=user_turn.turn_id,
            soft_prompts=[],
            control=BridgeControlSignal(
                target_temperature=0.2,
                empathy_priority=False,
                system_directives=["baseline_structured_json"],
                annotations={"baseline": True},
            ),
            salience=0.5,
            latent_vector=embedding[:16],
            bridge_metadata={"baseline_mode": "single_pass"},
        )

        left_result = self.left_hemisphere.reason(user_turn, bridge_packet, context)
        execution_results = self.action_runtime.run(left_result)
        succeeded = sum(1 for item in execution_results if item.status == "success")
        total = len(execution_results)
        tool_success_rate = round((succeeded / total), 4) if total else 1.0

        payload = {
            "session_id": user_turn.session_id,
            "turn_id": user_turn.turn_id,
            "input_text": user_turn.user_text,
            "response_text": left_result.response_text,
            "actions": [item.action_type for item in left_result.actions],
            "tool_success_rate": tool_success_rate,
            "latency_ms": round((perf_counter() - started_at) * 1000.0, 3),
            "embedding": embedding,
        }
        self._append_memory(payload)
        return payload

    def _build_context(self) -> MemoryContext:
        if not self.config.memory_path.exists():
            return MemoryContext()

        rows = [
            json.loads(line)
            for line in self.config.memory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        recent = rows[-self.config.memory_limit :]
        rules = [
            SemanticRule(
                rule_id=f"baseline-rule-{idx}",
                statement=f"{row.get('input_text', '')} => {row.get('response_text', '')}",
                strength=0.4,
                supporting_episodes=[str(row.get("turn_id", ""))],
                tags=["baseline", "jsonl_memory"],
            )
            for idx, row in enumerate(recent)
        ]
        return MemoryContext(semantic_rules=rules)

    def _append_memory(self, payload: dict[str, Any]) -> None:
        self.config.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.memory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
