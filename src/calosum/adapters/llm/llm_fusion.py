from __future__ import annotations

import asyncio
import hashlib
import math
import random
from dataclasses import dataclass, replace
from typing import Any, Literal

from calosum.shared.models.types import (
    ActionExecutionResult,
    CognitiveBridgePacket,
    LeftHemisphereResult,
    MemoryContext,
    UserTurn,
    CognitiveWorkspace,
)
from calosum.shared.utils.async_utils import maybe_await, run_sync

FusionSelectionMode = Literal["guided", "random"]


@dataclass(slots=True)
class FusionResult:
    result: LeftHemisphereResult
    selected_index: int
    method: str
    score: float | None = None
    scores: list[float] | None = None


class SemanticFusionSelector:
    def __init__(self, *, uncertainty_threshold: float = 0.5, embedding_dim: int = 384) -> None:
        self.uncertainty_threshold = uncertainty_threshold
        self.embedding_dim = embedding_dim

    def select(
        self,
        candidates: list[LeftHemisphereResult],
        jepa_pred: list[float],
        uncertainty: float,
    ) -> FusionResult:
        if not candidates:
            raise ValueError("SemanticFusionSelector.select requires at least one candidate")
        if uncertainty > self.uncertainty_threshold:
            return FusionResult(result=candidates[0], selected_index=0, method="passthrough")

        target = self._normalize_embedding(jepa_pred)
        scores: list[float] = []
        for candidate in candidates:
            embedding = self._encode_result(candidate)
            scores.append(self._cosine_similarity(embedding, target))
        best = max(range(len(scores)), key=scores.__getitem__)
        return FusionResult(
            result=candidates[best],
            selected_index=best,
            method="jepa_guided",
            score=round(scores[best], 6),
            scores=[round(item, 6) for item in scores],
        )

    def _encode_result(self, result: LeftHemisphereResult) -> list[float]:
        text = result.response_text.strip()
        if not text and result.actions:
            action_texts = [
                str(item.payload.get("text", "")).strip()
                for item in result.actions
                if isinstance(item.payload, dict)
            ]
            text = " ".join(part for part in action_texts if part)
        if not text:
            text = " ".join(result.reasoning_summary)
        return self._encode_text(text)

    def _encode_text(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        if not tokens:
            tokens = ["_empty_"]
        vector = [0.0] * self.embedding_dim
        for token in tokens:
            token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = token_hash % self.embedding_dim
            vector[idx] += 1.0
        for left, right in zip(tokens, tokens[1:], strict=False):
            bi_hash = int(hashlib.sha256(f"{left}|{right}".encode("utf-8")).hexdigest(), 16)
            idx = bi_hash % self.embedding_dim
            vector[idx] += 0.7
        return self._normalize_embedding(vector)

    def _normalize_embedding(self, vector: list[float]) -> list[float]:
        if len(vector) > self.embedding_dim:
            resized = vector[: self.embedding_dim]
        elif len(vector) < self.embedding_dim:
            resized = vector + ([0.0] * (self.embedding_dim - len(vector)))
        else:
            resized = list(vector)
        norm = math.sqrt(sum(value * value for value in resized))
        if norm == 0.0:
            return [0.0] * self.embedding_dim
        return [value / norm for value in resized]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        mag_l = math.sqrt(sum(a * a for a in left))
        mag_r = math.sqrt(sum(b * b for b in right))
        if mag_l == 0.0 or mag_r == 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (mag_l * mag_r)))

    def _tokenize(self, text: str) -> list[str]:
        return [
            token.strip(".,;:!?()[]{}\"'").lower()
            for token in text.split()
            if token.strip(".,;:!?()[]{}\"'")
        ]


@dataclass(slots=True)
class MultiSampleFusionConfig:
    enabled: bool = True
    n_candidates: int = 3
    uncertainty_threshold: float = 0.5
    selection_mode: FusionSelectionMode = "guided"
    temperature_offsets: tuple[float, ...] = (0.0, 0.1, -0.1)


class MultiSampleFusionLeftHemisphereAdapter:
    def __init__(
        self,
        provider: Any,
        config: MultiSampleFusionConfig | None = None,
        selector: SemanticFusionSelector | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or MultiSampleFusionConfig()
        self.selector = selector or SemanticFusionSelector(
            uncertainty_threshold=self.config.uncertainty_threshold
        )

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

    async def areason(
        self,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None = None,
        attempt: int = 0,
        workspace: CognitiveWorkspace | None = None,
    ) -> LeftHemisphereResult:
        uncertainty = self._extract_uncertainty(bridge_packet)
        if not self._should_fuse(uncertainty=uncertainty):
            result = await self._call_reason(
                user_turn=user_turn,
                bridge_packet=bridge_packet,
                memory_context=memory_context,
                runtime_feedback=runtime_feedback,
                attempt=attempt,
                workspace=workspace,
            )
            return self._annotate(
                result,
                method="passthrough",
                uncertainty=uncertainty,
                temperatures=[bridge_packet.control.target_temperature],
                selected_index=0,
            )

        temperatures = self._candidate_temperatures(bridge_packet.control.target_temperature)
        candidates = await asyncio.gather(
            *(
                self._call_reason(
                    user_turn=user_turn,
                    bridge_packet=self._packet_with_temperature(bridge_packet, temp),
                    memory_context=memory_context,
                    runtime_feedback=runtime_feedback,
                    attempt=attempt,
                    workspace=workspace,
                )
                for temp in temperatures
            )
        )

        fusion = self._select_candidate(
            user_turn=user_turn,
            candidates=candidates,
            jepa_prediction=bridge_packet.latent_vector,
            uncertainty=uncertainty,
        )
        return self._annotate(
            fusion.result,
            method=fusion.method,
            uncertainty=uncertainty,
            temperatures=temperatures,
            selected_index=fusion.selected_index,
            score=fusion.score,
            scores=fusion.scores,
        )

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
        try:
            repaired = await maybe_await(
                self.provider.arepair(
                    user_turn,
                    bridge_packet,
                    memory_context,
                    previous_result,
                    rejected_results,
                    attempt,
                    critique_feedback,
                    workspace,
                )
            )
        except TypeError:
            repaired = await maybe_await(
                self.provider.arepair(
                    user_turn,
                    bridge_packet,
                    memory_context,
                    previous_result,
                    rejected_results,
                    attempt,
                    critique_feedback,
                )
            )
        return self._annotate(
            repaired,
            method="repair_passthrough",
            uncertainty=self._extract_uncertainty(bridge_packet),
            temperatures=[bridge_packet.control.target_temperature],
            selected_index=0,
        )

    async def _call_reason(
        self,
        *,
        user_turn: UserTurn,
        bridge_packet: CognitiveBridgePacket,
        memory_context: MemoryContext,
        runtime_feedback: list[str] | None,
        attempt: int,
        workspace: CognitiveWorkspace | None,
    ) -> LeftHemisphereResult:
        try:
            return await maybe_await(
                self.provider.areason(
                    user_turn,
                    bridge_packet,
                    memory_context,
                    runtime_feedback,
                    attempt,
                    workspace,
                )
            )
        except TypeError:
            return await maybe_await(
                self.provider.areason(
                    user_turn,
                    bridge_packet,
                    memory_context,
                    runtime_feedback,
                    attempt,
                )
            )

    def _select_candidate(
        self,
        *,
        user_turn: UserTurn,
        candidates: list[LeftHemisphereResult],
        jepa_prediction: list[float],
        uncertainty: float,
    ) -> FusionResult:
        if self.config.selection_mode == "random":
            rng = random.Random(f"{user_turn.session_id}:{user_turn.turn_id}")
            index = rng.randrange(len(candidates))
            return FusionResult(
                result=candidates[index],
                selected_index=index,
                method="random_control",
            )
        return self.selector.select(candidates, jepa_prediction, uncertainty)

    def _annotate(
        self,
        result: LeftHemisphereResult,
        *,
        method: str,
        uncertainty: float,
        temperatures: list[float],
        selected_index: int,
        score: float | None = None,
        scores: list[float] | None = None,
    ) -> LeftHemisphereResult:
        result.telemetry["fusion_enabled"] = self.config.enabled
        result.telemetry["fusion_selection_mode"] = self.config.selection_mode
        result.telemetry["fusion_method"] = method
        result.telemetry["fusion_triggered"] = method not in {"passthrough", "repair_passthrough"}
        result.telemetry["fusion_candidate_count"] = len(temperatures)
        result.telemetry["fusion_temperatures"] = [round(item, 3) for item in temperatures]
        result.telemetry["fusion_selected_index"] = selected_index
        result.telemetry["fusion_uncertainty"] = round(uncertainty, 3)
        if score is not None:
            result.telemetry["fusion_score"] = score
        if scores is not None:
            result.telemetry["fusion_scores"] = scores
        return result

    def _packet_with_temperature(self, packet: CognitiveBridgePacket, temperature: float) -> CognitiveBridgePacket:
        control = replace(packet.control, target_temperature=round(temperature, 3))
        return replace(packet, control=control)

    def _candidate_temperatures(self, base: float) -> list[float]:
        offsets = self.config.temperature_offsets
        count = max(1, self.config.n_candidates)
        out = [round(max(0.0, min(1.5, base + offsets[idx % len(offsets)])), 3) for idx in range(count)]
        return out

    def _extract_uncertainty(self, bridge_packet: CognitiveBridgePacket) -> float:
        annotations = bridge_packet.control.annotations or {}
        value = annotations.get("jepa_uncertainty")
        if value is None:
            return 1.0
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 1.0

    def _should_fuse(self, *, uncertainty: float) -> bool:
        if not self.config.enabled:
            return False
        if self.config.n_candidates < 2:
            return False
        return uncertainty < self.config.uncertainty_threshold
