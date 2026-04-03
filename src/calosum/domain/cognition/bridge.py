from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from calosum.shared.models.ports import BridgeFusionPort, BridgeStateStorePort
from calosum.shared.models.types import (
    BridgeControlSignal,
    CognitiveWorkspace,
    InputPerceptionState,
    PerceptionSummary,
    SoftPromptToken,
)


@dataclass(slots=True)
class ContextCompressorConfig:
    target_dim: int = 384
    bottleneck_tokens: int = 6
    base_temperature: float = 0.25
    salience_threshold: float = 0.7
    max_directives: int = 4
    salience_gain: float = 1.0
    salience_bias: float = 0.0
    temperature_bias: float = 0.0
    top_p_floor: float = 0.35
    top_p_ceiling: float = 0.98
    logit_bias_gain: float = 1.8
    surprise_guardrail: float = 0.6


class ContextCompressor:
    """
    Traduz o estado latente em sinais contínuos de controle para o hemisfério esquerdo.
    """

    def __init__(
        self,
        config: ContextCompressorConfig | None = None,
        store: BridgeStateStorePort | None = None,
        fusion: BridgeFusionPort | None = None,
    ) -> None:
        self.config = config or ContextCompressorConfig()
        self.store = store
        self.fusion = fusion
        self._load_adaptation_state()

    def _load_adaptation_state(self) -> None:
        if not self.store:
            return

        data = self.store.load_adaptation_state()
        if not data:
            return

        for key in (
            "target_dim",
            "bottleneck_tokens",
            "base_temperature",
            "salience_threshold",
            "max_directives",
            "salience_gain",
            "salience_bias",
            "temperature_bias",
            "top_p_floor",
            "top_p_ceiling",
            "logit_bias_gain",
            "surprise_guardrail",
        ):
            if key in data and hasattr(self.config, key):
                setattr(self.config, key, data[key])

    def persist_adaptation_state(self) -> None:
        if not self.store:
            return
            
        payload = {
            "target_dim": self.config.target_dim,
            "bottleneck_tokens": self.config.bottleneck_tokens,
            "base_temperature": self.config.base_temperature,
            "salience_threshold": self.config.salience_threshold,
            "max_directives": self.config.max_directives,
            "salience_gain": self.config.salience_gain,
            "salience_bias": self.config.salience_bias,
            "temperature_bias": self.config.temperature_bias,
            "top_p_floor": self.config.top_p_floor,
            "top_p_ceiling": self.config.top_p_ceiling,
            "logit_bias_gain": self.config.logit_bias_gain,
            "surprise_guardrail": self.config.surprise_guardrail,
        }
        self.store.persist_adaptation_state(payload)

    def record_reflection_event(self, payload: dict[str, Any]) -> None:
        if self.store:
            self.store.record_reflection_event(payload)

    def translate(self, right_state: InputPerceptionState, workspace: CognitiveWorkspace | None = None) -> PerceptionSummary:
        state = self._apply_fusion(right_state)
        
        if workspace and workspace.runtime_feedback:
            last_feedback = workspace.runtime_feedback[-1]
            if last_feedback.get("rejected_count", 0) > 0:
                state = replace(
                    state,
                    surprise_score=min(1.0, state.surprise_score + 0.2),
                    salience=min(1.0, state.salience + 0.1),
                    telemetry={**state.telemetry, "runtime_backpressure": True},
                )
        packet = self._latent_translate(state)
            
        if workspace is not None:
            workspace.bridge_state.update({
                "target_temperature": packet.control.target_temperature,
                "empathy_priority": packet.control.empathy_priority,
                "directives": packet.control.system_directives,
                "target_top_p": packet.control.annotations.get("target_top_p"),
                "salience_calibrated": packet.salience,
            })
            
        return packet

    def _apply_fusion(self, right_state: InputPerceptionState) -> InputPerceptionState:
        if self.fusion is None:
            return right_state
        try:
            fused_latent, meta = self.fusion.fuse_latent(
                latent_vector=right_state.latent_vector,
                emotional_labels=right_state.emotional_labels,
                surprise=right_state.surprise_score,
                confidence=right_state.confidence,
                context_novelty=getattr(right_state, "context_novelty", 0.0),
            )
            telemetry = dict(right_state.telemetry)
            telemetry.update(meta)
            return InputPerceptionState(
                context_id=right_state.context_id,
                latent_vector=fused_latent,
                salience=right_state.salience,
                emotional_labels=right_state.emotional_labels,
                world_hypotheses=right_state.world_hypotheses,
                confidence=right_state.confidence,
                surprise_score=right_state.surprise_score,
                latent_mu=right_state.latent_mu,
                latent_logvar=right_state.latent_logvar,
                telemetry=telemetry,
            )
        except Exception:
            return right_state

    def _latent_translate(self, right_state: InputPerceptionState) -> PerceptionSummary:
        tokens = self._top_latent_tokens(right_state.latent_vector)
        for label in right_state.emotional_labels[: max(1, self.config.bottleneck_tokens // 3)]:
            tokens.append(
                SoftPromptToken(
                    token=f"<affect:{label}>",
                    weight=round(right_state.salience, 3),
                    provenance="latent_projection_interpreter",
                )
            )
        return self._build_packet(right_state, tokens[: self.config.bottleneck_tokens], right_state.salience)

    def _build_packet(self, right_state: InputPerceptionState, tokens: list[SoftPromptToken], salience: float) -> PerceptionSummary:
        calibrated_salience = round(
            min(1.0, max(0.0, salience * self.config.salience_gain + self.config.salience_bias)),
            3,
        )
        empathy_priority = calibrated_salience >= self.config.salience_threshold
        uncertainty = float(right_state.telemetry.get("jepa_uncertainty", 1.0))
        surprise_score = float(getattr(right_state, "surprise_score", 0.0))
        novelty = float(right_state.world_hypotheses.get("context_novelty", 0.0))

        control_temperature = self._target_temperature(
            salience=calibrated_salience,
            surprise=surprise_score,
            uncertainty=uncertainty,
        )
        target_top_p = self._target_top_p(
            salience=calibrated_salience,
            surprise=surprise_score,
            uncertainty=uncertainty,
            novelty=novelty,
        )
        logit_bias = self._target_logit_bias(
            salience=calibrated_salience,
            surprise=surprise_score,
            uncertainty=uncertainty,
        )
        directives = [
            "preserve typed reasoning",
            "ground decisions in available memory",
        ]
        if empathy_priority:
            directives.insert(0, "lead with empathy before dense logic")
            directives.append("prefer safe clarification under emotional uncertainty")
        ambiguity_score = float(right_state.world_hypotheses.get("interaction_complexity", 0.0))
        if surprise_score >= self.config.surprise_guardrail or ambiguity_score >= 0.5:
            directives.insert(
                0,
                "high uncertainty: prioritize epistemic foraging with tools before final response",
            )

        control = BridgeControlSignal(
            target_temperature=control_temperature,
            empathy_priority=empathy_priority,
            system_directives=directives[: self.config.max_directives],
            annotations={
                "salience_threshold": self.config.salience_threshold,
                "bottleneck_tokens": self.config.bottleneck_tokens,
                "bridge_mode": "latent_projection_interpreter",
                "raw_salience": salience,
                "calibrated_salience": calibrated_salience,
                "target_top_p": target_top_p,
                "target_logit_bias": logit_bias,
                "salience_gain": self.config.salience_gain,
                "salience_bias": self.config.salience_bias,
                "temperature_bias": self.config.temperature_bias,
                "jepa_uncertainty": uncertainty,
                "context_novelty": novelty,
            },
        )

        return PerceptionSummary(
            context_id=right_state.context_id,
            latent_vector=right_state.latent_vector,
            soft_prompts=tokens,
            control=control,
            salience=calibrated_salience,
            bridge_metadata={
                "emotional_labels": right_state.emotional_labels,
                "confidence": right_state.confidence,
                "raw_salience": salience,
            },
        )

    async def atranslate(self, right_state: InputPerceptionState, workspace: CognitiveWorkspace | None = None) -> PerceptionSummary:
        return self.translate(right_state, workspace)

    def train_step(self, latent_vector: list[float], target_salience: float, learning_rate: float = 0.01) -> dict[str, Any]:
        return {"trained": False, "reason": "bridge_control_is_analytic"}

    def get_bridge_parameters(self) -> list[Any]:
        """Return trainable parameters from the fusion adapter only."""
        params: list[Any] = []
        if self.fusion is not None and hasattr(self.fusion, "get_parameters"):
            params.extend(self.fusion.get_parameters())
        return params

    def _top_latent_tokens(self, latent_vector: list[float]) -> list[SoftPromptToken]:
        weighted = [(idx, value, abs(value)) for idx, value in enumerate(latent_vector)]
        weighted.sort(key=lambda item: item[2], reverse=True)
        tokens: list[SoftPromptToken] = []
        for idx, value, mag in weighted[: self.config.bottleneck_tokens]:
            tokens.append(
                SoftPromptToken(
                    token=f"<latent_dim:{idx}>",
                    weight=round(min(1.0, mag), 3),
                    provenance=f"latent_projection_interpreter:v={round(value, 4)}",
                )
            )
        return tokens

    def _target_temperature(self, *, salience: float, surprise: float, uncertainty: float) -> float:
        pressure = (0.50 * surprise) + (0.35 * uncertainty) + (0.15 * salience)
        value = self.config.base_temperature + self.config.temperature_bias - (0.28 * pressure) + (0.06 * salience)
        return round(min(1.0, max(0.05, value)), 2)

    def _target_top_p(self, *, salience: float, surprise: float, uncertainty: float, novelty: float) -> float:
        exploration = (0.35 * novelty) + (0.30 * salience) - (0.25 * uncertainty) - (0.10 * surprise)
        base = self.config.top_p_floor + (self.config.top_p_ceiling - self.config.top_p_floor) * (1.0 / (1.0 + pow(2.71828, -4.0 * exploration)))
        return round(min(self.config.top_p_ceiling, max(self.config.top_p_floor, base)), 3)

    def _target_logit_bias(self, *, salience: float, surprise: float, uncertainty: float) -> dict[str, float]:
        caution = min(1.0, max(0.0, (0.55 * uncertainty) + (0.45 * surprise)))
        clarify_bias = round(self.config.logit_bias_gain * caution, 3)
        concise_bias = round(max(0.0, 0.6 - salience) * 0.8, 3)
        return {"clarify_first": clarify_bias, "concise_steps": concise_bias}


CognitiveTokenizerConfig = ContextCompressorConfig
CognitiveTokenizer = ContextCompressor
