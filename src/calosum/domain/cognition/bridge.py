from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from calosum.shared.models.types import BridgeControlSignal, CognitiveBridgePacket, RightHemisphereState, SoftPromptToken, CognitiveWorkspace
from calosum.shared.models.ports import BridgeFusionPort, BridgeStateStorePort


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


class ContextCompressor:
    """
    Traduz o estado latente continuo do JEPA para uma interface discreta.

    O resultado e um pacote com:
    - context directives compactas;
    - sinais de controle para o hemisferio esquerdo;
    - metadados suficientes para observabilidade e auditoria.
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
        self._init_neural_bridge()

    def _init_neural_bridge(self) -> None:
        """
        Inicializa uma rede PyTorch leve (Information Bottleneck) para 
        traduzir o vetor latente do Hemisfério Direito.
        """
        try:
            import torch
            import torch.nn as nn
            
            # Target dimension dinâmico via configuração
            self.latent_dim = self.config.target_dim
            # Saída: 1 neurônio para Saliência + N neurônios para Context Directives
            self.output_dim = 1 + self.config.bottleneck_tokens
            
            self.projection = nn.Sequential(
                nn.Linear(self.latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_dim),
                nn.Sigmoid() # Normaliza as saídas entre 0 e 1
            )
            
            # Carrega pesos se existirem (para o loop de evolução)
            if self.store:
                self.store.load_weights(self.projection)
                
            self.use_neural = True
        except ImportError:
            self.use_neural = False

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
        }
        self.store.persist_adaptation_state(payload)

    def record_reflection_event(self, payload: dict[str, Any]) -> None:
        if self.store:
            self.store.record_reflection_event(payload)

    def translate(self, right_state: RightHemisphereState, workspace: CognitiveWorkspace | None = None) -> CognitiveBridgePacket:
        state = self._apply_fusion(right_state)
        if self.use_neural and len(state.latent_vector) == self.latent_dim:
            packet = self._neural_translate(state)
        else:
            packet = self._heuristic_translate(state)
            
        if workspace is not None:
            workspace.bridge_state.update({
                "target_temperature": packet.control.target_temperature,
                "empathy_priority": packet.control.empathy_priority,
                "directives": packet.control.system_directives,
                "salience_calibrated": packet.salience,
            })
            
        return packet

    def _apply_fusion(self, right_state: RightHemisphereState) -> RightHemisphereState:
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
            return RightHemisphereState(
                context_id=right_state.context_id,
                latent_vector=fused_latent,
                salience=right_state.salience,
                emotional_labels=right_state.emotional_labels,
                world_hypotheses=right_state.world_hypotheses,
                confidence=right_state.confidence,
                surprise_score=right_state.surprise_score,
                telemetry=telemetry,
            )
        except Exception:
            return right_state

    def _neural_translate(self, right_state: RightHemisphereState) -> CognitiveBridgePacket:
        import torch
        
        with torch.no_grad():
            tensor_in = torch.tensor(right_state.latent_vector, dtype=torch.float32)
            tensor_out = self.projection(tensor_in).tolist()
            
        # O primeiro valor é a Saliência Aprendida
        neural_salience = round(tensor_out[0], 3)
        # O resto são os pesos das Context Directives
        neural_weights = tensor_out[1:]
        
        tokens: list[SoftPromptToken] = []
        for index, label in enumerate(right_state.emotional_labels[: self.config.bottleneck_tokens]):
            weight = neural_weights[index] if index < len(neural_weights) else 0.0
            tokens.append(
                SoftPromptToken(
                    token=f"<affect:{label}>",
                    weight=round(weight, 3),
                    provenance="neural_bridge"
                )
            )
            
        return self._build_packet(right_state, tokens, neural_salience)

    def _heuristic_translate(self, right_state: RightHemisphereState) -> CognitiveBridgePacket:
        tokens: list[SoftPromptToken] = []
        for label in right_state.emotional_labels[: self.config.bottleneck_tokens]:
            tokens.append(
                SoftPromptToken(
                    token=f"<affect:{label}>",
                    weight=round(right_state.salience, 3),
                    provenance="jepa.emotional_labels",
                )
            )

        remaining = self.config.bottleneck_tokens - len(tokens)
        for index, value in enumerate(right_state.latent_vector[:remaining]):
            tokens.append(
                SoftPromptToken(
                    token=f"<latent:{index}>",
                    weight=round(abs(value), 3),
                    provenance="jepa.latent_projection",
                )
            )
        return self._build_packet(right_state, tokens, right_state.salience)

    def _build_packet(self, right_state: RightHemisphereState, tokens: list[SoftPromptToken], salience: float) -> CognitiveBridgePacket:
        calibrated_salience = round(
            min(1.0, max(0.0, salience * self.config.salience_gain + self.config.salience_bias)),
            3,
        )
        empathy_priority = calibrated_salience >= self.config.salience_threshold
        directives = [
            "preserve typed reasoning",
            "ground decisions in available memory",
        ]
        if empathy_priority:
            directives.insert(0, "lead with empathy before dense logic")
            directives.append("prefer safe clarification under emotional uncertainty")

        # Active Inference: Epistemic Foraging
        surprise_score = getattr(right_state, "surprise_score", 0.0)
        ambiguity_score = getattr(right_state, "world_hypotheses", {}).get("interaction_complexity", 0.0)
        if surprise_score >= 0.3 or ambiguity_score >= 0.5:
            directives.insert(0, "HIGH SURPRISE DETECTED: you MUST prioritize epistemic foraging using tools (search_web, execute_bash, read_file, introspect_self) to gather context and reduce uncertainty BEFORE providing a final answer.")

        # Modulação Dinâmica de Temperatura via Surpresa
        # Entradas surpreendentes (alto surprise_score) baixam a temperatura para forçar foco e analítica
        surprise_penalty = surprise_score * 0.25

        control = BridgeControlSignal(
            target_temperature=round(
                min(
                    1.0,
                    max(
                        0.05,
                        self.config.base_temperature
                        + (0.1 if empathy_priority else 0.0)
                        - surprise_penalty
                        + self.config.temperature_bias,
                    ),
                ),
                2,
            ),
            empathy_priority=empathy_priority,
            system_directives=directives[: self.config.max_directives],
            annotations={
                "salience_threshold": self.config.salience_threshold,
                "bottleneck_tokens": self.config.bottleneck_tokens,
                "neural_active": getattr(self, "use_neural", False),
                "raw_salience": salience,
                "calibrated_salience": calibrated_salience,
                "surprise_penalty": surprise_penalty,
                "salience_gain": self.config.salience_gain,
                "salience_bias": self.config.salience_bias,
                "temperature_bias": self.config.temperature_bias,
                "jepa_uncertainty": float(right_state.telemetry.get("jepa_uncertainty", 1.0)),
            },
        )

        return CognitiveBridgePacket(
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

    async def atranslate(self, right_state: RightHemisphereState, workspace: CognitiveWorkspace | None = None) -> CognitiveBridgePacket:
        return self.translate(right_state, workspace)

    def train_step(self, latent_vector: list[float], target_salience: float, learning_rate: float = 0.01) -> dict[str, Any]:
        """Train the neural bridge projection using reflection outcomes as signal.

        The target_salience comes from the GEA reflection controller's evaluation
        of which variant performed best — providing a supervised signal for the bridge.
        """
        if not self.use_neural or len(latent_vector) != self.latent_dim:
            return {"trained": False, "reason": "neural bridge not active or dim mismatch"}
        try:
            import torch
            optimizer = torch.optim.SGD(self.projection.parameters(), lr=learning_rate)
            tensor_in = torch.tensor(latent_vector, dtype=torch.float32)
            output = self.projection(tensor_in)
            predicted_salience = output[0]
            target = torch.tensor(target_salience, dtype=torch.float32)
            loss = torch.nn.functional.mse_loss(predicted_salience, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return {"trained": True, "loss": round(float(loss.item()), 6)}
        except Exception as exc:
            return {"trained": False, "reason": str(exc)}

    def get_bridge_parameters(self) -> list[Any]:
        """Return trainable parameters from both the projection and the fusion adapter."""
        params: list[Any] = []
        if self.use_neural:
            params.extend(self.projection.parameters())
        if self.fusion is not None and hasattr(self.fusion, "get_parameters"):
            params.extend(self.fusion.get_parameters())
        return params


CognitiveTokenizerConfig = ContextCompressorConfig
CognitiveTokenizer = ContextCompressor
