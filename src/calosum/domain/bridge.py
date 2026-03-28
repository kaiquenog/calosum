from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from calosum.shared.types import BridgeControlSignal, CognitiveBridgePacket, RightHemisphereState, SoftPromptToken


@dataclass(slots=True)
class CognitiveTokenizerConfig:
    bottleneck_tokens: int = 6
    base_temperature: float = 0.25
    salience_threshold: float = 0.7
    max_directives: int = 4
    weights_path: Path = Path(".calosum-runtime/state/bridge_weights.pt")


class CognitiveTokenizer:
    """
    Traduz o estado latente continuo do JEPA para uma interface discreta.

    O resultado e um pacote com:
    - soft prompts compactos;
    - sinais de controle para o hemisferio esquerdo;
    - metadados suficientes para observabilidade e auditoria.
    """

    def __init__(self, config: CognitiveTokenizerConfig | None = None) -> None:
        self.config = config or CognitiveTokenizerConfig()
        self._init_neural_bridge()

    def _init_neural_bridge(self) -> None:
        """
        Inicializa uma rede PyTorch leve (Information Bottleneck) para 
        traduzir o vetor latente do Hemisfério Direito.
        """
        try:
            import torch
            import torch.nn as nn
            
            # Assume 384 from Sprint 1 (all-MiniLM-L6-v2)
            self.latent_dim = 384 
            # Saída: 1 neurônio para Saliência + N neurônios para Soft Prompts
            self.output_dim = 1 + self.config.bottleneck_tokens
            
            self.projection = nn.Sequential(
                nn.Linear(self.latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_dim),
                nn.Sigmoid() # Normaliza as saídas entre 0 e 1
            )
            
            # Carrega pesos se existirem (para o loop de evolução)
            if self.config.weights_path.exists():
                self.projection.load_state_dict(torch.load(self.config.weights_path, weights_only=True))
                
            self.use_neural = True
        except ImportError:
            self.use_neural = False

    def translate(self, right_state: RightHemisphereState) -> CognitiveBridgePacket:
        if self.use_neural and len(right_state.latent_vector) == self.latent_dim:
            return self._neural_translate(right_state)
        return self._heuristic_translate(right_state)

    def _neural_translate(self, right_state: RightHemisphereState) -> CognitiveBridgePacket:
        import torch
        
        with torch.no_grad():
            tensor_in = torch.tensor(right_state.latent_vector, dtype=torch.float32)
            tensor_out = self.projection(tensor_in).tolist()
            
        # O primeiro valor é a Saliência Aprendida
        neural_salience = round(tensor_out[0], 3)
        # O resto são os pesos dos Soft Prompts
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
        empathy_priority = salience >= self.config.salience_threshold
        directives = [
            "preserve typed reasoning",
            "ground decisions in available memory",
        ]
        if empathy_priority:
            directives.insert(0, "lead with empathy before dense logic")
            directives.append("prefer safe clarification under emotional uncertainty")

        control = BridgeControlSignal(
            target_temperature=round(
                self.config.base_temperature + (0.1 if empathy_priority else 0.0), 2
            ),
            empathy_priority=empathy_priority,
            system_directives=directives[: self.config.max_directives],
            annotations={
                "salience_threshold": self.config.salience_threshold,
                "bottleneck_tokens": self.config.bottleneck_tokens,
                "neural_active": getattr(self, "use_neural", False)
            },
        )

        return CognitiveBridgePacket(
            context_id=right_state.context_id,
            soft_prompts=tokens,
            control=control,
            salience=salience,
            bridge_metadata={
                "emotional_labels": right_state.emotional_labels,
                "confidence": right_state.confidence,
            },
        )

    async def atranslate(self, right_state: RightHemisphereState) -> CognitiveBridgePacket:
        return self.translate(right_state)
