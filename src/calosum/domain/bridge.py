from __future__ import annotations

from dataclasses import dataclass, field

from calosum.shared.types import BridgeControlSignal, CognitiveBridgePacket, RightHemisphereState, SoftPromptToken


@dataclass(slots=True)
class CognitiveTokenizerConfig:
    bottleneck_tokens: int = 6
    base_temperature: float = 0.25
    salience_threshold: float = 0.7
    max_directives: int = 4


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

    def translate(self, right_state: RightHemisphereState) -> CognitiveBridgePacket:
        tokens = self._build_soft_prompts(right_state)
        empathy_priority = right_state.salience >= self.config.salience_threshold
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
            },
        )

        return CognitiveBridgePacket(
            context_id=right_state.context_id,
            soft_prompts=tokens,
            control=control,
            salience=right_state.salience,
            bridge_metadata={
                "emotional_labels": right_state.emotional_labels,
                "confidence": right_state.confidence,
            },
        )

    async def atranslate(self, right_state: RightHemisphereState) -> CognitiveBridgePacket:
        return self.translate(right_state)

    def _build_soft_prompts(self, right_state: RightHemisphereState) -> list[SoftPromptToken]:
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

        return tokens
