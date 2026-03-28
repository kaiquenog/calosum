from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .types import Modality, RightHemisphereState, UserTurn


@dataclass(slots=True)
class RightHemisphereJEPAConfig:
    model_name: str = "v-jepa-placeholder"
    latent_size: int = 16
    salience_keywords: dict[str, float] = field(
        default_factory=lambda: {
            "urgente": 1.0,
            "emergencia": 1.0,
            "triste": 0.85,
            "ansioso": 0.75,
            "feliz": 0.35,
            "frustrado": 0.8,
        }
    )


class RightHemisphereJEPA:
    """
    Fachada do hemisferio direito.

    Em producao, esta classe encapsularia um backbone V-JEPA/I-JEPA em PyTorch
    e operaria sobre fluxos multimodais para prever estados latentes ocultos.
    Neste esqueleto, a saida e sintetica e deterministica para documentar a
    interface de comunicacao com os demais modulos.
    """

    def __init__(self, config: RightHemisphereJEPAConfig | None = None) -> None:
        self.config = config or RightHemisphereJEPAConfig()

    def perceive(self, user_turn: UserTurn) -> RightHemisphereState:
        seed = self._build_seed(user_turn)
        latent_vector = self._latent_from_seed(seed, self.config.latent_size)
        emotional_labels = self._extract_emotional_labels(user_turn)
        salience = self._estimate_salience(user_turn, emotional_labels)
        world_hypotheses = {
            "interaction_complexity": min(1.0, len(user_turn.user_text) / 240.0),
            "sensor_diversity": min(1.0, len(user_turn.signals) / 6.0),
            "urgency": salience,
        }

        return RightHemisphereState(
            context_id=user_turn.turn_id,
            latent_vector=latent_vector,
            salience=salience,
            emotional_labels=emotional_labels or ["neutral"],
            world_hypotheses=world_hypotheses,
            confidence=0.72,
            telemetry={
                "model_name": self.config.model_name,
                "modalities_seen": [signal.modality.value for signal in user_turn.signals],
                "seed_fingerprint": seed[:12],
            },
        )

    def _build_seed(self, user_turn: UserTurn) -> str:
        raw_payload = "|".join(
            f"{signal.modality}:{signal.source}:{str(signal.payload)[:64]}:{signal.metadata}"
            for signal in user_turn.signals
        )
        return hashlib.sha256(
            f"{user_turn.session_id}|{user_turn.user_text}|{raw_payload}".encode("utf-8")
        ).hexdigest()

    def _latent_from_seed(self, seed: str, latent_size: int) -> list[float]:
        digest = bytes.fromhex(seed)
        vector: list[float] = []
        for index in range(latent_size):
            byte = digest[index % len(digest)]
            vector.append(round((byte / 255.0) * 2.0 - 1.0, 4))
        return vector

    def _extract_emotional_labels(self, user_turn: UserTurn) -> list[str]:
        labels: list[str] = []
        text = user_turn.user_text.lower()
        for keyword in self.config.salience_keywords:
            if keyword in text:
                labels.append(keyword)
        for signal in user_turn.signals:
            if signal.modality in {Modality.AUDIO, Modality.VIDEO, Modality.TYPING}:
                emotion = signal.metadata.get("emotion")
                if isinstance(emotion, str):
                    labels.append(emotion.lower())
        return sorted(set(labels))

    def _estimate_salience(self, user_turn: UserTurn, emotional_labels: list[str]) -> float:
        text = user_turn.user_text.lower()
        salience = 0.15 + min(0.25, len(user_turn.signals) * 0.05)
        for label in emotional_labels:
            salience = max(salience, self.config.salience_keywords.get(label, 0.45))
        if "!" in text:
            salience = min(1.0, salience + 0.1)
        return round(min(1.0, salience), 3)
