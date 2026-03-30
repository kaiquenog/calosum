from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any

from calosum.shared.types import MemoryContext, Modality, RightHemisphereState, UserTurn, CognitiveWorkspace


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
            "raiva": 0.9,
            "medo": 0.9,
            "preocupado": 0.8,
            "dor": 0.9,
            "desespero": 0.95
        }
    )
    salience_window_size: int = 6
    salience_smoothing_alpha: float = 0.5
    salience_max_step: float = 0.22


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
        self._salience_history_by_session: dict[str, list[float]] = defaultdict(list)

    def perceive(self, user_turn: UserTurn, memory_context: Any | None = None, workspace: CognitiveWorkspace | None = None) -> RightHemisphereState:
        seed = self._build_seed(user_turn)
        latent_vector = self._latent_from_seed(seed, self.config.latent_size)
        emotional_labels = self._extract_emotional_labels(user_turn)
        raw_salience = self._estimate_salience(user_turn, emotional_labels)
        runtime_feedback_bias = self._runtime_feedback_bias(workspace)
        salience = self._calibrate_salience(user_turn.session_id, min(1.0, raw_salience + runtime_feedback_bias))
        world_hypotheses = {
            "interaction_complexity": min(1.0, len(user_turn.user_text) / 240.0),
            "sensor_diversity": min(1.0, len(user_turn.signals) / 6.0),
            "urgency": salience,
            "semantic_density": 0.5,  # Placeholder for V2 latents density
            "operational_risk": runtime_feedback_bias,
        }

        # V2 Vision: Surprise is calculated via KL-Divergence between predicted and actual latents.
        # In the domain layer, we represent the intent; the adapter implements the math.
        surprise_score = self._calculate_surprise(latent_vector, memory_context)

        state = RightHemisphereState(
            context_id=user_turn.turn_id,
            latent_vector=latent_vector,
            salience=salience,
            emotional_labels=emotional_labels or ["neutral"],
            world_hypotheses=world_hypotheses,
            confidence=0.72,
            surprise_score=surprise_score,
            telemetry={
                "model_name": self.config.model_name,
                "right_backend": "heuristic_jepa",
                "right_model_name": self.config.model_name,
                "right_mode": "heuristic",
                "degraded_reason": None,
                "modalities_seen": [signal.modality.value for signal in user_turn.signals],
                "seed_fingerprint": seed[:12],
                "raw_salience": raw_salience,
                "runtime_feedback_bias": runtime_feedback_bias,
            },
        )
        
        if workspace is not None:
            workspace.right_notes.update({
                "salience": salience,
                "raw_salience": raw_salience,
                "runtime_feedback_bias": runtime_feedback_bias,
                "surprise_score": surprise_score,
                "emotional_labels": emotional_labels or ["neutral"],
            })
            
        return state

    async def aperceive(self, user_turn: UserTurn, memory_context: Any | None = None, workspace: CognitiveWorkspace | None = None) -> RightHemisphereState:
        return self.perceive(user_turn, memory_context, workspace)

    def _calculate_surprise(self, latent_vector: list[float], memory_context: Any | None) -> float:
        if not memory_context or not memory_context.recent_episodes:
            return 0.5  # default surprise if no memory
        
        # Calculate cosine distance to average of recent memories
        try:
            import math
            recent_vectors = [ep.right_state.latent_vector for ep in memory_context.recent_episodes if len(ep.right_state.latent_vector) == len(latent_vector)]
            if not recent_vectors:
                return 0.5
                
            avg_vector = [sum(x)/len(recent_vectors) for x in zip(*recent_vectors)]
            
            dot_product = sum(a*b for a, b in zip(latent_vector, avg_vector))
            mag_a = math.sqrt(sum(a*a for a in latent_vector))
            mag_b = math.sqrt(sum(b*b for b in avg_vector))
            
            if mag_a == 0 or mag_b == 0:
                return 0.5
                
            cosine_similarity = dot_product / (mag_a * mag_b)
            # Distance is 1 - similarity. So max surprise is when similarity is -1 (distance 2)
            distance = 1.0 - cosine_similarity
            
            # Normalize to 0-1 range (distance is 0 to 2)
            return round(distance / 2.0, 3)
        except Exception:
            return 0.5

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

    def _calibrate_salience(self, session_id: str, raw_salience: float) -> float:
        history = self._salience_history_by_session[session_id]
        if not history:
            history.append(raw_salience)
            return round(raw_salience, 3)

        moving_avg = sum(history) / len(history)
        alpha = min(1.0, max(0.0, self.config.salience_smoothing_alpha))
        blended = (alpha * raw_salience) + ((1.0 - alpha) * moving_avg)

        previous = history[-1]
        max_step = max(0.01, self.config.salience_max_step)
        lower_bound = max(0.0, previous - max_step)
        upper_bound = min(1.0, previous + max_step)
        calibrated = min(upper_bound, max(lower_bound, blended))

        history.append(calibrated)
        max_window = max(2, self.config.salience_window_size)
        if len(history) > max_window:
            del history[:-max_window]
        return round(calibrated, 3)

    def _runtime_feedback_bias(self, workspace: CognitiveWorkspace | None) -> float:
        if workspace is None:
            return 0.0
        previous_feedback = workspace.task_frame.get("previous_runtime_feedback", [])
        if not isinstance(previous_feedback, list) or not previous_feedback:
            return 0.0
        rejected = sum(int(item.get("rejected_count", 0)) for item in previous_feedback if isinstance(item, dict))
        executed = sum(int(item.get("executed_count", 0)) for item in previous_feedback if isinstance(item, dict))
        attempts = max(1, len(previous_feedback))
        rejection_rate = rejected / max(1, rejected + executed)
        intensity = min(0.15, (rejection_rate * 0.12) + (attempts * 0.01))
        return round(max(0.0, intensity), 3)
