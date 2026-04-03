from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any

from calosum.shared.models.types import MemoryContext, Modality, InputPerceptionState, UserTurn, CognitiveWorkspace


@dataclass(slots=True)
class InputPerceptionJEPAConfig:
    model_name: str = "v-jepa-heuristic"
    latent_size: int = 384
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


class InputPerceptionJEPA:
    """
    Fachada do hemisferio direito (Perception).

    Implementa um JEPA heurístico baseado em embeddings lexicais ou modelos locais
    quando backends pesados não estão disponíveis.
    """

    def __init__(
        self,
        config: InputPerceptionJEPAConfig | None = None,
        vision_adapter: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.config = config or InputPerceptionJEPAConfig()
        self.vision_adapter = vision_adapter
        self.embedder = embedder
        self._salience_history_by_session: dict[str, list[float]] = defaultdict(list)

    def perceive(self, user_turn: UserTurn, memory_context: Any | None = None, workspace: CognitiveWorkspace | None = None) -> InputPerceptionState:
        # Get latent vector using the best available embedder
        latent_vector = self._get_latent_vector(user_turn.user_text)
        
        emotional_labels = self._extract_emotional_labels(user_turn)
        raw_salience = self._estimate_salience(user_turn, emotional_labels)
        
        # Process visual signals if present
        visual_latents: list[float] = []
        if self.vision_adapter and hasattr(self.vision_adapter, "embed_image"):
            for signal in user_turn.signals:
                if signal.modality == Modality.VIDEO and isinstance(signal.payload, bytes):
                    visual_latents.extend(self.vision_adapter.embed_image(signal.payload))

        runtime_feedback_bias = self._runtime_feedback_bias(workspace)
        salience = self._calibrate_salience(user_turn.session_id, min(1.0, raw_salience + runtime_feedback_bias))
        
        # Surprise is calculated via Cosine Distance intent
        surprise_score = self._calculate_surprise(latent_vector, memory_context)

        world_hypotheses = {
            "interaction_complexity": min(1.0, len(user_turn.user_text) / 240.0),
            "sensor_diversity": min(1.0, len(user_turn.signals) / 6.0),
            "visual_richness": min(1.0, len(visual_latents) / 1024.0),
            "urgency": salience,
            "semantic_density": self._semantic_density(latent_vector),
            "operational_risk": runtime_feedback_bias,
        }

        state = InputPerceptionState(
            context_id=user_turn.turn_id,
            latent_vector=latent_vector,
            salience=salience,
            emotional_labels=emotional_labels or ["neutral"],
            world_hypotheses=world_hypotheses,
            confidence=0.85 if self.embedder else 0.45,
            surprise_score=surprise_score,
            telemetry={
                "model_name": self.config.model_name,
                "right_backend": "heuristic_jepa_v2",
                "right_model_name": self.config.model_name,
                "right_mode": "lexical_enhanced" if not self.embedder else "embedded",
                "degraded_reason": None if self.embedder else "no_active_embedder_fallback_to_lexical",
                "modalities_seen": [signal.modality.value for signal in user_turn.signals],
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

    async def aperceive(self, user_turn: UserTurn, memory_context: Any | None = None, workspace: CognitiveWorkspace | None = None) -> InputPerceptionState:
        return self.perceive(user_turn, memory_context, workspace)

    def _get_latent_vector(self, text: str) -> list[float]:
        if self.embedder and hasattr(self.embedder, "embed_texts"):
            try:
                vectors = self.embedder.embed_texts([text])
                return vectors[0]
            except Exception:
                pass
        return self._lexical_vector(text)

    def _lexical_vector(self, text: str) -> list[float]:
        tokens = re.findall(r"\w+", text.lower()) or ["silence"]
        vector = [0.0] * self.config.latent_size
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(0, len(digest), 4):
                chunk = digest[index:index + 4]
                position = int.from_bytes(chunk[:2], "big") % self.config.latent_size
                sign = 1.0 if chunk[2] % 2 == 0 else -1.0
                magnitude = 1.0 + (chunk[3] / 255.0)
                vector[position] += sign * magnitude
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0: return vector
        return [round(v / norm, 6) for v in vector]

    def _calculate_surprise(self, latent_vector: list[float], memory_context: Any | None) -> float:
        if not memory_context or not memory_context.recent_episodes:
            return 0.2  # default baseline surprise
        
        recent_vectors = [
            ep.right_state.latent_vector 
            for ep in memory_context.recent_episodes 
            if len(ep.right_state.latent_vector) == len(latent_vector)
        ]
        if not recent_vectors:
            return 0.2
            
        avg_vector = [sum(x)/len(recent_vectors) for x in zip(*recent_vectors)]
        
        dot_product = sum(a*b for a, b in zip(latent_vector, avg_vector))
        mag_a = math.sqrt(sum(a*a for a in latent_vector))
        mag_b = math.sqrt(sum(b*b for b in avg_vector))
        
        if mag_a == 0 or mag_b == 0:
            return 0.5
            
        cosine_similarity = dot_product / (mag_a * mag_b)
        # We use (1 - sim) / 2 to map to [0, 1]
        distance = (1.0 - cosine_similarity) / 2.0
        
        return round(distance, 3)

    def _semantic_density(self, vector: list[float]) -> float:
        if not vector: return 0.0
        mean_abs = sum(abs(v) for v in vector) / len(vector)
        return round(max(0.0, min(1.0, mean_abs * 2.0)), 3)

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
