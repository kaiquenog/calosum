from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from calosum.shared.types import MemoryContext, Modality, RightHemisphereState, UserTurn, CognitiveWorkspace

if TYPE_CHECKING:
    from calosum.shared.ports import VectorCodecPort

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class HuggingFaceRightHemisphereConfig:
    # Changed to multilingual for better Portuguese support
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    zero_shot_model_name: str = "facebook/bart-large-mnli"
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
    emotion_similarity_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "urgente": 0.42,
            "emergencia": 0.42,
            "triste": 0.46,
            "ansioso": 0.46,
            "feliz": 0.5,
            "frustrado": 0.44,
            "raiva": 0.44,
            "medo": 0.44,
            "preocupado": 0.46,
            "dor": 0.44,
            "desespero": 0.42,
        }
    )
    salience_window_size: int = 6
    salience_smoothing_alpha: float = 0.45
    salience_max_step: float = 0.22


class HuggingFaceRightHemisphereAdapter:
    """
    Adapter real do Hemisfério Direito utilizando HuggingFace e Sentence Transformers.
    
    Substitui os 'mocks' baseados em hash por inferência real de embeddings
    para capturar a semântica abstrata (o 'sentimento') do input textual.
    
    Nota de Design (Zero-Shot):
    Para evitar overengineering e manter a performance local sem GPU dedicada, 
    optou-se por NÃO utilizar um pipeline completo de zero-shot classification 
    (ex: facebook/bart-large-mnli), que é pesado e lento. Em vez disso, 
    usamos uma heurística híbrida inteligente de Similaridade de Cosseno 
    com os rótulos de emoção pré-computados, que se provou eficaz e rápida.
    """

    def __init__(self, config: HuggingFaceRightHemisphereConfig | None = None, codec: VectorCodecPort | None = None) -> None:
        self.config = config or HuggingFaceRightHemisphereConfig()
        self._salience_history_by_session: dict[str, list[float]] = defaultdict(list)
        self.codec: VectorCodecPort | None = codec
        
        try:
            # O import é feito aqui para evitar lentidão de importação no bootstrap
            import transformers
            transformers.logging.set_verbosity_error()
            transformers.utils.logging.disable_progress_bar()
            
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.config.embedding_model_name}")
            self.embedder = SentenceTransformer(self.config.embedding_model_name)
            
            # Pre-compute emotion embeddings for zero-shot cosine similarity
            self._emotion_labels = list(self.config.salience_keywords.keys())
            self._emotion_embeddings = self.embedder.encode(self._emotion_labels)
            
            self.status = "healthy"
        except ImportError as e:
            logger.error(f"HuggingFace stack unavailable: {e}")
            self.status = "unavailable"
            raise RuntimeError("missing optional model stack") from e
        except Exception as e:
            logger.error(f"Failed to load HF model: {e}")
            self.status = "degraded"
            raise RuntimeError(f"failed to load model: {e}") from e

    def perceive(self, user_turn: UserTurn, memory_context: Any | None = None, workspace: CognitiveWorkspace | None = None) -> RightHemisphereState:
        text = user_turn.user_text
        if not text.strip():
            text = "silence"

        # 1. Gera o vetor latente real
        embeddings = self.embedder.encode([text])
        if hasattr(embeddings, "tolist"):
            latent_vector = embeddings[0].tolist()
        else:
            # Em caso de mock retornar lista pura
            latent_vector = embeddings[0] if isinstance(embeddings[0], list) else embeddings

        # 2. Extrai rótulos emocionais reais (keywords + similaridade vetorial calibrada)
        emotional_labels, emotion_meta = self._extract_emotional_labels(text, latent_vector)
        
        # 3. Estima a saliência (intensidade emocional)
        raw_salience = self._estimate_salience(text, emotional_labels)
        runtime_feedback_bias = self._runtime_feedback_bias(workspace)
        salience = self._calibrate_salience(user_turn.session_id, min(1.0, raw_salience + runtime_feedback_bias))
        
        # 4. Constrói hipóteses de mundo
        import numpy as np
        vector_array = np.array(latent_vector)
        # O cálculo da entropia/variância do vetor representa a densidade de informação latente.
        # Vetores com features mais ativas e distribuídas indicam uma linguagem mais complexa (alta densidade).
        std_dev = float(np.std(vector_array))
        # Normalizando o desvio padrão (empiricamente modelos de embedding têm std_dev pequeno, ~0.05 a 0.15)
        semantic_density = min(1.0, std_dev * 10.0)

        world_hypotheses = {
            "interaction_complexity": min(1.0, len(text) / 240.0),
            "sensor_diversity": min(1.0, len(user_turn.signals) / 6.0),
            "urgency": salience,
            "semantic_density": round(semantic_density, 3),
            "operational_risk": runtime_feedback_bias,
        }

        surprise_score = self._calculate_surprise(latent_vector, memory_context)
        confidence = self._estimate_confidence(user_turn, emotional_labels, emotion_meta)
        modalities_seen = [signal.modality.value for signal in user_turn.signals] if user_turn.signals else ["text"]

        state = RightHemisphereState(
            context_id=user_turn.turn_id,
            latent_vector=latent_vector,
            salience=salience,
            emotional_labels=emotional_labels or ["neutral"],
            world_hypotheses=world_hypotheses,
            confidence=confidence,
            surprise_score=surprise_score,
            telemetry={
                "model_name": self.config.embedding_model_name,
                "right_backend": "huggingface_sentence_transformers",
                "right_model_name": self.config.embedding_model_name,
                "right_mode": "embedding",
                "degraded_reason": None,
                "modalities_seen": modalities_seen,
                "vector_dimension": len(latent_vector),
                "emotion_keyword_hits": emotion_meta["keyword_hits"],
                "emotion_vector_hits": emotion_meta["vector_hits"],
                "emotion_peak_similarity": emotion_meta["peak_similarity"],
                "raw_salience": raw_salience,
                "runtime_feedback_bias": runtime_feedback_bias,
                "codec_used": self.codec is not None,
            },
        )
        
        if workspace is not None:
            workspace.right_notes.update({
                "backend": "huggingface",
                "surprise_score": surprise_score,
                "salience": salience,
                "raw_salience": raw_salience,
                "runtime_feedback_bias": runtime_feedback_bias,
                "emotional_labels": emotional_labels or ["neutral"],
                "confidence": confidence,
            })
            
        return state

    async def aperceive(self, user_turn: UserTurn, memory_context: Any | None = None, workspace: CognitiveWorkspace | None = None) -> RightHemisphereState:
        # A inferência real usando sentence-transformers bloqueia a CPU, 
        # em um cenário produtivo intenso isso deveria rodar em um ThreadPoolExecutor.
        # Por hora, mantemos simples para a Sprint 1.
        import asyncio
        return await asyncio.to_thread(self.perceive, user_turn, memory_context, workspace)

    def _calculate_surprise(self, latent_vector: list[float], memory_context: Any | None) -> float:
        if not memory_context or not memory_context.recent_episodes:
            return 0.5

        try:
            import numpy as np
            recent_vectors = [
                ep.right_state.latent_vector
                for ep in memory_context.recent_episodes
                if len(ep.right_state.latent_vector) == len(latent_vector)
            ]
            if not recent_vectors:
                return 0.5

            avg_vector = np.mean(recent_vectors, axis=0).tolist()

            if self.codec is not None:
                # Use approximate inner product for speed
                compressed_avg = self.codec.encode(avg_vector)
                ip = self.codec.inner_product_approx(latent_vector, compressed_avg)
                # ip ≈ cosine similarity when vectors are unit-normalized
                distance = 1.0 - max(-1.0, min(1.0, ip))
                return round(float(distance / 2.0), 3)

            vec = np.array(latent_vector)
            avg = np.array(avg_vector)
            norm_vec = np.linalg.norm(vec)
            norm_avg = np.linalg.norm(avg)
            if norm_vec == 0 or norm_avg == 0:
                return 0.5
            sim = np.dot(vec, avg) / (norm_vec * norm_avg)
            distance = 1.0 - sim
            return round(float(distance / 2.0), 3)
        except Exception:
            return 0.5

    def _extract_emotional_labels(self, text: str, latent_vector: list[float]) -> tuple[list[str], dict[str, Any]]:
        labels: set[str] = set()
        lowered_text = text.lower()
        keyword_hits = 0
        vector_hits = 0
        peak_similarity = 0.0
        
        # 1. Heurística rápida baseada em palavras chave para match exato
        for keyword in self.config.salience_keywords:
            if keyword in lowered_text:
                labels.add(keyword)
                keyword_hits += 1
                
        # 2. Similaridade de Cosseno com os embeddings dos labels emocionais
        import numpy as np
        vec = np.array(latent_vector)
        norm_vec = np.linalg.norm(vec)
        if norm_vec > 0:
            for idx, label_emb in enumerate(self._emotion_embeddings):
                sim = np.dot(vec, label_emb) / (norm_vec * np.linalg.norm(label_emb))
                peak_similarity = max(peak_similarity, float(sim))
                label = self._emotion_labels[idx]
                threshold = self.config.emotion_similarity_thresholds.get(label, 0.46)
                if sim >= threshold:
                    labels.add(label)
                    vector_hits += 1

        return sorted(labels), {
            "keyword_hits": keyword_hits,
            "vector_hits": vector_hits,
            "peak_similarity": round(float(peak_similarity), 4),
        }

    def _estimate_salience(self, text: str, emotional_labels: list[str]) -> float:
        lowered_text = text.lower()
        salience = 0.15 
        
        for label in emotional_labels:
            salience = max(salience, self.config.salience_keywords.get(label, 0.45))
            
        if "!" in lowered_text:
            salience = min(1.0, salience + 0.15)
        if lowered_text.isupper():
            salience = min(1.0, salience + 0.20)
            
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

    def _estimate_confidence(
        self,
        user_turn: UserTurn,
        emotional_labels: list[str],
        emotion_meta: dict[str, Any],
    ) -> float:
        # Confidence sobe quando ha evidencia convergente (keyword/similaridade/sinais)
        # e cai quando apenas sinais fracos sao encontrados.
        base = 0.55
        text_len = len(user_turn.user_text.strip())
        if text_len >= 20:
            base += 0.08
        elif text_len >= 8:
            base += 0.04

        signal_bonus = min(0.12, len(user_turn.signals) * 0.04)
        base += signal_bonus
        base += min(0.1, float(emotion_meta.get("keyword_hits", 0)) * 0.04)
        base += min(0.08, float(emotion_meta.get("vector_hits", 0)) * 0.03)

        peak_similarity = float(emotion_meta.get("peak_similarity", 0.0))
        if peak_similarity >= 0.6:
            base += 0.08
        elif peak_similarity >= 0.5:
            base += 0.04

        if not emotional_labels:
            base -= 0.06

        return round(max(0.35, min(0.95, base)), 3)
