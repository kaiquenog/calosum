from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from calosum.shared.types import MemoryContext, Modality, RightHemisphereState, UserTurn

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

    def __init__(self, config: HuggingFaceRightHemisphereConfig | None = None) -> None:
        self.config = config or HuggingFaceRightHemisphereConfig()
        
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

    def perceive(self, user_turn: UserTurn, memory_context: Any | None = None) -> RightHemisphereState:
        text = user_turn.user_text
        if not text.strip():
            text = "silence"

        # 1. Gera o vetor latente real
        embeddings = self.embedder.encode([text])
        latent_vector = embeddings[0].tolist()

        # 2. Extrai rótulos emocionais reais (via similaridade vetorial leve)
        emotional_labels = self._extract_emotional_labels(text, latent_vector)
        
        # 3. Estima a saliência (intensidade emocional)
        salience = self._estimate_salience(text, emotional_labels)
        
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
            "semantic_density": round(semantic_density, 3)
        }

        surprise_score = self._calculate_surprise(latent_vector, memory_context)

        return RightHemisphereState(
            context_id=user_turn.turn_id,
            latent_vector=latent_vector,
            salience=salience,
            emotional_labels=emotional_labels or ["neutral"],
            world_hypotheses=world_hypotheses,
            confidence=0.85,
            surprise_score=surprise_score,
            telemetry={
                "model_name": self.config.embedding_model_name,
                "modalities_seen": [signal.modality.value for signal in user_turn.signals] if user_turn.signals else ["text"],
                "vector_dimension": len(latent_vector)
            },
        )

    async def aperceive(self, user_turn: UserTurn, memory_context: Any | None = None) -> RightHemisphereState:
        # A inferência real usando sentence-transformers bloqueia a CPU, 
        # em um cenário produtivo intenso isso deveria rodar em um ThreadPoolExecutor.
        # Por hora, mantemos simples para a Sprint 1.
        import asyncio
        return await asyncio.to_thread(self.perceive, user_turn, memory_context)

    def _calculate_surprise(self, latent_vector: list[float], memory_context: Any | None) -> float:
        if not memory_context or not memory_context.recent_episodes:
            return 0.5
            
        try:
            import numpy as np
            recent_vectors = [ep.right_state.latent_vector for ep in memory_context.recent_episodes if len(ep.right_state.latent_vector) == len(latent_vector)]
            if not recent_vectors:
                return 0.5
                
            avg_vector = np.mean(recent_vectors, axis=0)
            vec = np.array(latent_vector)
            
            norm_vec = np.linalg.norm(vec)
            norm_avg = np.linalg.norm(avg_vector)
            
            if norm_vec == 0 or norm_avg == 0:
                return 0.5
                
            sim = np.dot(vec, avg_vector) / (norm_vec * norm_avg)
            distance = 1.0 - sim
            return round(float(distance / 2.0), 3)
        except Exception:
            return 0.5

    def _extract_emotional_labels(self, text: str, latent_vector: list[float]) -> list[str]:
        labels: list[str] = []
        lowered_text = text.lower()
        
        # 1. Heurística rápida baseada em palavras chave para match exato
        for keyword in self.config.salience_keywords:
            if keyword in lowered_text:
                labels.append(keyword)
                
        # 2. Similaridade de Cosseno com os embeddings dos labels emocionais
        import numpy as np
        vec = np.array(latent_vector)
        norm_vec = np.linalg.norm(vec)
        if norm_vec > 0:
            for idx, label_emb in enumerate(self._emotion_embeddings):
                sim = np.dot(vec, label_emb) / (norm_vec * np.linalg.norm(label_emb))
                # Threshold calibrado para o modelo multilingual
                if sim > 0.35:
                    labels.append(self._emotion_labels[idx])
        
        return sorted(set(labels))

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
