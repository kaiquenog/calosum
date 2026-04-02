from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PredictionMethod = Literal["jepa_trained", "jepa_heuristic", "mean_pooling"]


@dataclass(slots=True)
class ContextEmbedding:
    vector: list[float]
    turns_count: int
    turn_embeddings: list[list[float]] = field(default_factory=list)
    turn_ids: list[str] = field(default_factory=list)
    context_terms: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.vector) != 384:
            raise ValueError(f"context embedding must be 384-dim, got {len(self.vector)}")
        if self.turns_count < 0:
            raise ValueError(f"turns_count must be non-negative, got {self.turns_count}")


@dataclass(slots=True)
class ResponsePrediction:
    predicted_embedding: list[float]
    uncertainty: float
    prediction_method: PredictionMethod

    def __post_init__(self) -> None:
        if len(self.predicted_embedding) != 384:
            raise ValueError(
                f"predicted_embedding must be 384-dim, got {len(self.predicted_embedding)}"
            )
        if not (0.0 <= self.uncertainty <= 1.0):
            raise ValueError(f"uncertainty must be between 0.0 and 1.0, got {self.uncertainty}")


@dataclass(slots=True)
class SurpriseScore:
    score: float
    prediction_error: float
    uncertainty: float
    prediction_method: PredictionMethod
    source: str = "jepa_prediction_error"
    ignored_due_to_uncertainty: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")
        if not (0.0 <= self.prediction_error <= 1.0):
            raise ValueError(
                f"prediction_error must be between 0.0 and 1.0, got {self.prediction_error}"
            )
        if not (0.0 <= self.uncertainty <= 1.0):
            raise ValueError(f"uncertainty must be between 0.0 and 1.0, got {self.uncertainty}")
