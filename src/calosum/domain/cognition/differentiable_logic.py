from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from calosum.shared.models.types import MemoryContext, InputPerceptionState

class LatentPredicate(Protocol):
    """Protocol for a differentiable predicate grounded in latent space."""
    def compute_satisfaction(self, latent_vector: list[float]) -> float:
        ...

@dataclass(slots=True)
class LogicTensorNetwork:
    """
    Logic Tensor Network (LTN) with real fuzzy logic grounding.
    Implements Lukasiewicz and Product t-norms for differentiable
    symbolic reasoning over latent representations.

    Based on: Badreddine et al. (2022) — Logic Tensor Networks.
    """
    predicates: dict[str, LatentPredicate] = field(default_factory=dict)
    default_tnorm: str = "lukasiewicz"

    def ground_rule(self, rule_statement: str, latent_vector: list[float]) -> float:
        """
        Calculates the satisfaction degree [0, 1] of a symbolic rule
        given the current latent state using real fuzzy logic semantics.
        """
        terms = self._parse_rule_terms(rule_statement)
        if not terms:
            return 1.0

        satisfactions = []
        for term in terms:
            if term in self.predicates:
                sat = max(0.0, min(1.0, self.predicates[term].compute_satisfaction(latent_vector)))
                satisfactions.append(sat)
            else:
                sat = self._default_satisfaction(term, latent_vector)
                satisfactions.append(sat)

        if not satisfactions:
            return 1.0

        return self._apply_tnorm(satisfactions)

    def _apply_tnorm(self, values: list[float]) -> float:
        """Apply the configured t-norm to combine satisfaction degrees."""
        if self.default_tnorm == "lukasiewicz":
            return self._lukasiewicz_tnorm(values)
        elif self.default_tnorm == "product":
            return self._product_tnorm(values)
        elif self.default_tnorm == "godel":
            return self._godel_tnorm(values)
        return self._lukasiewicz_tnorm(values)

    @staticmethod
    def _lukasiewicz_tnorm(values: list[float]) -> float:
        """Lukasiewicz t-norm: max(0, sum(a_i) - (n-1))."""
        if not values:
            return 1.0
        return max(0.0, sum(values) - (len(values) - 1))

    @staticmethod
    def _product_tnorm(values: list[float]) -> float:
        """Product t-norm: product of all values."""
        result = 1.0
        for v in values:
            result *= v
        return result

    @staticmethod
    def _godel_tnorm(values: list[float]) -> float:
        """Gödel t-norm: min of all values."""
        return min(values) if values else 1.0

    def _parse_rule_terms(self, rule_statement: str) -> list[str]:
        """Extract predicate terms from a rule statement."""
        lower = rule_statement.lower()
        terms: list[str] = []
        for keyword in self.predicates:
            if keyword.lower() in lower:
                terms.append(keyword)
        if not terms:
            known_markers = {
                "urgente": "salience", "complexo": "complexity",
                "ambiguo": "ambiguity", "seguro": "safety",
                "empatico": "empathy", "criativo": "creativity",
            }
            for marker, predicate in known_markers.items():
                if marker in lower:
                    terms.append(predicate)
        return terms

    def _default_satisfaction(self, term: str, latent_vector: list[float]) -> float:
        """Default satisfaction based on latent vector energy in relevant dimensions."""
        if not latent_vector:
            return 0.5
        seed = sum(ord(c) for c in term)
        dim = len(latent_vector)
        indices = [(seed + i * 7) % dim for i in range(min(8, dim))]
        energy = sum(abs(latent_vector[i]) for i in indices) / len(indices)
        return max(0.0, min(1.0, energy))

    def lukasiewicz_implication(self, a: float, b: float) -> float:
        """Lukasiewicz implication: min(1, 1 - a + b)."""
        return min(1.0, 1.0 - a + b)

    def lukasiewicz_disjunction(self, values: list[float]) -> float:
        """Lukasiewicz s-norm (disjunction): min(1, sum(a_i))."""
        return min(1.0, sum(values))

    def negation(self, a: float) -> float:
        """Fuzzy negation: 1 - a."""
        return 1.0 - a


@dataclass(slots=True)
class CognitiveDissonanceMetric:
    """
    Measures the alignment between the Right Hemisphere's intuition
    and the Left Hemisphere's logical expectations.
    """
    def calculate(self, right_state: InputPerceptionState, logic_grounding: float) -> float:
        # Dissonance is high when the logic satisfaction degree
        # is low despite high salience, or vice versa.
        return abs(right_state.salience - logic_grounding)

from collections import deque
import numpy as np
from calosum.shared.models.types import MemoryContext
from calosum.shared.utils.free_energy import kl_divergence_gaussian, expected_free_energy_refined
from calosum.shared.utils.surprise_metrics import calibrated_surprise_score

_surprise_history = deque(maxlen=50)

def _get_prior(memory_context: MemoryContext | None, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    if memory_context and memory_context.recent_episodes:
        last_ep = memory_context.recent_episodes[-1]
        if last_ep.right_state.latent_mu:
            if last_ep.right_state.latent_logvar:
                return np.array(last_ep.right_state.latent_mu), np.array(last_ep.right_state.latent_logvar)
            return np.array(last_ep.right_state.latent_mu), np.ones(shape) * -2.0
        if last_ep.right_state.latent_vector:
            return np.array(last_ep.right_state.latent_vector), np.ones(shape) * -2.0
    return np.zeros(shape), np.zeros(shape)

def apply_active_inference(base_state: InputPerceptionState, memory_context: MemoryContext | None) -> InputPerceptionState:
    mu = np.array(base_state.latent_mu) if base_state.latent_mu else np.array(base_state.latent_vector)
    logvar = np.array(base_state.latent_logvar) if base_state.latent_logvar else np.ones_like(mu) * -2.0
    
    prior_mu, prior_logvar = _get_prior(memory_context, mu.shape)
    raw_surprise = kl_divergence_gaussian(mu, logvar, prior_mu, prior_logvar)
    _surprise_history.append(raw_surprise)
    
    calibrated_surprise = calibrated_surprise_score(raw_surprise, list(_surprise_history))
        
    base_confidence = max(0.0, min(1.0, float(base_state.confidence)))
    
    efe, _ = expected_free_energy_refined(mu, logvar, prior_mu, prior_logvar, epistemic_weight=1.5)
    
    merged_telemetry = dict(base_state.telemetry)
    merged_telemetry.update({
        "surprise_backend": "domain_differentiable_logic",
        "efe_score": round(efe, 4),
        "z_score_surprise": calibrated_surprise,
    })
    fused_salience = max(float(base_state.salience), calibrated_surprise)
    
    return InputPerceptionState(
        context_id=base_state.context_id,
        latent_vector=base_state.latent_vector,
        salience=round(max(0.0, min(1.0, fused_salience)), 4),
        emotional_labels=base_state.emotional_labels,
        world_hypotheses=base_state.world_hypotheses,
        confidence=base_confidence,
        surprise_score=calibrated_surprise,
        perception_status=base_state.perception_status,
        latent_mu=mu.tolist(),
        latent_logvar=logvar.tolist(),
        telemetry=merged_telemetry,
    )
