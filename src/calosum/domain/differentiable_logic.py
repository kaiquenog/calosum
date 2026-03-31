from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from calosum.shared.types import MemoryContext, RightHemisphereState

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
    def calculate(self, right_state: RightHemisphereState, logic_grounding: float) -> float:
        # Dissonance is high when the logic satisfaction degree
        # is low despite high salience, or vice versa.
        return abs(right_state.salience - logic_grounding)
