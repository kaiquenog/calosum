from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StructuredMismatchSignal:
    """Sinal estruturado emitido quando runtime/verifier detectam divergencia operacional."""

    source: str
    severity: float
    reasons: list[str] = field(default_factory=list)
    rejected_action_types: list[str] = field(default_factory=list)
    failure_types: list[str] = field(default_factory=list)
    recommended_bridge_directives: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "severity": round(max(0.0, min(1.0, float(self.severity))), 4),
            "reasons": list(self.reasons),
            "rejected_action_types": list(self.rejected_action_types),
            "failure_types": list(self.failure_types),
            "recommended_bridge_directives": list(self.recommended_bridge_directives),
        }
