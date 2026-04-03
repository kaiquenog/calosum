from __future__ import annotations

import json
from calosum.shared.models.types import utc_now


def record_cognitive_diary(
    path,
    *,
    turn_id: str,
    observation: str,
    action: str,
    confidence: float,
) -> None:
    payload = {
        "turn_id": turn_id,
        "observation": observation,
        "action": action,
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "recorded_at": utc_now().isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
