from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass
from typing import Any

from calosum.shared.types import CognitiveWorkspace, MemoryContext, RightHemisphereState, UserTurn


@dataclass(slots=True)
class JepaRsConfig:
    binary_path: str = "jepa-rs"
    timeout_seconds: float = 20.0
    model_path: str | None = None
    latent_size: int = 384


class JepaRsRightHemisphereAdapter:
    """Rust backend adapter for local JEPA inference via JSON IPC."""

    def __init__(self, config: JepaRsConfig | None = None) -> None:
        self.config = config or JepaRsConfig()

    def perceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> RightHemisphereState:
        payload = {
            "text": user_turn.user_text,
            "signals_count": len(user_turn.signals),
            "model_path": self.config.model_path,
            "latent_size": self.config.latent_size,
        }
        result = self._invoke_backend(payload)

        latent = result.get("latent_vector")
        if not isinstance(latent, list) or not latent:
            raise RuntimeError("jepa-rs backend returned invalid latent_vector")

        surprise = float(result.get("surprise_score", 0.5))
        surprise = max(0.0, min(1.0, surprise))
        salience = max(0.0, min(1.0, float(result.get("salience", 0.45))))
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.7))))

        state = RightHemisphereState(
            context_id=user_turn.turn_id,
            latent_vector=[float(v) for v in latent],
            salience=salience,
            emotional_labels=[str(x) for x in result.get("emotional_labels", ["neutral"])][:6],
            world_hypotheses={
                "interaction_complexity": min(1.0, len(user_turn.user_text) / 240.0),
                "semantic_density": min(1.0, abs(sum(float(v) for v in latent[:32])) / 16.0),
                "predictive_alignment": max(0.0, 1.0 - surprise),
            },
            confidence=confidence,
            surprise_score=surprise,
            telemetry={
                "model_name": "jepa-rs",
                "right_backend": "jepa_rs",
                "right_model_name": "jepa-rs",
                "right_mode": "predictive",
                "degraded_reason": None,
                "binary_path": self.config.binary_path,
            },
        )

        if workspace is not None:
            workspace.right_notes.update(
                {
                    "backend": "jepa_rs",
                    "surprise_score": surprise,
                    "confidence": confidence,
                }
            )
        return state

    async def aperceive(
        self,
        user_turn: UserTurn,
        memory_context: MemoryContext | None = None,
        workspace: CognitiveWorkspace | None = None,
    ) -> RightHemisphereState:
        return await asyncio.to_thread(self.perceive, user_turn, memory_context, workspace)

    def _invoke_backend(self, payload: dict[str, Any]) -> dict[str, Any]:
        cmd = [self.config.binary_path, "infer", "--json"]
        completed = subprocess.run(
            cmd,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(f"jepa-rs failed: rc={completed.returncode} stderr={stderr}")

        out = completed.stdout.strip()
        if not out:
            raise RuntimeError("jepa-rs returned empty output")

        try:
            parsed = json.loads(out)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"jepa-rs returned non-json output: {out[:200]}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("jepa-rs payload must be an object")
        return parsed
