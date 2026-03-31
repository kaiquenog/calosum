from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class CrossAttentionBridgeConfig:
    target_dim: int = 384
    temperature: float = 0.75


class CrossAttentionBridgeAdapter:
    """Cross-attention fusion for right-latent compression in the bridge."""

    def __init__(self, config: CrossAttentionBridgeConfig | None = None) -> None:
        self.config = config or CrossAttentionBridgeConfig()

    def fuse_latent(
        self,
        latent_vector: list[float],
        emotional_labels: list[str],
    ) -> tuple[list[float], dict[str, Any]]:
        x = self._fit(np.asarray(latent_vector, dtype=np.float32), self.config.target_dim)
        if x.size == 0:
            return [], {"fusion_backend": "cross_attention", "attention_entropy": 0.0}

        keys = self._label_matrix(emotional_labels, self.config.target_dim)
        scores = (keys @ x) / max(1e-5, self.config.temperature)
        attn = self._softmax(scores)
        context = attn @ keys

        gated = (0.72 * x) + (0.28 * context)
        norm = np.linalg.norm(gated)
        fused = gated if norm == 0 else gated / norm

        entropy = float(-np.sum(attn * np.log(np.clip(attn, 1e-9, 1.0))))
        return fused.astype(np.float32).tolist(), {
            "fusion_backend": "cross_attention",
            "attention_entropy": round(entropy, 6),
            "attention_heads": int(keys.shape[0]),
            "target_dim": self.config.target_dim,
        }

    def _label_matrix(self, labels: list[str], dim: int) -> np.ndarray:
        labels = labels or ["neutral"]
        rows: list[np.ndarray] = []
        for label in labels[:8]:
            h = hashlib.sha256(label.encode("utf-8")).digest()
            row = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            row = self._fit(row, dim)
            row = (row / 255.0) * 2.0 - 1.0
            rows.append(row)
        return np.stack(rows, axis=0)

    def _fit(self, vec: np.ndarray, size: int) -> np.ndarray:
        if vec.size == size:
            return vec
        if vec.size > size:
            return vec[:size]
        out = np.zeros(size, dtype=np.float32)
        out[: vec.size] = vec
        return out

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        total = float(np.sum(exp))
        if total <= 0:
            return np.full_like(values, 1.0 / len(values))
        return exp / total
