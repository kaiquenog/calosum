from __future__ import annotations

from typing import Any
import numpy as np
from calosum.shared.models.types import SoftPromptToken


class SoftPromptProjector:
    """Projector for aligning JEPA latent representations to LLM soft prompts."""

    def __init__(self, latent_dim: int = 768, prompt_dim: int = 1536) -> None:
        self.latent_dim = latent_dim
        self.prompt_dim = prompt_dim
        # Heuristic projection matrix if not trained
        self.projection_matrix = np.random.randn(latent_dim, prompt_dim).astype(np.float32) * 0.01

    def project(self, latent: list[float] | np.ndarray) -> list[SoftPromptToken]:
        """Project a latent vector into a sequence of soft prompt tokens."""
        latent_np = np.asarray(latent, dtype=np.float32)
        if latent_np.ndim == 1:
            latent_np = latent_np.reshape(1, -1)
            
        projected = latent_np @ self.projection_matrix
        
        # Convert projected values to a set of weighted tokens
        # For now, we use a simple heuristic to create tokens
        tokens = []
        for i in range(min(8, projected.shape[1] // 64)):
            chunk = projected[0, i*64:(i+1)*64]
            weight = float(np.tanh(np.mean(np.abs(chunk))))
            tokens.append(
                SoftPromptToken(
                    token=f"latent_feat_{i}",
                    weight=weight,
                    provenance="soft_prompt_projector_v1"
                )
            )
        return tokens
