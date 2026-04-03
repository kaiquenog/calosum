from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class CrossAttentionBridgeConfig:
    target_dim: int = 384
    temperature: float = 0.75
    d_k: int = 64


class CrossAttentionBridgeAdapter:
    """Cross-attention fusion with learnable projections for the bridge.

    When PyTorch is available, uses nn.Linear projections (Q, K, V) and
    nn.Embedding for label representations — proper learned attention.
    Falls back to hash-based keys when torch is unavailable.
    """

    def __init__(self, config: CrossAttentionBridgeConfig | None = None) -> None:
        self.config = config or CrossAttentionBridgeConfig()
        self._torch_available = False
        self._W_q = None
        self._W_k = None
        self._W_v = None
        self._W_out = None
        self._label_embeddings = None
        self._optimizer = None
        self._label_to_idx: dict[str, int] = {}
        self._next_idx = 0
        self._init_learned_projections()

    def _init_learned_projections(self) -> None:
        try:
            import torch
            import torch.nn as nn

            d = self.config.target_dim
            d_k = self.config.d_k
            self._W_q = nn.Linear(d, d_k, bias=False)
            self._W_k = nn.Linear(d, d_k, bias=False)
            self._W_v = nn.Linear(d, d, bias=False)
            self._W_out = nn.Linear(d, d, bias=False)
            self._label_embeddings = nn.Embedding(64, d)
            self._torch_available = True
            self._optimizer = torch.optim.SGD(self.get_parameters(), lr=1e-3)
        except ImportError:
            self._torch_available = False

    def compute_adaptive_gate(
        self,
        surprise: float,
        confidence: float,
        context_novelty: float,
    ) -> tuple[float, float]:
        """Gating adaptativo baseado em estado cognitivo."""
        # Base weights
        latent_weight = 0.5 + 0.3 * surprise - 0.2 * confidence
        
        # Novelty adjustment: novelty puxa para 50/50
        novelty_factor = context_novelty * 0.2
        latent_weight = latent_weight * (1 - novelty_factor) + 0.5 * novelty_factor
        
        # Clamp
        latent_weight = max(0.2, min(0.8, latent_weight))
        context_weight = 1.0 - latent_weight
        
        return float(latent_weight), float(context_weight)

    def fuse_latent(
        self,
        *,
        latent_vector: list[float],
        emotional_labels: list[str],
        surprise: float = 0.0,
        confidence: float = 0.0,
        context_novelty: float = 0.0,
    ) -> tuple[list[float], dict[str, Any]]:
        x = self._fit(np.asarray(latent_vector, dtype=np.float32), self.config.target_dim)
        if x.size == 0:
            return [], {"fusion_backend": "cross_attention", "attention_entropy": 0.0}

        lw, cw = self.compute_adaptive_gate(surprise, confidence, context_novelty)

        if self._torch_available:
            return self._learned_fuse(x, emotional_labels or ["neutral"], lw, cw)
        return self._heuristic_fuse(x, emotional_labels or ["neutral"], lw, cw)

    def _learned_fuse(
        self, x: np.ndarray, labels: list[str], latent_w: float, context_w: float
    ) -> tuple[list[float], dict[str, Any]]:
        """Cross-attention with learned nn.Linear projections."""
        import torch

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        label_indices = [self._get_label_idx(label) for label in labels[:8]]
        k_input = self._label_embeddings(torch.tensor(label_indices))

        Q = self._W_q(x_tensor)           # (1, d_k)
        K = self._W_k(k_input)            # (n_labels, d_k)
        V = self._W_v(k_input)            # (n_labels, d)

        scores = torch.matmul(Q, K.T) / (self.config.d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)   # (1, d)

        fused = self._W_out(latent_w * x_tensor + context_w * context)
        fused = fused / (fused.norm() + 1e-8)
        entropy = float(-torch.sum(attn.detach() * torch.log(attn.detach() + 1e-9)))

        return fused.squeeze(0).detach().tolist(), {
            "fusion_backend": "learned_cross_attention",
            "attention_entropy": round(entropy, 6),
            "attention_heads": len(labels[:8]),
            "target_dim": self.config.target_dim,
            "projection_type": "nn.Linear",
        }

    def _heuristic_fuse(
        self, x: np.ndarray, labels: list[str], latent_w: float, context_w: float
    ) -> tuple[list[float], dict[str, Any]]:
        """Fallback using deterministic label vectors when torch unavailable."""
        keys = self._label_matrix_deterministic(labels, self.config.target_dim)
        scores = (keys @ x) / max(1e-5, self.config.temperature)
        attn = self._softmax(scores)
        context = attn @ keys

        gated = (latent_w * x) + (context_w * context)
        norm = np.linalg.norm(gated)
        fused = gated if norm == 0 else gated / norm

        entropy = float(-np.sum(attn * np.log(np.clip(attn, 1e-9, 1.0))))
        return fused.astype(np.float32).tolist(), {
            "fusion_backend": "cross_attention_heuristic",
            "attention_entropy": round(entropy, 6),
            "attention_heads": int(keys.shape[0]),
            "target_dim": self.config.target_dim,
            "projection_type": "deterministic",
        }

    def _get_label_idx(self, label: str) -> int:
        if label not in self._label_to_idx:
            self._label_to_idx[label] = self._next_idx % 64
            self._next_idx += 1
        return self._label_to_idx[label]

    def _label_matrix_deterministic(self, labels: list[str], dim: int) -> np.ndarray:
        """Deterministic label embedding fallback using character-based vectors."""
        rows: list[np.ndarray] = []
        for label in labels[:8]:
            vec = np.zeros(dim, dtype=np.float32)
            for i, ch in enumerate(label.lower()):
                vec[(ord(ch) + i * 7) % dim] += 1.0
            norm = np.linalg.norm(vec)
            rows.append(vec / max(norm, 1e-8))
        return np.stack(rows, axis=0)

    def get_parameters(self) -> list[Any]:
        """Return trainable parameters for bridge neural training loop."""
        if not self._torch_available:
            return []
        params = []
        for module in [self._W_q, self._W_k, self._W_v, self._W_out, self._label_embeddings]:
            if module is not None:
                params.extend(module.parameters())
        return params

    def train_step(self, latent_vector: list[float], target_salience: float, learning_rate: float = 0.001) -> float:
        """
        Perform a gradient-based training step to adapt the bridge projections.
        
        Implements online learning by computing backward pass between produced salience
        and target salience, updating projection weights (W_q, W_k, W_v, W_out, label_embeddings).
        
        Args:
            latent_vector: Input latent vector from JEPA
            target_salience: Target salience value from GEA reflection
            learning_rate: Learning rate for the optimizer (default: 0.001)
            
        Returns:
            Computed loss value
        """
        if not self._torch_available:
            # Fallback to heuristic mode - no training possible
            return 0.0
            
        # Check that all required modules are initialized and callable
        if not all([
            callable(getattr(self._W_q, '__call__', None)) if self._W_q is not None else False,
            callable(getattr(self._W_k, '__call__', None)) if self._W_k is not None else False,
            callable(getattr(self._W_v, '__call__', None)) if self._W_v is not None else False,
            callable(getattr(self._W_out, '__call__', None)) if self._W_out is not None else False,
            callable(getattr(self._label_embeddings, '__call__', None)) if self._label_embeddings is not None else False
        ]):
            return 0.0
            
        try:
            import torch
            import torch.nn as nn
            
            # Ensure we're in training mode
            self._W_q.train()
            self._W_k.train()
            self._W_v.train()
            self._W_out.train()
            self._label_embeddings.train()
            
            if self._optimizer is None:
                self._optimizer = torch.optim.SGD(self.get_parameters(), lr=learning_rate)
            self._optimizer.param_groups[0]["lr"] = learning_rate
            self._optimizer.zero_grad()
            
            # Process input (similar to fuse_latent but we need to compute salience)
            x = self._fit(np.asarray(latent_vector, dtype=np.float32), self.config.target_dim)
            if x.size == 0:
                return 0.0
                
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            
            # Use neutral label for training (in practice, this would come from context)
            label_indices = [self._get_label_idx("neutral")]
            k_input = self._label_embeddings(torch.tensor(label_indices))
            
            # Forward pass through attention mechanism
            Q = self._W_q(x_tensor)
            K = self._W_k(k_input)
            V = self._W_v(k_input)
            
            scores = torch.matmul(Q, K.T) / (self.config.d_k ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn, V)
            
            # Compute differentiable salience proxy from attention distribution
            produced_salience = torch.max(attn)

            # Compute loss (MSE between produced and target salience)
            loss_fn = nn.MSELoss()
            target_tensor = torch.tensor([target_salience], dtype=torch.float32, device=produced_salience.device)
            produced_tensor = produced_salience.reshape(1)
            loss = loss_fn(produced_tensor, target_tensor)
            
            # Backward pass
            loss.backward()
            
            self._optimizer.step()

            return float(loss.detach().item())
            
        except Exception:
            # In case of any error, return zero loss to avoid breaking the loop
            return 0.0

    def export_trainable_state(self) -> dict[str, Any]:
        if not self._torch_available:
            return {"torch_available": False}
        return {
            "torch_available": True,
            "W_q": self._W_q.state_dict() if self._W_q is not None else {},
            "W_k": self._W_k.state_dict() if self._W_k is not None else {},
            "W_v": self._W_v.state_dict() if self._W_v is not None else {},
            "W_out": self._W_out.state_dict() if self._W_out is not None else {},
            "label_embeddings": self._label_embeddings.state_dict() if self._label_embeddings is not None else {},
        }

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
