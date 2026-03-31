from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Any

import numpy as np

from calosum.shared.ports import VisionEmbeddingPort


@dataclass(slots=True)
class LocalClipVisionConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512


class LocalClipVisionAdapter(VisionEmbeddingPort):
    """Local CLIP-based vision embedding adapter with deterministic fallback."""

    def __init__(self, config: LocalClipVisionConfig | None = None) -> None:
        self.config = config or LocalClipVisionConfig()
        self._processor: Any | None = None
        self._model: Any | None = None

    def embed_image(self, image_data: bytes) -> list[float]:
        image = self._load_image(image_data)
        if image is None:
            return self._byte_feature_fallback(image_data)

        if self._ensure_clip_loaded():
            try:
                import torch

                inputs = self._processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    features = self._model.get_image_features(**inputs)
                vec = features[0].detach().cpu().numpy().astype(np.float32)
                vec = self._fit_size(vec, self.config.embedding_dim)
                norm = np.linalg.norm(vec)
                if norm != 0:
                    vec = vec / norm
                return vec.tolist()
            except Exception:
                return self._byte_feature_fallback(image_data)

        return self._byte_feature_fallback(image_data)

    async def aembed_image(self, image_data: bytes) -> list[float]:
        return await asyncio.to_thread(self.embed_image, image_data)

    def _ensure_clip_loaded(self) -> bool:
        if self._processor is not None and self._model is not None:
            return True
        try:
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.config.model_name)
            self._model = CLIPModel.from_pretrained(self.config.model_name)
            self._model.eval()
            return True
        except Exception:
            self._processor = None
            self._model = None
            return False

    def _load_image(self, image_data: bytes):
        try:
            from PIL import Image

            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception:
            return None

    def _byte_feature_fallback(self, image_data: bytes) -> list[float]:
        if not image_data:
            return [0.0 for _ in range(self.config.embedding_dim)]
        arr = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)
        if arr.size == 0:
            return [0.0 for _ in range(self.config.embedding_dim)]

        # Real deterministic signal extraction from bytes
        chunks = np.array_split(arr, self.config.embedding_dim)
        vec = np.asarray([float(np.mean(chunk)) if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)
        vec = (vec / 127.5) - 1.0
        norm = np.linalg.norm(vec)
        if norm != 0:
            vec = vec / norm
        return vec.tolist()

    def _fit_size(self, vec: np.ndarray, size: int) -> np.ndarray:
        if vec.size == size:
            return vec
        if vec.size > size:
            return vec[:size]
        out = np.zeros(size, dtype=np.float32)
        out[: vec.size] = vec
        return out
