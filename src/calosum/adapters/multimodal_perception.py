from __future__ import annotations

import asyncio
import random
from typing import Any

from calosum.shared.ports import VisionEmbeddingPort

class MockVisionAdapter(VisionEmbeddingPort):
    """
    Simula um extrator de features visuais (tipo CLIP) 
    sem carregar pesos de 400MB.
    """
    
    def embed_image(self, image_data: bytes) -> list[float]:
        # Simula um vetor de 512 dimensões com base no hash dos dados
        random.seed(sum(image_data) % 10000)
        return [random.uniform(-1, 1) for _ in range(512)]

    async def aembed_image(self, image_data: bytes) -> list[float]:
        await asyncio.sleep(0.05) # Simula latência de inferência
        return self.embed_image(image_data)
