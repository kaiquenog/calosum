from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    from calosum.shared.models.ports import VectorCodecPort

from calosum.shared.utils.async_utils import run_sync

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TextEmbeddingAdapterConfig:
    provider: str = "auto"
    api_url: str | None = None
    api_key: str | None = None
    model_name: str = "text-embedding-3-small"
    vector_size: int = 384
    timeout_s: float = 60.0


class TextEmbeddingAdapter:
    """
    Backend de embeddings de texto com fallback explicito.

    Ordem de preferencia em `auto`:
    - OpenAI oficial / endpoint OpenAI-compatible configurado
    - Sentence Transformers local, se a stack opcional estiver disponivel
    - embedding lexical deterministico, como fallback estavel
    """

    def __init__(
        self,
        config: TextEmbeddingAdapterConfig | None = None,
        client: httpx.AsyncClient | None = None,
        codec: VectorCodecPort | None = None,
    ) -> None:
        self.config = config or TextEmbeddingAdapterConfig()
        headers: dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        if self.config.provider.strip().lower() == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/kaiquenog/calosum"
            headers["X-Title"] = "Calosum Agent Framework"

        self.client = client or httpx.AsyncClient(headers=headers, timeout=self.config.timeout_s)
        self._sentence_transformer: Any | None = None
        self.codec: VectorCodecPort | None = codec

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return run_sync(self.aembed_texts(texts))

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        normalized = [text.strip() or "silence" for text in texts]
        provider = self._resolve_provider()

        try:
            if provider in {"openai", "openai_compatible"}:
                return await self._embed_with_http(normalized)
            if provider == "huggingface":
                return self._embed_with_sentence_transformers(normalized)
        except Exception as exc:
            logger.warning(
                "Embedding backend %s failed, falling back to lexical embeddings: %s",
                provider,
                exc,
            )

        return [self._lexical_vector_for_text(text) for text in normalized]

    async def embed_texts_compressed(self, texts: list[str]) -> list[bytes]:
        """Embed texts and compress each vector with the injected codec.

        Falls back to raw float32 serialization when no codec is configured.
        """
        vectors = await self.aembed_texts(texts)
        if self.codec is None:
            import struct
            return [
                b"".join(struct.pack(">f", v) for v in vec) for vec in vectors
            ]
        return [self.codec.encode(vec) for vec in vectors]

    def backend_name(self) -> str:
        try:
            return self._resolve_provider()
        except Exception:
            return "lexical"

    def _resolve_provider(self) -> str:
        provider = self.config.provider.strip().lower()
        if provider in {"openai", "openai_compatible", "huggingface", "lexical", "openrouter"}:
            return provider

        if self.config.api_url:
            hostname = (urlparse(self.config.api_url).hostname or "").lower()
            if hostname == "api.openai.com":
                return "openai"
            return "openai_compatible"

        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return "lexical"
        return "huggingface"

    async def _embed_with_http(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.config.model_name,
            "input": texts,
            "dimensions": self.config.vector_size,
        }
        response = await self.client.post(self._embeddings_url(), json=payload)
        response.raise_for_status()
        data = response.json()
        rows = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        vectors = [self._normalize_vector(item.get("embedding", [])) for item in rows]
        if len(vectors) != len(texts):
            raise ValueError("embedding endpoint returned an unexpected number of vectors")
        return vectors

    def _embeddings_url(self) -> str:
        if self.config.provider.strip().lower() == "openrouter" and self.config.api_url is None:
            return "https://openrouter.ai/api/v1/embeddings"
            
        base = (self.config.api_url or "https://api.openai.com/v1").rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        if base.endswith("/embeddings"):
            return base
        if base.endswith("/v1"):
            return f"{base}/embeddings"
        return f"{base}/v1/embeddings"

    def _embed_with_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        if self._sentence_transformer is None:
            import transformers

            transformers.logging.set_verbosity_error()
            transformers.utils.logging.disable_progress_bar()

            from sentence_transformers import SentenceTransformer

            self._sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        matrix = self._sentence_transformer.encode(texts)
        return [self._normalize_vector(row.tolist()) for row in matrix]

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        if not vector:
            return [0.0] * self.config.vector_size

        if len(vector) > self.config.vector_size:
            vector = vector[: self.config.vector_size]
        elif len(vector) < self.config.vector_size:
            vector = vector + ([0.0] * (self.config.vector_size - len(vector)))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return [0.0] * self.config.vector_size
        return [round(value / norm, 6) for value in vector]

    def _lexical_vector_for_text(self, text: str) -> list[float]:
        tokens = re.findall(r"\w+", text.lower()) or ["silence"]
        vector = [0.0] * self.config.vector_size

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(0, len(digest), 4):
                chunk = digest[index:index + 4]
                position = int.from_bytes(chunk[:2], "big") % self.config.vector_size
                sign = 1.0 if chunk[2] % 2 == 0 else -1.0
                magnitude = 1.0 + (chunk[3] / 255.0)
                vector[position] += sign * magnitude

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [round(value / norm, 6) for value in vector]
