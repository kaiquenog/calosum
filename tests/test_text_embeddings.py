from __future__ import annotations

import math
import unittest

from calosum.adapters.text_embeddings import TextEmbeddingAdapter, TextEmbeddingAdapterConfig


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self) -> None:
        self.last_url = None
        self.last_payload = None

    async def post(self, url: str, json):
        self.last_url = url
        self.last_payload = json
        return _FakeResponse(
            {
                "data": [
                    {"index": 0, "embedding": [3.0, 4.0]},
                    {"index": 1, "embedding": [0.0, 5.0]},
                ]
            }
        )


class TextEmbeddingAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_embedding_adapter_normalizes_vectors_and_payload(self) -> None:
        client = _FakeAsyncClient()
        adapter = TextEmbeddingAdapter(
            TextEmbeddingAdapterConfig(
                provider="openai",
                api_url="https://api.openai.com/v1",
                api_key="sk-test",
                model_name="text-embedding-3-small",
                vector_size=4,
            ),
            client=client,
        )

        vectors = await adapter.aembed_texts(["olá", "mundo"])

        self.assertEqual(client.last_url, "https://api.openai.com/v1/embeddings")
        self.assertEqual(client.last_payload["dimensions"], 4)
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 4)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vectors[0])), 1.0, places=5)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vectors[1])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
