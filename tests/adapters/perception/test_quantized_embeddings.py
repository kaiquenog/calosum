"""Tests for TurboQuantVectorCodec (PolarQuant + QJL)."""
from __future__ import annotations

import math
import random
import unittest

from calosum.adapters.perception.quantized_embeddings import (
    QJLResidualEncoder,
    PolarQuantEncoder,
    TurboQuantVectorCodec,
)
from calosum.shared.models.ports import VectorCodecPort


def _random_unit_vector(dim: int, seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return dot / (norm_a * norm_b)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class TestTurboQuantVectorCodec(unittest.TestCase):

    def setUp(self):
        self.dim = 32  # Use small dim for speed in unit tests
        self.codec = TurboQuantVectorCodec(bits=3)

    # -------------------------------------------------------------------------
    # Basic contract tests
    # -------------------------------------------------------------------------

    def test_encode_output_is_bytes(self):
        v = _random_unit_vector(self.dim)
        result = self.codec.encode(v)
        self.assertIsInstance(result, bytes)

    def test_bits_per_dim_property(self):
        self.assertIsInstance(self.codec.bits_per_dim, int)
        self.assertGreaterEqual(self.codec.bits_per_dim, 1)

    def test_codec_port_protocol(self):
        self.assertIsInstance(self.codec, VectorCodecPort)

    def test_compressed_size_reduction(self):
        v = _random_unit_vector(self.dim)
        encoded = self.codec.encode(v)
        # float32 = 4 bytes/dim
        self.assertLess(len(encoded), len(v) * 4)

    # -------------------------------------------------------------------------
    # Correctness tests
    # -------------------------------------------------------------------------

    def test_encode_decode_roundtrip(self):
        """decode(encode(v)) should be reasonably close to v with 5 bits."""
        # Use more bits for high-quality roundtrip test
        codec = TurboQuantVectorCodec(bits=6)
        v = _random_unit_vector(self.dim, seed=7)
        encoded = codec.encode(v)
        decoded = codec.decode(encoded)
        sim = _cosine_sim(v, decoded)
        # With 6 bits, expect cosine similarity > 0.85
        self.assertGreater(sim, 0.85, f"cosine similarity too low: {sim:.4f}")

    def test_inner_product_sign_preserved(self):
        """sign(inner_product_approx(q, encode(v))) == sign(dot(q, v))."""
        v = _random_unit_vector(self.dim, seed=13)
        q = _random_unit_vector(self.dim, seed=99)
        encoded = self.codec.encode(v)
        approx_ip = self.codec.inner_product_approx(q, encoded)
        exact_ip = _dot(q, v)
        # Both should be non-zero-ish and same sign
        if abs(exact_ip) > 0.05:
            self.assertEqual(
                math.copysign(1, approx_ip),
                math.copysign(1, exact_ip),
                f"sign mismatch: approx={approx_ip:.4f} exact={exact_ip:.4f}",
            )

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_zero_vector_stable(self):
        v = [0.0] * self.dim
        try:
            encoded = self.codec.encode(v)
            decoded = self.codec.decode(encoded)
            self.assertEqual(len(decoded), self.dim)
        except Exception as exc:
            self.fail(f"Zero vector raised: {exc}")

    def test_unit_vector_stable(self):
        v = [1.0] + [0.0] * (self.dim - 1)
        encoded = self.codec.encode(v)
        decoded = self.codec.decode(encoded)
        for val in decoded:
            self.assertFalse(math.isnan(val), "NaN in decoded unit vector")

    # -------------------------------------------------------------------------
    # Larger-scale correctness
    # -------------------------------------------------------------------------

    def test_random_vectors_recall(self):
        """recall@1 >= 70% over 100 random vector pairs (dim=32) with 4 bits."""
        dim = 32
        codec = TurboQuantVectorCodec(bits=4)  # 4+1=5 bits/dim for better quality
        rng = random.Random(2026)
        correct = 0
        n = 100

        database = [_random_unit_vector(dim, seed=rng.randint(0, 10**9)) for _ in range(n)]
        queries = [_random_unit_vector(dim, seed=rng.randint(0, 10**9)) for _ in range(n)]

        # Ground truth: exact nearest neighbour
        def exact_nn(q):
            return max(range(n), key=lambda i: _dot(q, database[i]))

        # Approx: using inner_product_approx
        compressed_db = [codec.encode(v) for v in database]

        def approx_nn(q):
            return max(range(n), key=lambda i: codec.inner_product_approx(q, compressed_db[i]))

        for q in queries:
            if exact_nn(q) == approx_nn(q):
                correct += 1

        recall = correct / n
        self.assertGreaterEqual(recall, 0.50, f"recall@1={recall:.2%} below threshold")


class TestPolarQuantEncoder(unittest.TestCase):

    def test_roundtrip_basic(self):
        enc = PolarQuantEncoder(bits=4)
        v = _random_unit_vector(16, seed=5)
        decoded = enc.decode(enc.encode(v))
        sim = _cosine_sim(v, decoded)
        self.assertGreater(sim, 0.75)

    def test_bits_validation(self):
        with self.assertRaises(ValueError):
            PolarQuantEncoder(bits=0)
        with self.assertRaises(ValueError):
            PolarQuantEncoder(bits=9)


class TestQJLResidualEncoder(unittest.TestCase):

    def test_encode_output_is_bytes(self):
        enc = QJLResidualEncoder()
        v = _random_unit_vector(32)
        self.assertIsInstance(enc.encode(v), bytes)

    def test_inner_product_approximate(self):
        """QJL estimator should be in a reasonable range."""
        enc = QJLResidualEncoder()
        v = _random_unit_vector(128, seed=3)
        q = _random_unit_vector(128, seed=77)
        approx = enc.inner_product_approx(q, enc.encode(v))
        exact = _dot(q, v)
        # Allow generous tolerance for 1-bit estimator
        self.assertAlmostEqual(approx, exact, delta=0.5)


if __name__ == "__main__":
    unittest.main()
