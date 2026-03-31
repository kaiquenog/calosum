"""Codec de compressão vetorial baseado em PolarQuant + QJL (TurboQuant, ICLR 2026)."""
from __future__ import annotations

import math
import struct

from calosum.shared.ports import VectorCodecPort


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(v: list[float]) -> tuple[float, list[float]]:
    """Return (norm, unit_vector). Returns (0.0, zeros) for zero vector."""
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return 0.0, [0.0] * len(v)
    return norm, [x / norm for x in v]


def _pad_to_power_of_two(v: list[float]) -> list[float]:
    """Pad vector with zeros to the next power of two (needed for WHT)."""
    n = len(v)
    if n == 0:
        return v
    p = 1
    while p < n:
        p <<= 1
    return v + [0.0] * (p - n)


def _hadamard_rademacher(v: list[float], seed: int) -> list[float]:
    """Apply a deterministic Walsh-Hadamard + Rademacher random rotation.

    Uses a seeded xorshift32 to generate sign-flip pattern (Rademacher), then
    applies an iterative Walsh-Hadamard transform.  This is data-oblivious:
    no matrix is stored; the seed fully determines the map.

    The input is padded to the next power of two for WHT correctness.
    Only the first `len(v)` output values are returned.
    """
    n_orig = len(v)
    padded = _pad_to_power_of_two(v)
    n = len(padded)

    # --- Rademacher sign flip (seeded xorshift32) ---
    result = list(padded)
    state = (seed & 0xFFFFFFFF) or 1  # ensure non-zero
    for i in range(n):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= (state >> 17) & 0xFFFFFFFF
        state ^= (state << 5) & 0xFFFFFFFF
        sign = 1.0 if (state & 1) == 0 else -1.0
        result[i] *= sign

    # --- Walsh-Hadamard transform (iterative, normalized) ---
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a, b = result[j], result[j + h]
                result[j] = a + b
                result[j + h] = a - b
        h <<= 1

    scale = 1.0 / math.sqrt(n) if n > 0 else 1.0
    return [x * scale for x in result[:n_orig]]


# ---------------------------------------------------------------------------
# PolarQuantEncoder
# ---------------------------------------------------------------------------

class PolarQuantEncoder:
    """Encodes a unit vector as quantized generalized spherical angles.

    For an n-dimensional unit vector, produces n-1 angles using the standard
    generalized spherical coordinate mapping:

        x₁ = cos(θ₁)
        xᵢ = sin(θ₁)·…·sin(θᵢ₋₁)·cos(θᵢ)  for 1 < i < n
        xₙ = sin(θ₁)·…·sin(θₙ₋₁)

    The first n-2 angles θ₁,…,θₙ₋₂ are in [0, π].
    The last angle θₙ₋₁ is in [0, 2π].

    Norm is stored as a float32 header so decode can restore scale.
    """

    def __init__(self, bits: int = 3) -> None:
        if bits < 1 or bits > 8:
            raise ValueError("PolarQuantEncoder: bits must be in [1, 8]")
        self.bits = bits
        self._levels = 1 << bits  # 2^bits quantization levels

    def encode(self, vector: list[float]) -> bytes:
        norm, unit = _normalize(vector)
        angles = self._to_polar_angles(unit)
        quantized = self._quantize_angles(angles)
        packed = self._pack_angles(quantized)
        # Header: norm (float32, 4 bytes) + dimension (uint16, 2 bytes)
        header = struct.pack(">fH", norm, len(vector))
        return header + packed

    def decode(self, data: bytes) -> list[float]:
        norm, dim = struct.unpack_from(">fH", data, 0)
        packed = data[6:]
        n_angles = max(0, dim - 1)
        quantized = self._unpack_angles(packed, n_angles)
        angles = self._dequantize_angles(quantized, dim)
        unit = self._from_polar_angles(angles, dim)
        return [x * norm for x in unit]

    def inner_product_approx(self, query: list[float], data: bytes) -> float:
        """Approximate inner product using the decoded vector."""
        decoded = self.decode(data)
        n = min(len(query), len(decoded))
        return sum(query[i] * decoded[i] for i in range(n))

    # --- private ---

    def _to_polar_angles(self, unit: list[float]) -> list[float]:
        """Convert n-dimensional unit vector to n-1 generalized spherical angles.

        θ₁,…,θₙ₋₂ ∈ [0, π]   (determined by acos of x_i / r_i)
        θₙ₋₁      ∈ [0, 2π]  (determined by atan2 of last two components)
        """
        n = len(unit)
        if n <= 1:
            return []
        angles: list[float] = []
        # For all but the last angle, use acos with the accumulated radius
        for i in range(n - 2):
            r = math.sqrt(sum(unit[j] * unit[j] for j in range(i, n)))
            if r < 1e-12:
                angles.append(0.0)
            else:
                cos_a = max(-1.0, min(1.0, unit[i] / r))
                angles.append(math.acos(cos_a))  # in [0, π]
        # Last angle uses atan2 for full [0, 2π] range
        last = math.atan2(unit[n - 1], unit[n - 2])
        if last < 0:
            last += 2 * math.pi
        angles.append(last)
        return angles

    def _from_polar_angles(self, angles: list[float], dim: int) -> list[float]:
        """Reconstruct unit vector from generalized spherical angles."""
        if dim == 1:
            return [1.0]
        result: list[float] = []
        sin_acc = 1.0
        for i, angle in enumerate(angles[:-1]):
            result.append(sin_acc * math.cos(angle))
            sin_acc = sin_acc * math.sin(angle)
        # Last two components from atan2 angle
        last_angle = angles[-1] if angles else 0.0
        result.append(sin_acc * math.cos(last_angle))
        result.append(sin_acc * math.sin(last_angle))
        # Pad or truncate to dim
        while len(result) < dim:
            result.append(0.0)
        return result[:dim]

    def _quantize_angles(self, angles: list[float]) -> list[int]:
        """Quantize angles to integer levels."""
        levels = self._levels
        return [int(a / (2 * math.pi) * levels) % levels for a in angles]

    def _dequantize_angles(self, quantized: list[int], dim: int) -> list[float]:
        """Dequantize levels back to angles in [0, 2π]."""
        levels = self._levels
        step = 2 * math.pi / levels
        return [q * step + step / 2 for q in quantized]

    def _pack_angles(self, quantized: list[int]) -> bytes:
        """Pack quantized angles (each < 2^bits) into a compact bytestring."""
        bits = self.bits
        buf = bytearray()
        acc = 0
        n_bits = 0
        mask = self._levels - 1
        for val in quantized:
            acc = (acc << bits) | (val & mask)
            n_bits += bits
            while n_bits >= 8:
                n_bits -= 8
                buf.append((acc >> n_bits) & 0xFF)
        if n_bits > 0:
            buf.append((acc << (8 - n_bits)) & 0xFF)
        return bytes(buf)

    def _unpack_angles(self, data: bytes, n_angles: int) -> list[int]:
        """Unpack n_angles quantized values from a packed bytestring."""
        bits = self.bits
        mask = self._levels - 1
        result: list[int] = []
        acc = 0
        n_bits = 0
        byte_iter = iter(data)
        for _ in range(n_angles):
            while n_bits < bits:
                try:
                    acc = (acc << 8) | next(byte_iter)
                    n_bits += 8
                except StopIteration:
                    acc <<= 8
                    n_bits += 8
            n_bits -= bits
            result.append((acc >> n_bits) & mask)
        return result


# ---------------------------------------------------------------------------
# QJLResidualEncoder
# ---------------------------------------------------------------------------

class QJLResidualEncoder:
    """1-bit-per-dimension sign quantizer after random rotation.

    Provides an unbiased estimator of the inner product between a query and
    the sign-compressed residual, as described in QJL (arXiv 2406.03482).
    """

    def __init__(self, seed: int = 0xCAFEBABE) -> None:
        self.seed = seed

    def encode(self, residual: list[float]) -> bytes:
        rotated = _hadamard_rademacher(residual, self.seed)
        n = len(rotated)
        n_bytes = (n + 7) // 8
        buf = bytearray(n_bytes)
        for i, val in enumerate(rotated):
            if val >= 0:
                buf[i // 8] |= (1 << (7 - (i % 8)))
        header = struct.pack(">H", n)
        return header + bytes(buf)

    def inner_product_approx(self, query: list[float], data: bytes) -> float:
        """Estimate inner product between query and compressed residual."""
        (n,) = struct.unpack_from(">H", data, 0)
        sign_bytes = data[2:]
        rotated_query = _hadamard_rademacher(query, self.seed)
        n_used = min(n, len(rotated_query))
        scale = math.sqrt(math.pi / (2 * n_used)) if n_used > 0 else 1.0
        acc = 0.0
        for i in range(n_used):
            bit = (sign_bytes[i // 8] >> (7 - (i % 8))) & 1
            sign = 1.0 if bit else -1.0
            acc += sign * rotated_query[i]
        return acc * scale


# ---------------------------------------------------------------------------
# TurboQuantVectorCodec — composed codec
# ---------------------------------------------------------------------------

class TurboQuantVectorCodec:
    """PolarQuant (bits/dim) + QJL residual (1 bit/dim).

    Default: 3 + 1 = 4 bits/dim total.
    Satisfies VectorCodecPort.
    """

    def __init__(self, bits: int = 3) -> None:
        self._pq = PolarQuantEncoder(bits=bits)
        self._qjl = QJLResidualEncoder()
        self._bits = bits

    @property
    def bits_per_dim(self) -> int:
        return self._bits + 1  # PolarQuant + QJL

    def encode(self, vector: list[float]) -> bytes:
        # Stage 1: PolarQuant
        pq_bytes = self._pq.encode(vector)
        pq_decoded = self._pq.decode(pq_bytes)

        # Stage 2: QJL on residual
        dim = len(vector)
        residual = [vector[i] - pq_decoded[i] for i in range(dim)]
        qjl_bytes = self._qjl.encode(residual)

        header = struct.pack(">H", len(pq_bytes))
        return header + pq_bytes + qjl_bytes

    def decode(self, compressed: bytes) -> list[float]:
        (pq_len,) = struct.unpack_from(">H", compressed, 0)
        pq_bytes = compressed[2: 2 + pq_len]
        return self._pq.decode(pq_bytes)

    def inner_product_approx(self, query: list[float], compressed: bytes) -> float:
        """Weighted combination of PolarQuant and QJL inner product estimates."""
        (pq_len,) = struct.unpack_from(">H", compressed, 0)
        pq_bytes = compressed[2: 2 + pq_len]
        qjl_bytes = compressed[2 + pq_len:]

        pq_ip = self._pq.inner_product_approx(query, pq_bytes)
        qjl_ip = self._qjl.inner_product_approx(query, qjl_bytes)

        w_pq = self._bits / (self._bits + 1)
        w_qjl = 1.0 / (self._bits + 1)
        return w_pq * pq_ip + w_qjl * qjl_ip


__all__ = ["TurboQuantVectorCodec", "PolarQuantEncoder", "QJLResidualEncoder"]
