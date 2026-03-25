"""
utils.py - Bit packing utilities and helpers for TurboQuant

Supports 2-bit, 3-bit, and 4-bit packing/unpacking via numpy.
"""

import numpy as np


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------

def pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    """
    Pack an array of unsigned integer indices into a compact uint8 array.

    Parameters
    ----------
    indices : np.ndarray, shape (n,), dtype uint8/uint16
        Values in [0, 2**bits - 1].
    bits : int
        Bit-width per index (2, 3, 4, 8 …).

    Returns
    -------
    packed : np.ndarray, dtype uint8
        Packed bytes. Length = ceil(n * bits / 8).
    """
    if bits == 8:
        return indices.astype(np.uint8)

    n = len(indices)
    total_bits = n * bits
    n_bytes = (total_bits + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)

    for i, val in enumerate(indices):
        bit_pos = i * bits
        byte_pos = bit_pos >> 3
        bit_off = bit_pos & 7
        # Write up to 2 bytes (handles up to 8-bit values split across bytes)
        packed[byte_pos] |= (int(val) & 0xFF) << bit_off & 0xFF
        if bit_off + bits > 8 and byte_pos + 1 < n_bytes:
            packed[byte_pos + 1] |= (int(val) >> (8 - bit_off)) & 0xFF

    return packed


def unpack_indices(packed: np.ndarray, n: int, bits: int) -> np.ndarray:
    """
    Unpack a compact uint8 array into integer indices.

    Parameters
    ----------
    packed : np.ndarray, dtype uint8
    n      : int  — number of indices to unpack
    bits   : int  — bit-width per index

    Returns
    -------
    indices : np.ndarray, shape (n,), dtype uint16
    """
    if bits == 8:
        return packed[:n].astype(np.uint16)

    mask = (1 << bits) - 1
    indices = np.empty(n, dtype=np.uint16)

    for i in range(n):
        bit_pos = i * bits
        byte_pos = bit_pos >> 3
        bit_off = bit_pos & 7
        val = int(packed[byte_pos]) >> bit_off
        if bit_off + bits > 8 and byte_pos + 1 < len(packed):
            val |= int(packed[byte_pos + 1]) << (8 - bit_off)
        indices[i] = val & mask

    return indices


def pack_indices_fast(indices: np.ndarray, bits: int) -> np.ndarray:
    """
    Vectorised packing for common widths (2, 4 bits) using numpy strides.
    Falls back to scalar version for odd widths.
    """
    if bits == 4:
        n = len(indices)
        padded = indices
        if n % 2 != 0:
            padded = np.append(indices, 0)
        pairs = padded.reshape(-1, 2).astype(np.uint8)
        packed = (pairs[:, 0] & 0x0F) | ((pairs[:, 1] & 0x0F) << 4)
        return packed.astype(np.uint8)

    if bits == 2:
        n = len(indices)
        remainder = (-n) % 4
        padded = np.append(indices, np.zeros(remainder, dtype=np.uint8)).astype(np.uint8)
        quads = padded.reshape(-1, 4)
        packed = (
            (quads[:, 0] & 0x03)
            | ((quads[:, 1] & 0x03) << 2)
            | ((quads[:, 2] & 0x03) << 4)
            | ((quads[:, 3] & 0x03) << 6)
        )
        return packed.astype(np.uint8)

    return pack_indices(indices, bits)


def unpack_indices_fast(packed: np.ndarray, n: int, bits: int) -> np.ndarray:
    """Vectorised unpacking for 2 and 4 bits."""
    if bits == 4:
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        unpacked = np.empty(len(packed) * 2, dtype=np.uint8)
        unpacked[0::2] = lo
        unpacked[1::2] = hi
        return unpacked[:n]

    if bits == 2:
        b0 = packed & 0x03
        b1 = (packed >> 2) & 0x03
        b2 = (packed >> 4) & 0x03
        b3 = (packed >> 6) & 0x03
        unpacked = np.empty(len(packed) * 4, dtype=np.uint8)
        unpacked[0::4] = b0
        unpacked[1::4] = b1
        unpacked[2::4] = b2
        unpacked[3::4] = b3
        return unpacked[:n]

    return unpack_indices(packed, n, bits)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for bits in (2, 3, 4, 8):
        n = 37
        orig = np.random.randint(0, 2**bits, size=n, dtype=np.uint16)
        if bits in (2, 4):
            packed = pack_indices_fast(orig.astype(np.uint8), bits)
            restored = unpack_indices_fast(packed, n, bits).astype(np.uint16)
        else:
            packed = pack_indices(orig, bits)
            restored = unpack_indices(packed, n, bits)
        assert np.array_equal(orig, restored), f"FAIL bits={bits}"
        print(f"bits={bits}: n={n} → packed={len(packed)} bytes "
              f"(ratio {n*bits/8/len(packed):.2f}) ✓")
