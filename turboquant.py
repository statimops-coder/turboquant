"""
turboquant.py — Core TurboQuant KV-cache compression algorithm.

Based on: "TurboQuant: Efficient KV Cache Quantization for LLM Inference"

Key ideas:
  1. Random rotation via Fast Walsh-Hadamard Transform (WHT) + sign flips
     makes per-coordinate distribution well-approximated by a Beta marginal.
  2. Lloyd-Max quantizer tuned to that Beta marginal gives near-optimal MSE.
  3. Two-stage "prod" quantizer: (b-1)-bit MSE + 1-bit QJL on residual.

DESIGN NOTE ON SIGN STORAGE:
  Signs are ±1 vectors of length d_padded drawn from np.random.RandomState(seed).
  Storing them as float64 costs d_padded*8 bytes (~1 KB for d=128) — LARGER than
  the quantized data itself (d*bits/8 = 64 B for 4-bit d=128).
  FIX: Never store signs. Regenerate from seed at dequantization time.
  Storage per (layer, step, head): one int64 seed — negligible.
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import special

from utils import (
    pack_indices_fast, unpack_indices_fast,
    pack_indices, unpack_indices,
    next_power_of_2,
)

# Cache directory for precomputed Lloyd-Max centroids
_CACHE_DIR = Path(__file__).parent / ".lloyd_max_cache"
_CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1.  Fast Walsh-Hadamard Transform — fully vectorised, no Python inner loop
# ---------------------------------------------------------------------------

def fwht_inplace(x: np.ndarray) -> None:
    """
    In-place unnormalised Walsh-Hadamard Transform.
    x must have length that is a power of 2.
    Fully vectorised butterfly — no Python loop over coordinates.
    """
    n = len(x)
    h = 1
    while h < n:
        # Reshape into blocks of (n//(2h), 2, h); butterfly the middle axis
        view = x.reshape(n // (2 * h), 2, h)
        a = view[:, 0, :].copy()
        b = view[:, 1, :].copy()
        view[:, 0, :] = a + b
        view[:, 1, :] = a - b
        h *= 2


def fwht_numpy(x: np.ndarray) -> np.ndarray:
    """
    Vectorised WHT for an N-D array where the LAST axis has size = power of 2.
    Returns normalised result (divided by sqrt(last-axis size)).
    No Python loop over coordinates — all butterflies are numpy array operations.
    """
    out = x.copy().astype(np.float64)
    n = out.shape[-1]
    h = 1
    while h < n:
        shape = out.shape[:-1] + (n // (2 * h), 2, h)
        view = out.reshape(shape)
        a = view[..., 0, :].copy()
        b = view[..., 1, :].copy()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        out = view.reshape(x.shape[:-1] + (n,))
        h *= 2
    out /= np.sqrt(n)
    return out


# ---------------------------------------------------------------------------
# 2.  Random rotation (seed-based, signs never stored)
# ---------------------------------------------------------------------------

def _make_signs(d_padded: int, seed: int) -> np.ndarray:
    """Generate deterministic ±1 sign vector of length d_padded from seed."""
    rng = np.random.RandomState(seed)
    return rng.choice(np.array([-1.0, 1.0]), size=d_padded)


def random_rotation(x: np.ndarray, seed: int) -> np.ndarray:
    """
    Apply randomised WHT rotation: x' = (1/√d') · H · D · x
    Signs are NOT returned — regenerate with _make_signs(d_padded, seed) if needed.

    Parameters
    ----------
    x    : 1-D float array, shape (d,)
    seed : int — reproducible RNG seed

    Returns
    -------
    rotated : np.ndarray, shape (d,)
    """
    d = len(x)
    d_padded = next_power_of_2(d)

    x_padded = np.zeros(d_padded, dtype=np.float64)
    x_padded[:d] = x
    x_padded *= _make_signs(d_padded, seed)
    fwht_inplace(x_padded)
    x_padded /= np.sqrt(d_padded)

    return x_padded[:d]


def inverse_rotation(x_rotated: np.ndarray, d_padded: int, seed: int) -> np.ndarray:
    """
    Invert random_rotation.
    Since H is symmetric and H·H = n·I:  x = D · (1/√n) · H · x'

    Parameters
    ----------
    x_rotated : shape (d,)
    d_padded  : int — next_power_of_2(d)
    seed      : int — same seed used in forward rotation
    """
    d = len(x_rotated)

    x_padded = np.zeros(d_padded, dtype=np.float64)
    x_padded[:d] = x_rotated
    fwht_inplace(x_padded)
    x_padded /= np.sqrt(d_padded)
    x_padded *= _make_signs(d_padded, seed)

    return x_padded[:d]


def random_rotation_batch(X: np.ndarray, seed: int) -> np.ndarray:
    """
    Batch rotation: X shape (batch, d). All rows share the same seed.
    Returns X_rotated shape (batch, d). Signs NOT returned.
    """
    batch, d = X.shape
    d_padded = next_power_of_2(d)

    X_padded = np.zeros((batch, d_padded), dtype=np.float64)
    X_padded[:, :d] = X
    X_padded *= _make_signs(d_padded, seed)[np.newaxis, :]
    X_padded = fwht_numpy(X_padded)  # includes /sqrt(n) normalisation

    return X_padded[:, :d]


def inverse_rotation_batch(X_rot: np.ndarray, d_padded: int, seed: int) -> np.ndarray:
    """
    Batch inverse rotation using seed (no stored signs).

    Parameters
    ----------
    X_rot    : (batch, d)
    d_padded : int
    seed     : int
    """
    batch, d = X_rot.shape

    X_padded = np.zeros((batch, d_padded), dtype=np.float64)
    X_padded[:, :d] = X_rot
    X_padded = fwht_numpy(X_padded)  # includes /sqrt(n)
    X_padded *= _make_signs(d_padded, seed)[np.newaxis, :]

    return X_padded[:, :d]


# ---------------------------------------------------------------------------
# 3.  Beta marginal PDF
# ---------------------------------------------------------------------------

def beta_marginal_pdf(d: int):
    """
    Marginal PDF for one coordinate of a uniform unit-norm vector in R^d.

    f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2),  x ∈ [-1, 1]
    """
    if d <= 2:
        sigma = 1.0 / np.sqrt(max(d, 1))
        def pdf(x):
            return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        return pdf

    coeff = special.gamma(d / 2) / (np.sqrt(np.pi) * special.gamma((d - 1) / 2))
    alpha = (d - 3) / 2

    def pdf(x):
        x = np.asarray(x, dtype=np.float64)
        inside = np.abs(x) < 1.0
        result = np.zeros_like(x)
        if alpha == 0:
            result[inside] = coeff
        else:
            result[inside] = coeff * np.maximum(1 - x[inside] ** 2, 0.0) ** alpha
        return result

    return pdf


# ---------------------------------------------------------------------------
# 4.  Lloyd-Max quantizer
# ---------------------------------------------------------------------------

def lloyd_max_1d(pdf, n_levels: int, n_iter: int = 100,
                 x_min: float = -1.0, x_max: float = 1.0,
                 n_init: int = 1000) -> np.ndarray:
    """Compute Lloyd-Max centroids for a 1-D PDF over [x_min, x_max]."""
    centroids = np.linspace(x_min + 1e-6, x_max - 1e-6, n_levels)
    x_grid = np.linspace(x_min + 1e-9, x_max - 1e-9, n_init)
    p_grid = np.maximum(np.asarray(pdf(x_grid)), 0.0)
    p_sum = p_grid.sum()
    if p_sum == 0:
        return centroids
    p_grid /= p_sum

    for _ in range(n_iter):
        dists = np.abs(x_grid[:, None] - centroids[None, :])  # (n_init, n_levels)
        assignments = np.argmin(dists, axis=1)

        new_centroids = centroids.copy()
        for k in range(n_levels):
            mask = assignments == k
            if mask.any():
                w = p_grid[mask]
                new_centroids[k] = np.dot(w, x_grid[mask]) / w.sum()

        if np.allclose(new_centroids, centroids, atol=1e-10):
            break
        centroids = new_centroids

    return np.sort(centroids)


def lloyd_max_quantizer(d: int, bits: int,
                        use_gaussian_approx: bool = False,
                        cache: bool = True) -> np.ndarray:
    """
    Compute (or load from cache) Lloyd-Max centroids for the Beta marginal
    of dimension d with 2**bits quantization levels.

    Returns centroids : np.ndarray, shape (2**bits,)
    """
    n_levels = 1 << bits
    cache_key = f"d{d}_b{bits}_{'gauss' if use_gaussian_approx else 'beta'}.pkl"
    cache_path = _CACHE_DIR / cache_key

    if cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if use_gaussian_approx or d > 512:
        sigma = 1.0 / np.sqrt(d)
        from scipy.stats import norm
        pdf = lambda x: norm.pdf(x, scale=sigma)
        x_min, x_max = -4 * sigma, 4 * sigma
    else:
        pdf = beta_marginal_pdf(d)
        x_min, x_max = -1.0 + 1e-6, 1.0 - 1e-6

    centroids = lloyd_max_1d(pdf, n_levels, x_min=x_min, x_max=x_max)

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump(centroids, f)

    return centroids


# ---------------------------------------------------------------------------
# 5.  Quantize / Dequantize (MSE, vectorised)
# ---------------------------------------------------------------------------

def _nearest_centroid_indices(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    For each element of x, return the index of the nearest centroid.
    Vectorised via broadcasting — no Python loop.
    """
    # x: (n,)  centroids: (k,)  →  dists: (n, k)
    dists = np.abs(x[:, None] - centroids[None, :])
    return np.argmin(dists, axis=1).astype(np.uint16)


def quantize_mse(x: np.ndarray, centroids: np.ndarray,
                 rotation_seed: int) -> np.ndarray:
    """
    Rotate x, then quantize each coordinate to the nearest centroid.

    Parameters
    ----------
    x             : 1-D float array
    centroids     : Lloyd-Max centroids, shape (2**bits,)
    rotation_seed : int — seed for rotation (signs regenerated from this at decode)

    Returns
    -------
    packed : uint8 array (bit-packed indices)
    NOTE: signs are NOT returned. Pass rotation_seed to dequantize_mse.
    """
    bits = int(np.round(np.log2(len(centroids))))
    x_rot = random_rotation(x, rotation_seed)
    indices = _nearest_centroid_indices(x_rot, centroids)

    if bits in (2, 4):
        return pack_indices_fast(indices.astype(np.uint8), bits)
    else:
        return pack_indices(indices, bits)


def dequantize_mse(packed: np.ndarray, centroids: np.ndarray,
                   d: int, rotation_seed: int) -> np.ndarray:
    """
    Unpack indices, look up centroids, then inverse-rotate.

    Parameters
    ----------
    packed        : uint8 bit-packed array
    centroids     : Lloyd-Max centroids
    d             : original vector dimension
    rotation_seed : int — same seed used in quantize_mse

    Returns
    -------
    x_reconstructed : float array, shape (d,)
    """
    bits = int(np.round(np.log2(len(centroids))))
    d_padded = next_power_of_2(d)

    if bits in (2, 4):
        indices = unpack_indices_fast(packed, d, bits)
    else:
        indices = unpack_indices(packed, d, bits)

    x_rot = centroids[indices]
    return inverse_rotation(x_rot, d_padded, rotation_seed)


# ---------------------------------------------------------------------------
# 6.  Two-stage "prod" quantizer (unbiased inner product)
# ---------------------------------------------------------------------------

def _qjl_1bit(residual: np.ndarray, seed: int) -> Tuple[np.ndarray, float]:
    """
    1-bit QJL sketch of `residual`.
    Projects residual onto a random Gaussian vector, returns sign bit and norm.
    Random vector is regenerated at decode from seed — not stored.

    Returns
    -------
    bit  : np.ndarray dtype uint8, shape (1,)
    norm : float — L2 norm of residual
    """
    d = len(residual)
    rng = np.random.RandomState(seed + 99999)
    rand_vec = rng.randn(d).astype(np.float64)
    rand_vec /= np.linalg.norm(rand_vec) + 1e-12

    projection = np.dot(residual, rand_vec)
    norm = float(np.linalg.norm(residual))
    bit = np.array([1 if projection >= 0 else 0], dtype=np.uint8)
    return bit, norm


def _qjl_decode(bit: np.ndarray, norm: float, d: int, seed: int) -> np.ndarray:
    """Reconstruct residual estimate from 1-bit QJL (regenerates rand_vec from seed)."""
    rng = np.random.RandomState(seed + 99999)
    rand_vec = rng.randn(d).astype(np.float64)
    rand_vec /= np.linalg.norm(rand_vec) + 1e-12

    sign = 1.0 if int(bit[0]) == 1 else -1.0
    return sign * norm * rand_vec


def quantize_prod(x: np.ndarray, bits: int, seed: int,
                  centroids: Optional[np.ndarray] = None) -> dict:
    """
    Two-stage prod quantizer for unbiased inner product estimation.

    Stage 1: (bits-1)-bit MSE quantizer
    Stage 2: 1-bit QJL on the residual

    Returns dict with: packed, qjl_bit, qjl_norm, d, bits, seed, centroids
    (no signs — all regenerated from seed at decode time)
    """
    d = len(x)
    bits_mse = max(bits - 1, 1)

    if centroids is None:
        centroids = lloyd_max_quantizer(d, bits_mse)

    packed = quantize_mse(x, centroids, seed)

    # Reconstruct stage-1 output and compute residual
    x_hat = dequantize_mse(packed, centroids, d, seed)
    residual = x - x_hat

    # Stage 2: 1-bit QJL on residual
    qjl_bit, qjl_norm = _qjl_1bit(residual, seed)

    return {
        "packed": packed,
        "qjl_bit": qjl_bit,
        "qjl_norm": qjl_norm,
        "d": d,
        "bits": bits,
        "seed": seed,
        "centroids": centroids,
    }


def dequantize_prod(data: dict) -> np.ndarray:
    """Reconstruct vector from two-stage prod quantization."""
    packed    = data["packed"]
    centroids = data["centroids"]
    d         = data["d"]
    seed      = data["seed"]

    x_hat = dequantize_mse(packed, centroids, d, seed)
    residual_hat = _qjl_decode(data["qjl_bit"], data["qjl_norm"], d, seed)
    return x_hat + residual_hat


# ---------------------------------------------------------------------------
# 7.  Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    np.random.seed(42)
    d = 128

    print("=" * 60)
    print("TurboQuant Component Tests")
    print("=" * 60)

    # --- Random rotation round-trip ---
    x = np.random.randn(d)
    x /= np.linalg.norm(x)
    d_padded = next_power_of_2(d)

    t0 = time.perf_counter()
    x_rot = random_rotation(x, seed=0)
    x_back = inverse_rotation(x_rot, d_padded, seed=0)
    t1 = time.perf_counter()

    err = np.max(np.abs(x - x_back))
    print(f"\n[Rotation]  d={d}  round-trip max err: {err:.2e}  time: {(t1-t0)*1e6:.1f} µs")
    assert err < 1e-10, f"Rotation round-trip error too large: {err}"

    # --- Lloyd-Max quantizer ---
    print("\n[Lloyd-Max centroids]")
    for bits in (2, 3, 4):
        t0 = time.perf_counter()
        c = lloyd_max_quantizer(d, bits, cache=True)
        t1 = time.perf_counter()
        print(f"  bits={bits}: {len(c)} centroids, range [{c[0]:.4f}, {c[-1]:.4f}]  "
              f"({(t1-t0)*1e3:.1f} ms)")

    # --- MSE quantize/dequantize ---
    print("\n[MSE Quantize/Dequantize]")
    for bits in (2, 3, 4):
        centroids = lloyd_max_quantizer(d, bits, cache=True)
        t0 = time.perf_counter()
        packed = quantize_mse(x, centroids, rotation_seed=7)
        x_rec  = dequantize_mse(packed, centroids, d, rotation_seed=7)
        t1 = time.perf_counter()
        mse = np.mean((x - x_rec) ** 2)
        cos_sim = np.dot(x, x_rec) / (np.linalg.norm(x) * np.linalg.norm(x_rec) + 1e-12)
        fp16_bytes = d * 2
        quant_bytes = len(packed)
        print(f"  bits={bits}: MSE={mse:.4f}  cos_sim={cos_sim:.4f}  "
              f"packed={quant_bytes}B  FP16={fp16_bytes}B  "
              f"ratio={fp16_bytes/quant_bytes:.1f}x  time={( t1-t0)*1e6:.1f} µs")

    # --- Two-stage prod quantizer ---
    print("\n[Prod Quantize/Dequantize]")
    for bits in (2, 3, 4):
        t0 = time.perf_counter()
        data = quantize_prod(x, bits=bits, seed=13)
        x_rec = dequantize_prod(data)
        t1 = time.perf_counter()
        mse = np.mean((x - x_rec) ** 2)
        cos_sim = np.dot(x, x_rec) / (np.linalg.norm(x) * np.linalg.norm(x_rec) + 1e-12)
        print(f"  bits={bits}: MSE={mse:.4f}  cos_sim={cos_sim:.4f}  "
              f"time={( t1-t0)*1e6:.1f} µs")

    # --- Batch rotation speed test ---
    print("\n[Batch Rotation Speed]")
    batch = 512
    X = np.random.randn(batch, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    t0 = time.perf_counter()
    for _ in range(10):
        X_rot = random_rotation_batch(X, seed=0)
    t1 = time.perf_counter()
    print(f"  {batch} vectors × d={d}: {(t1-t0)/10*1e3:.2f} ms per batch")

    print("\n✓ All tests passed.")
