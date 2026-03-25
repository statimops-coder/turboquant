# TurboQuant

KV-cache compression for local LLM inference on Apple Silicon. Based on the [TurboQuant paper](https://arxiv.org/abs/2504.19874) (Google Research, 2025).

## What it does

Compresses the KV-cache during LLM inference by 4-6x with minimal quality loss. This means:

- **16GB Mac**: Run 14B models instead of 8B
- **24GB Mac**: Run 32B models instead of 14B  
- **Any hardware**: 5x longer context windows at the same memory

## How it works

1. **Random rotation** via fast Walsh-Hadamard transform — spreads information uniformly across coordinates
2. **Optimal scalar quantization** per coordinate — Lloyd-Max quantizer tuned to the exact mathematical distribution after rotation
3. **Seed-based reconstruction** — rotation parameters regenerated from seed, not stored (zero overhead)

## Results

Benchmarked on Apple M4 with Qwen2.5-0.5B:

| Bits | Compression | Cosine Similarity | Quantize Speed |
|------|-------------|-------------------|----------------|
| 2    | 6.4x        | 0.942             | 0.10 ms/vec    |
| 3    | 4.6x        | 0.984             | 0.10 ms/vec    |
| 4    | 3.6x        | 0.995             | 0.09 ms/vec    |

The paper reports quality-neutral results at 3.5 bits/channel. Our implementation matches this — 3-bit achieves 98.4% cosine similarity.

## Quick start

```bash
pip install numpy scipy
python benchmark.py
```

### Basic usage

```python
from turboquant import quantize_mse, dequantize_mse, lloyd_max_quantizer
import numpy as np

dim = 128  # KV-cache head dimension
bits = 3

# Precompute quantizer (do once per dimension)
centroids = lloyd_max_quantizer(dim, bits)

# Quantize a vector
x = np.random.randn(dim)
x = x / np.linalg.norm(x)
seed = 42

packed = quantize_mse(x, centroids, seed)
x_hat = dequantize_mse(packed, centroids, dim, seed)

print(f"Compression: {dim * 2 / len(packed):.1f}x")
print(f"Cosine similarity: {np.dot(x, x_hat) / (np.linalg.norm(x) * np.linalg.norm(x_hat)):.4f}")
```

## Architecture

```
Input vector (FP16, d dimensions)
    │
    ▼
Random sign flip (seed-based, regenerated at decode)
    │
    ▼
Fast Walsh-Hadamard Transform
    │
    ▼
Per-coordinate Lloyd-Max quantization (b bits each)
    │
    ▼
Bit-packed output (d×b bits + 4 bytes seed)
```

Decompression reverses the process: unpack → lookup centroids → inverse WHT → inverse sign flip. All deterministic from the seed.

## Implementation details

- **No stored rotation matrices**: Signs regenerated from seed at decode time (zero memory overhead)
- **Vectorized numpy**: All operations are batched, no Python loops over coordinates  
- **Cached quantizers**: Lloyd-Max centroids computed once per (dimension, bits) pair
- **Beta marginal distribution**: Exact density function used for optimal centroid placement, not Gaussian approximation

## Files

| File | Description |
|------|-------------|
| `turboquant.py` | Core algorithm — rotation, quantization, bit packing |
| `utils.py` | Helper functions |
| `benchmark.py` | Benchmark script with MLX model integration |

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — Daliri et al., Google Research, 2025
- [Fast Walsh-Hadamard Transform](https://en.wikipedia.org/wiki/Hadamard_transform) — used for random rotation

## License

MIT
