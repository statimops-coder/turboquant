"""
benchmark.py — Benchmark TurboQuant KV-cache compression vs baseline.

Measures:
  - Memory savings per vector (FP16 vs quantized, no sign overhead)
  - Cosine similarity and inner-product error (compression quality)
  - Tokens/sec when running a real MLX model (optional)

Usage:
    python3 benchmark.py [--no-model] [--bits 2 3 4] [--model mlx-community/Qwen2.5-0.5B-Instruct-4bit]
"""

import argparse
import gc
import sys
import time
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="TurboQuant Benchmark")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                        help="MLX model to benchmark")
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4],
                        help="Bit widths to benchmark")
    parser.add_argument("--prompt", default=(
        "The history of artificial intelligence begins in antiquity, with myths, "
        "stories and rumors of artificial beings endowed with intelligence or consciousness "
        "by master craftsmen. The seeds of modern AI were planted by philosophers who attempted "
        "to describe the process of human thinking as the mechanical manipulation of symbols."
    ), help="Test prompt for benchmarking")
    parser.add_argument("--n-tokens", type=int, default=60,
                        help="Number of tokens to generate")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip model benchmark, only run synthetic quality analysis")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Memory utilities
# ---------------------------------------------------------------------------

def get_memory_mb() -> float:
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return usage.ru_maxrss / 1024 / 1024
        else:
            return usage.ru_maxrss / 1024
    except Exception:
        return 0.0


def get_mlx_memory_mb() -> float:
    try:
        import mlx.core as mx
        return mx.metal.get_active_memory() / 1024 / 1024
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Synthetic compression quality analysis (no model needed)
# ---------------------------------------------------------------------------

def analyze_compression_quality(d: int = 128, n_vecs: int = 500,
                                  bits_list: List[int] = [2, 3, 4]) -> dict:
    """
    Generate random unit-norm vectors, compress/decompress, measure quality.
    """
    from turboquant import lloyd_max_quantizer, quantize_mse, dequantize_mse
    from utils import next_power_of_2

    print()
    print("=" * 72)
    print(f"  KV-Vector Compression Quality  (d={d}, n={n_vecs} vectors)")
    print("=" * 72)

    results = {}
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_vecs, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    # Random query vectors for inner-product error
    Q = rng.standard_normal((100, d))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    true_scores = X @ Q.T  # (n_vecs, 100)

    # Storage size constants
    fp32_bytes = d * 4
    fp16_bytes = d * 2

    print(f"\n  {'Config':<22} {'cos_sim':>8} {'cos_p5':>8} {'IP_MAE':>8} "
          f"{'B/vec':>7} {'vs FP16':>8} {'µs/vec':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

    # Baseline: FP16
    print(f"  {'FP16 (baseline)':<22} {'1.0000':>8} {'1.0000':>8} {'0.0000':>8} "
          f"{fp16_bytes:>7} {'1.0×':>8} {'—':>8}")

    for bits in bits_list:
        centroids = lloyd_max_quantizer(d, bits, cache=True)
        cos_sims = []
        approx_scores = np.zeros((n_vecs, 100))

        t0 = time.perf_counter()
        for i, x in enumerate(X):
            packed = quantize_mse(x, centroids, rotation_seed=i)
            x_rec  = dequantize_mse(packed, centroids, d, rotation_seed=i)
            cos_sims.append(
                float(np.dot(x, x_rec) / (np.linalg.norm(x_rec) + 1e-12))
            )
            approx_scores[i] = x_rec @ Q.T
        elapsed = time.perf_counter() - t0

        cos_arr = np.array(cos_sims)
        ip_err  = np.abs(true_scores - approx_scores)

        # Storage: packed indices only, no signs
        quant_bytes = (d * bits + 7) // 8  # packed indices
        seed_bytes  = 8                     # one int64 seed per vector
        total_bytes = quant_bytes + seed_bytes

        ratio = fp16_bytes / total_bytes

        results[bits] = {
            "cos_sim_mean": float(cos_arr.mean()),
            "cos_sim_p5": float(np.percentile(cos_arr, 5)),
            "ip_mae": float(ip_err.mean()),
            "ip_rmse": float(np.sqrt((ip_err**2).mean())),
            "us_per_vec": elapsed / n_vecs * 1e6,
            "bytes_per_vec": total_bytes,
            "compression_vs_fp16": ratio,
        }

        print(f"  {f'TurboQuant {bits}-bit':<22} "
              f"{cos_arr.mean():>8.4f} "
              f"{np.percentile(cos_arr, 5):>8.4f} "
              f"{ip_err.mean():>8.4f} "
              f"{total_bytes:>7} "
              f"{ratio:>7.1f}× "
              f"{elapsed/n_vecs*1e6:>8.1f}")

    print()
    print(f"  Note: 'B/vec' = packed bytes + 8B seed (no sign arrays stored)")
    print(f"        FP16={fp16_bytes}B/vec, FP32={fp32_bytes}B/vec")

    return results


# ---------------------------------------------------------------------------
# Model-based benchmark
# ---------------------------------------------------------------------------

def benchmark_with_model(model_id: str, bits_list: List[int],
                          prompt: str, n_tokens: int):
    """Load an MLX model and benchmark TurboQuant vs baseline inference."""
    import mlx.core as mx
    import mlx_lm
    from mlx_kv_hook import patch_model_kv_cache, TurboQuantKVCache

    print()
    print("=" * 72)
    print(f"  Model Benchmark: {model_id}")
    print(f"  Prompt: {prompt[:60]}...")
    print(f"  Generating {n_tokens} tokens per run")
    print("=" * 72)

    print(f"\nLoading model...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(model_id)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

    results = {}

    # Baseline (no compression)
    print("\n[Baseline — no compression]")
    mx.metal.clear_cache()
    gc.collect()
    mem_before = get_mlx_memory_mb()
    t0 = time.perf_counter()
    output = mlx_lm.generate(
        model, tokenizer, prompt=prompt,
        max_tokens=n_tokens, verbose=False,
    )
    baseline_time = time.perf_counter() - t0
    mem_after = get_mlx_memory_mb()

    baseline_tps = n_tokens / baseline_time
    results["baseline"] = {
        "tokens_per_sec": baseline_tps,
        "memory_delta_mb": mem_after - mem_before,
        "bits": 16,
    }
    print(f"  Speed: {baseline_tps:.1f} tok/s")
    print(f"  Memory delta: {mem_after - mem_before:.1f} MB")
    print(f"  Output: {output[:80]!r}")

    # TurboQuant at each bit width
    for bits in bits_list:
        print(f"\n[TurboQuant {bits}-bit]")

        patch_model_kv_cache(model, bits=bits, verbose=False)

        mx.metal.clear_cache()
        gc.collect()
        mem_before = get_mlx_memory_mb()

        t0 = time.perf_counter()
        try:
            output_tq = mlx_lm.generate(
                model, tokenizer, prompt=prompt,
                max_tokens=n_tokens, verbose=False,
            )
            tq_time = time.perf_counter() - t0
            mem_after = get_mlx_memory_mb()
            tq_tps = n_tokens / tq_time

            results[f"turbo_{bits}bit"] = {
                "tokens_per_sec": tq_tps,
                "memory_delta_mb": mem_after - mem_before,
                "bits": bits,
                "output_preview": output_tq[:80],
            }
            print(f"  Speed: {tq_tps:.1f} tok/s  ({tq_tps/baseline_tps*100:.0f}% of baseline)")
            print(f"  Memory delta: {mem_after - mem_before:.1f} MB")
            print(f"  Output: {output_tq[:80]!r}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[f"turbo_{bits}bit"] = {"error": str(e), "bits": bits}

        # Restore original make_cache for next run
        if hasattr(model, "_original_make_cache") and model._original_make_cache:
            model.make_cache = model._original_make_cache

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_model_summary(results: dict):
    print()
    print("=" * 72)
    print("  MODEL BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"  {'Config':<22} {'tok/s':>8} {'Mem ΔMB':>10} {'vs Baseline':>12}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*12}")

    baseline_tps = results.get("baseline", {}).get("tokens_per_sec", 1.0)

    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<22} {'ERROR':>8}")
            continue
        tps  = r.get("tokens_per_sec", 0)
        mem  = r.get("memory_delta_mb", 0)
        ratio = tps / baseline_tps if baseline_tps else 0
        print(f"  {name:<22} {tps:>8.1f} {mem:>10.1f} {ratio:>11.1%}")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # First run self-test to catch any bugs early
    print("Running turboquant self-test...")
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "turboquant.py"],
        capture_output=True, text=True,
        cwd=str(__import__("pathlib").Path(__file__).parent)
    )
    if result.returncode != 0:
        print("SELF-TEST FAILED:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    print("Self-test passed ✓")

    # Synthetic quality analysis (always runs, no GPU needed)
    analyze_compression_quality(d=128, n_vecs=500, bits_list=args.bits)

    if not args.no_model:
        print(f"\nRunning model benchmark with {args.model}...")
        print("(Pass --no-model to skip, or ^C to cancel)\n")
        try:
            import mlx.core as mx  # noqa — check import before proceeding
            import mlx_lm          # noqa
            results = benchmark_with_model(
                model_id=args.model,
                bits_list=args.bits,
                prompt=args.prompt,
                n_tokens=args.n_tokens,
            )
            print_model_summary(results)
        except KeyboardInterrupt:
            print("\nBenchmark interrupted.")
        except ImportError as e:
            print(f"\nModel benchmark skipped — missing dependency: {e}")
            print("Run: pip3 install mlx mlx-lm")
        except Exception as e:
            print(f"\nModel benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n(Model benchmark skipped — pass without --no-model to run it)")
