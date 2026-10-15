"""pytest-benchmark for the pure-Python triple loop (kernel only: inputs are loaded before timing).

Configured through environment variables (set by scripts/run_benchmarks.py):
  MATMUL_N       matrix size n
  MATMUL_DATA    folder with A_n.bin / B_n.bin            (default: data)
  MATMUL_ROUNDS  measured rounds                          (default: 5)
  MATMUL_WARMUP  warm-up rounds, not reported             (default: 1)
Each round runs batch_size(n) multiplications (64 at n = 64, 8 at n = 128, else 1);
pytest-benchmark reports the time per multiplication.

Run: MATMUL_N=256 pytest src/python/test_matmul_benchmark.py --benchmark-only --benchmark-json=out.json
"""
import os

from matmul import batch_size, checksum, load, multiply, peak_rss_kb

N = int(os.environ.get("MATMUL_N", "64"))
DATA = os.environ.get("MATMUL_DATA", "data")
ROUNDS = int(os.environ.get("MATMUL_ROUNDS", "5"))
WARMUP = int(os.environ.get("MATMUL_WARMUP", "1"))


def test_matmul(benchmark):
    a = load(f"{DATA}/A_{N}.bin", N)
    b = load(f"{DATA}/B_{N}.bin", N)
    c = [0.0] * (N * N)
    benchmark.pedantic(multiply, args=(a, b, c, N), rounds=ROUNDS, iterations=batch_size(N),
                       warmup_rounds=WARMUP)
    benchmark.extra_info["n"] = N
    benchmark.extra_info["checksum"] = f"{checksum(c):.12e}"      # consumes the result
    benchmark.extra_info["peak_rss_kb"] = peak_rss_kb()
