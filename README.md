# Assignment 1 — Basic Matrix Multiplication in Python, Java and C

Big Data, ULPGC, 2026-2027 · Andrea Patruno

Research question: *How does the matrix size n affect the runtime and the memory use of the same basic
triple-loop matrix multiplication implemented in Python, Java and C, on one machine?*

The report is `Assignment01_Patruno_Report.pdf`. Repository: https://github.com/andreapatruno1/Assignment1_BD

## Contents

| Path | What it is |
|---|---|
| `src/python/matmul.py` | Python implementation (pure Python, flat list, i-j-k triple loop) |
| `src/java/MatMul.java` | Java implementation (flat `double[]`, i-j-k triple loop) |
| `src/c/matmul.c` | C implementation (flat `double*`, i-j-k triple loop) |
| `scripts/gen_inputs.py` | generates the shared input matrices `data/A_n.bin`, `data/B_n.bin` |
| `scripts/build.sh` | compiles Java and C into `build/` |
| `scripts/verify.py` | correctness tests (known answers, identity, zero, NumPy reference) |
| `scripts/run_benchmarks.py` | benchmark protocol (sizes, repetitions, warm-up, budgets) |
| `scripts/env_info.sh` | records the environment in `results/environment.txt` |
| `scripts/analyze.py` | summary table and figures from the raw data |
| `results/raw_times.csv` | every measured repetition (raw data) |
| `results/limits.csv` | timeouts / budget failures |
| `results/correctness.csv` | results of the correctness tests |
| `results/summary.csv` | median, Q1, Q3, IQR, min, max, ns per iteration, memory |
| `results/environment.txt` | CPU, RAM, OS, compiler, JVM and Python versions |
| `results/fig/` | figures used in the report |

## Requirements

- Linux (the memory metric reads `VmHWM` from `/proc/self/status`)
- GCC (tested: 11.4.0), JDK 11 or newer (`javac`, `java`; tested runtime: OpenJDK 11.0.32)
- Python 3.10+ (tested: CPython 3.10.12) with the packages in `requirements.txt`
  (NumPy is used only to generate inputs and as correctness reference, never inside the timed code)

```bash
pip install -r requirements.txt
```

## Exact commands

```bash
# 1. build (C: gcc -O2 -std=c11 -Wall ; Java: javac --release 11)
bash scripts/build.sh

# 2. generate the shared inputs (float64, little-endian, row-major, seed 42 + n, uniform [-1, 1))
python3 scripts/gen_inputs.py              # sizes 64 128 256 512 1024 2048 (~87 MB in data/)

# 3. correctness tests -> results/correctness.csv (must all pass)
python3 scripts/verify.py

# 4. environment -> results/environment.txt
bash scripts/env_info.sh

# 5. benchmarks -> results/raw_times.csv and results/limits.csv (about 15 minutes)
python3 scripts/run_benchmarks.py
#    or one configuration at a time:
python3 scripts/run_benchmarks.py --lang c --n 1024

# 6. tables and figures -> results/summary.csv and results/fig/
python3 scripts/analyze.py
```

Each implementation can also be run directly:

```bash
python3 src/python/matmul.py bench <n> <warmup> <reps> data
java -Xmx2g -cp build/java MatMul bench <n> <warmup> <reps> data
build/c/matmul bench <n> <warmup> <reps> data
# output CSV: language,n,rep,batch,time_per_mult_s,checksum,peak_rss_kb
```

## Protocol (fixed before running)

- Sizes n = 64, 128, 256, 512, 1024, 2048; 5 measured repetitions per configuration.
- Warm-up runs (not reported): Java 3, C 1, Python 1.
- Timer: monotonic (`clock_gettime(CLOCK_MONOTONIC)`, `System.nanoTime`, `time.perf_counter_ns`) around
  the triple loop only. Input loading and allocation of A, B and C are outside the timed region.
- n ≤ 128: each sample times a batch of 2^24 / n^3 multiplications (64 at n = 64, 8 at n = 128) and
  reports time per multiplication.
- After each sample the checksum (left-to-right sum of C) is computed, so the result is consumed.
- Budgets: 20 s for one multiplication, 2 GB peak resident memory. If 8 x the previous median exceeds
  20 s, one multiplication is first attempted with a 20 s timeout; on timeout the size is recorded in
  `results/limits.csv` and larger sizes are not attempted.
- Memory metric: peak resident set size of the process (`VmHWM`), read by each program itself at the end.

