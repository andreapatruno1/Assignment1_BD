# Assignment 1 — Basic Matrix Multiplication in Python, Java and C

Big Data, ULPGC, 2026-2027 · Andrea Patruno

Research question: *How does the matrix size n affect the runtime and the memory use of the same basic
triple-loop matrix multiplication implemented in Python, Java and C, on one machine?*

The report (PDF) is submitted separately; its source is `report/report.md`. Repository: https://github.com/andreapatruno1/Assignment1_BD

## Benchmarking tools

| Language | Tool | Where |
|---|---|---|
| Java | JMH 1.35 (Maven) | `src/java/src/main/java/matmul/MatMulBenchmark.java` |
| Python | pytest-benchmark | `src/python/test_matmul_benchmark.py` |
| C | `perf stat` (program compiled with `gcc -O2`) | `scripts/run_benchmarks.py` |

## Contents

| Path | What it is |
|---|---|
| `src/python/matmul.py` | Python implementation (pure Python, flat list, i-j-k triple loop) |
| `src/python/test_matmul_benchmark.py` | pytest-benchmark test |
| `src/java/` | Maven project: `matmul/MatMul.java` (implementation), `matmul/MatMulBenchmark.java` (JMH) |
| `src/c/matmul.c` | C implementation (flat `double*`, i-j-k triple loop, kernel timer) |
| `scripts/gen_inputs.py` | generates the shared input matrices `data/A_n.bin`, `data/B_n.bin` |
| `scripts/build.sh` | builds `src/java/target/benchmarks.jar` and `build/c/matmul` |
| `scripts/verify.py` | correctness tests (known answers, identity, zero, NumPy reference) |
| `scripts/run_benchmarks.py` | runs JMH, pytest-benchmark and perf with the protocol below |
| `scripts/env_info.sh` | records the environment in `results/environment.txt` |
| `scripts/analyze.py` | summary tables and figures from the raw data |
| `results/raw_times.csv` | every measured repetition, all languages (raw data) |
| `results/tools/` | native output of the tools (JMH json, pytest-benchmark json, perf stat) |
| `results/perf_c.csv`, `results/perf_summary.csv` | perf hardware counters for C |
| `results/limits.csv` | timeouts / budget failures |
| `results/correctness.csv` | results of the correctness tests |
| `results/summary.csv` | median, Q1, Q3, IQR, min, max, ns per iteration, memory |
| `results/environment.txt` | CPU, RAM, OS, compiler, JVM and Python versions |
| `results/fig/` | figures used in the report |

## Requirements (Linux or WSL2, Ubuntu)

```bash
sudo apt install -y build-essential openjdk-21-jdk maven linux-tools-generic \
    python3-numpy python3-matplotlib python3-pytest python3-pytest-benchmark
sudo sysctl -w kernel.perf_event_paranoid=1      # lets perf read hardware counters (until reboot)
```

Tested with GCC 15.2.0, OpenJDK 21.0.12, Maven, JMH 1.35, CPython 3.14.4, pytest-benchmark,
NumPy and Matplotlib (`requirements.txt`). NumPy is used only to generate inputs and as correctness
reference, never inside the timed code. The memory metric reads `VmHWM` from `/proc/self/status` (Linux only).

## Exact commands

```bash
# 1. build (Java: mvn package -> src/java/target/benchmarks.jar ; C: gcc -O2 -std=c11 -Wall)
bash scripts/build.sh

# 2. generate the shared inputs (float64, little-endian, row-major, seed 42 + n, uniform [-1, 1))
python3 scripts/gen_inputs.py              # sizes 64 128 256 512 1024 2048 (~87 MB in data/)

# 3. correctness tests -> results/correctness.csv (must all pass)
python3 scripts/verify.py

# 4. environment -> results/environment.txt
bash scripts/env_info.sh

# 5. benchmarks -> results/raw_times.csv, results/limits.csv, results/perf_c.csv (about 10 minutes)
python3 scripts/run_benchmarks.py
#    or one configuration at a time:
python3 scripts/run_benchmarks.py --lang java --n 512

# 6. tables and figures -> results/summary.csv, results/perf_summary.csv, results/fig/
python3 scripts/analyze.py
```

Each tool can also be run directly (from the repository root):

```bash
java -jar src/java/target/benchmarks.jar MatMulBenchmark -p n=512 -p dataDir=data
(cd src/python && MATMUL_N=256 MATMUL_DATA=../../data python3 -m pytest test_matmul_benchmark.py --benchmark-only)
perf stat -e task-clock,cycles,instructions,cache-references,cache-misses build/c/matmul bench 512 1 5 data
```

## Protocol (fixed before running)

- Sizes n = 64, 128, 256, 512, 1024, 2048; 5 measured repetitions per configuration.
- Timed region: the triple loop only; input loading and allocation of A, B and C are outside it.
  - Java (JMH): `Mode.AverageTime`, 3 warm-up + 5 measurement iterations of 1 s, 1 fork (`-Xmx2g`);
    one value per iteration = average time per multiplication. The result is returned to JMH.
  - Python (pytest-benchmark): `benchmark.pedantic`, 1 warm-up round + 5 rounds; each round runs
    2^24 / n^3 multiplications for n ≤ 128 (64 at n = 64, 8 at n = 128), else 1.
  - C (perf): the program times each repetition with `clock_gettime(CLOCK_MONOTONIC)` (1 warm-up,
    5 measured, same batching as Python) and runs under `perf stat`, which records the hardware counters.
- After each sample the checksum (left-to-right sum of C) is computed or the result is consumed.
- Budgets: 20 s for one multiplication, 2 GB peak resident memory. If 8 x the previous median exceeds
  20 s, one multiplication is first attempted with a 20 s timeout (tool start-up included); on timeout the
  size is recorded in `results/limits.csv` and larger sizes are not attempted.
- Memory metric: peak resident set size (`VmHWM`) of the process that runs the kernel (JMH fork JVM,
  pytest process, C program), read at the end of the run.
