"""Runs the benchmark protocol and appends raw measurements to results/raw_times.csv.

Protocol (fixed before running):
  sizes        64, 128, 256, 512, 1024, 2048 (n = side of the n x n matrices)
  repetitions  5 measured repetitions per configuration
  warm-up      Java 3 runs (JIT compilation), C and Python 1 run (caches); warm-ups are not reported
  time budget  20 s for ONE multiplication (keeps one configuration, warm-ups included, under ~3 min)
  memory       2 GB of peak resident memory per process
Size escalation: n doubles, so the triple loop needs ~8x more work. If 8 x (median at the previous
size) exceeds the time budget, one single multiplication is first attempted with a 20 s timeout.
If it times out, the timeout is recorded in results/limits.csv and larger sizes are not attempted
for that language.

Usage:
  python3 scripts/run_benchmarks.py                      all languages, all sizes
  python3 scripts/run_benchmarks.py --lang c --n 1024    one configuration only
"""
import argparse
import csv
import os
import statistics
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SIZES = [64, 128, 256, 512, 1024, 2048]
REPS = 5
WARMUP = {"python": 1, "java": 3, "c": 1}
TIME_BUDGET_S = 20
MEM_BUDGET_KB = 2_000_000
CMD = {
    "python": ["python3", os.path.join(ROOT, "src/python/matmul.py")],
    "java": ["java", "-Xmx2g", "-cp", os.path.join(ROOT, "build/java"), "MatMul"],
    "c": [os.path.join(ROOT, "build/c/matmul")],
}
RAW = os.path.join(ROOT, "results/raw_times.csv")
LIMITS = os.path.join(ROOT, "results/limits.csv")
RAW_HEADER = ["language", "n", "rep", "batch", "time_per_mult_s", "checksum", "peak_rss_kb"]


def append(path, header, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def bench(lang, n, warmup, reps, timeout):
    out = subprocess.run(CMD[lang] + ["bench", str(n), str(warmup), str(reps), os.path.join(ROOT, "data")],
                         capture_output=True, text=True, check=True, timeout=timeout)
    return [line.split(",") for line in out.stdout.strip().splitlines()]


def previous_median(lang, n):
    if not os.path.exists(RAW):
        return None
    times = [float(r["time_per_mult_s"]) for r in csv.DictReader(open(RAW))
             if r["language"] == lang and int(r["n"]) == n]
    return statistics.median(times) if times else None


def run_config(lang, n, previous_median=None):
    """Returns the median time, or None if the size exceeds the budget."""
    if previous_median is not None and previous_median * 8 > TIME_BUDGET_S:
        try:
            bench(lang, n, 0, 1, TIME_BUDGET_S)           # probe: one multiplication, not reported
        except subprocess.TimeoutExpired:
            append(LIMITS, ["language", "n", "event", "limit"],
                   [[lang, n, "timeout (single multiplication not finished)", f"{TIME_BUDGET_S} s"]])
            print(f"{lang} n={n}: TIMEOUT after {TIME_BUDGET_S} s")
            return None
    rows = bench(lang, n, WARMUP[lang], REPS, (WARMUP[lang] + REPS) * TIME_BUDGET_S + 60)
    append(RAW, RAW_HEADER, rows)
    median = statistics.median(float(r[4]) for r in rows)
    print(f"{lang} n={n}: median {median:.6f} s, peak RSS {rows[0][6]} kB")
    if int(rows[0][6]) > MEM_BUDGET_KB:
        append(LIMITS, ["language", "n", "event", "limit"], [[lang, n, "memory budget exceeded", "2 GB"]])
        return None
    return median


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=list(CMD))
    p.add_argument("--n", type=int)
    args = p.parse_args()
    if args.lang and args.n:
        run_config(args.lang, args.n, previous_median(args.lang, args.n // 2))
        return
    for lang in (["c", "java", "python"] if not args.lang else [args.lang]):
        median = None
        for n in SIZES:
            median = run_config(lang, n, median)
            if median is None:
                break


if __name__ == "__main__":
    main()
