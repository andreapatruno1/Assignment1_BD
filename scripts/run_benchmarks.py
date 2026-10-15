"""Runs the benchmark protocol with the course tools and writes results/raw_times.csv.

Tools (as in the course material):
  Java    JMH 1.35            (src/java/target/benchmarks.jar, class MatMulBenchmark)
  Python  pytest-benchmark    (src/python/test_matmul_benchmark.py)
  C       perf stat           (build/c/matmul, compiled with gcc -O2; kernel time from clock_gettime
                               inside the program, hardware counters from perf)

Protocol (fixed before running):
  sizes        64, 128, 256, 512, 1024, 2048 (n = side of the n x n matrices)
  repetitions  5 measured repetitions per configuration
  warm-up      Java 3 JMH iterations of 1 s, C and Python 1 run; warm-ups are not reported
  time budget  20 s for ONE multiplication; memory budget 2 GB peak resident memory
Size escalation: if 8 x (median at the previous size) exceeds 20 s, one single multiplication is first
attempted with a 20 s timeout (tool start-up included). If it times out, the timeout is recorded in
results/limits.csv and larger sizes are not attempted for that language.

Outputs:
  results/raw_times.csv   language,n,rep,batch,time_per_mult_s,checksum,peak_rss_kb
  results/limits.csv      timeouts and budget failures
  results/perf_c.csv      perf stat counters of each C configuration
  results/tools/          native output of each tool (JMH json, pytest-benchmark json, perf stat)

Usage:
  python3 scripts/run_benchmarks.py                      all languages, all sizes
  python3 scripts/run_benchmarks.py --lang c --n 1024    one configuration only
"""
import argparse
import csv
import glob
import json
import os
import shutil
import signal
import statistics
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
TOOLS = os.path.join(RES, "tools")
SIZES = [64, 128, 256, 512, 1024, 2048]
REPS = 5
TIME_BUDGET_S = 20
MEM_BUDGET_KB = 2_000_000
RAW = os.path.join(RES, "raw_times.csv")
LIMITS = os.path.join(RES, "limits.csv")
PERF_CSV = os.path.join(RES, "perf_c.csv")
RAW_HEADER = ["language", "n", "rep", "batch", "time_per_mult_s", "checksum", "peak_rss_kb"]
PERF_EVENTS = "task-clock,cycles,instructions,cache-references,cache-misses"
JMH_JAR = os.path.join(ROOT, "src/java/target/benchmarks.jar")


def append(path, header, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def run(cmd, timeout, **kw):
    """subprocess.run with a timeout that also kills child processes (e.g. the JVM forked by JMH)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True, **kw)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    if proc.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{err}")
    return out


def batch_size(n):
    return max(1, (1 << 24) // (n * n * n))


def perf_binary():
    # on WSL2 /usr/bin/perf is a wrapper that refuses the Microsoft kernel: use the real binary
    candidates = sorted(glob.glob("/usr/lib/linux-tools/*/perf") + glob.glob("/usr/lib/linux-tools-*/perf"))
    if candidates:
        return candidates[-1]
    found = shutil.which("perf")
    if not found:
        raise SystemExit("perf not found: install it (see README)")
    return found


# ---------- C: perf stat ----------
def run_c(n, probe, timeout):
    exe = os.path.join(ROOT, "build/c/matmul")
    if probe:
        run([exe, "bench", str(n), "0", "1", DATA], timeout)
        return None
    perf_out = os.path.join(TOOLS, f"perf_c_{n}.txt")
    out = run([perf_binary(), "stat", "-x", ",", "-o", perf_out, "-e", PERF_EVENTS, "--",
               exe, "bench", str(n), "1", str(REPS), DATA], timeout)
    counters = {}
    for line in open(perf_out):
        parts = line.strip().split(",")
        if line.startswith("#") or len(parts) < 3 or not parts[2]:
            continue
        event = parts[2].split(":")[0]
        if "/" in event:                       # hybrid CPUs report e.g. cpu_core/cycles/ and cpu_atom/cycles/
            event = event.split("/")[1]
        try:
            counters[event] = counters.get(event, 0.0) + float(parts[0])
        except ValueError:                     # "<not supported>" or "<not counted>"
            counters.setdefault(event, parts[0])
    multiplications = (1 + REPS) * batch_size(n)
    append(PERF_CSV, ["n", "multiplications"] + PERF_EVENTS.split(","),
           [[n, multiplications] + [counters.get(e, "") for e in PERF_EVENTS.split(",")]])
    return [line.split(",") for line in out.strip().splitlines()]


# ---------- Java: JMH ----------
def run_java(n, probe, timeout):
    # JMH runs inside results/tools with relative paths: -jvmArgsAppend splits on spaces,
    # and the project path contains spaces.
    info, json_out = f"jmh_info_{n}.csv", f"jmh_{n}.json"
    data_rel = os.path.relpath(DATA, TOOLS)
    cmd = ["java", "-jar", JMH_JAR, "MatMulBenchmark", "-p", f"n={n}", "-p", f"dataDir={data_rel}",
           "-rf", "json", "-rff", json_out, "-jvmArgsAppend", f"-Dmatmul.info={info}"]
    if probe:
        cmd += ["-wi", "0", "-i", "1"]
    if os.path.exists(os.path.join(TOOLS, info)):
        os.remove(os.path.join(TOOLS, info))
    log = run(cmd, timeout, cwd=TOOLS)
    if probe:
        return None
    results = json.load(open(os.path.join(TOOLS, json_out)))
    if not results:
        raise SystemExit("JMH produced no result:\n" + log[-3000:])
    values = [v for fork in results[0]["primaryMetric"]["rawData"] for v in fork]   # s/op per iteration
    n_info, checksum, rss = open(os.path.join(TOOLS, info)).read().strip().splitlines()[-1].split(",")
    return [["java", n, r, "auto", f"{v:.9f}", checksum, rss] for r, v in enumerate(values)]


# ---------- Python: pytest-benchmark ----------
def run_python(n, probe, timeout):
    json_out = os.path.join(TOOLS, f"pytest_{n}.json")
    env = dict(os.environ, MATMUL_N=str(n), MATMUL_DATA=DATA,
               MATMUL_ROUNDS="1" if probe else str(REPS), MATMUL_WARMUP="0" if probe else "1")
    run(["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         os.path.join(ROOT, "src/python/test_matmul_benchmark.py"),
         "--benchmark-only", f"--benchmark-json={json_out}"],
        timeout, env=env, cwd=os.path.join(ROOT, "src/python"))
    if probe:
        return None
    bench = json.load(open(json_out))["benchmarks"][0]
    info = bench["extra_info"]
    return [["python", n, r, bench["stats"]["iterations"], f"{v:.9f}", info["checksum"], info["peak_rss_kb"]]
            for r, v in enumerate(bench["stats"]["data"])]


RUNNERS = {"c": run_c, "java": run_java, "python": run_python}


def previous_median(lang, n):
    if not os.path.exists(RAW):
        return None
    times = [float(r["time_per_mult_s"]) for r in csv.DictReader(open(RAW))
             if r["language"] == lang and int(r["n"]) == n]
    return statistics.median(times) if times else None


def run_config(lang, n, prev_median=None):
    """Returns the median time, or None if the size exceeds the budget."""
    os.makedirs(TOOLS, exist_ok=True)
    if prev_median is not None and prev_median * 8 > TIME_BUDGET_S:
        try:
            RUNNERS[lang](n, True, TIME_BUDGET_S)           # probe: one multiplication, not reported
        except subprocess.TimeoutExpired:
            append(LIMITS, ["language", "n", "event", "limit"],
                   [[lang, n, "timeout (single multiplication not finished)", f"{TIME_BUDGET_S} s"]])
            print(f"{lang} n={n}: TIMEOUT after {TIME_BUDGET_S} s")
            return None
    rows = RUNNERS[lang](n, False, 10 * REPS * TIME_BUDGET_S)
    append(RAW, RAW_HEADER, rows)
    median = statistics.median(float(r[4]) for r in rows)
    print(f"{lang} n={n}: median {median:.6f} s, peak RSS {rows[0][6]} kB")
    if int(rows[0][6]) > MEM_BUDGET_KB:
        append(LIMITS, ["language", "n", "event", "limit"], [[lang, n, "memory budget exceeded", "2 GB"]])
        return None
    return median


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=list(RUNNERS))
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
