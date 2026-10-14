"""Builds the summary table and the figures of the report from results/raw_times.csv.

Outputs:
  results/summary.csv        per (language, n): repetitions, median, Q1, Q3, IQR, min, max,
                             time per inner iteration (median / n^3), peak RSS, array-storage estimate
  results/fig/time.png       median time vs n (log-log), IQR as error bars, n^3 reference line
  results/fig/per_iter.png   median time / n^3 (ns per inner-loop iteration)
  results/fig/memory.png     peak resident memory vs n
Quartiles: statistics.quantiles(..., n=4, method="inclusive") over the 5 measured repetitions.
"""
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "fig")
LANGS = [("c", "C"), ("java", "Java"), ("python", "Python")]
# bytes per stored element: C/Java double = 8; CPython list = 8-byte pointer + 24-byte float object
BYTES_PER_ELEMENT = {"c": 8, "java": 8, "python": 32}


def main():
    os.makedirs(FIG, exist_ok=True)
    raw = list(csv.DictReader(open(os.path.join(RES, "raw_times.csv"))))
    summary = []
    for lang, _ in LANGS:
        for n in sorted({int(r["n"]) for r in raw if r["language"] == lang}):
            rows = [r for r in raw if r["language"] == lang and int(r["n"]) == n]
            t = [float(r["time_per_mult_s"]) for r in rows]
            q1, med, q3 = statistics.quantiles(t, n=4, method="inclusive")
            summary.append({
                "language": lang, "n": n, "reps": len(t), "batch": rows[0]["batch"],
                "median_s": med, "q1_s": q1, "q3_s": q3, "iqr_s": q3 - q1, "min_s": min(t), "max_s": max(t),
                "ns_per_iteration": med / n ** 3 * 1e9,
                "peak_rss_MB": int(rows[0]["peak_rss_kb"]) / 1024,
                "array_storage_MB": 3 * n * n * BYTES_PER_ELEMENT[lang] / 2 ** 20,
                "checksum": rows[0]["checksum"],
            })
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        for s in summary:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in s.items()})

    def series(lang, key):
        sel = [s for s in summary if s["language"] == lang]
        return [s["n"] for s in sel], [s[key] for s in sel], sel

    # time vs n
    plt.figure(figsize=(6.2, 3.9))
    for lang, label in LANGS:
        ns, med, sel = series(lang, "median_s")
        err = [[s["median_s"] - s["q1_s"] for s in sel], [s["q3_s"] - s["median_s"] for s in sel]]
        plt.errorbar(ns, med, yerr=err, marker="o", capsize=3, label=label)
    ns, med, _ = series("c", "median_s")
    plt.plot(ns, [med[0] * (n / ns[0]) ** 3 for n in ns], "k:", label="n³ reference (from C, n=64)")
    plt.xscale("log", base=2); plt.yscale("log")
    plt.xlabel("matrix size n"); plt.ylabel("time per multiplication (s)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(FIG, "time.png"), dpi=150); plt.close()

    # time per inner iteration
    plt.figure(figsize=(6.2, 3.6))
    for lang, label in LANGS:
        ns, v, _ = series(lang, "ns_per_iteration")
        plt.plot(ns, v, marker="o", label=label)
    plt.xscale("log", base=2); plt.yscale("log")
    plt.xlabel("matrix size n"); plt.ylabel("median time / n³ (ns)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(FIG, "per_iter.png"), dpi=150); plt.close()

    # memory
    plt.figure(figsize=(6.2, 3.6))
    for lang, label in LANGS:
        ns, v, _ = series(lang, "peak_rss_MB")
        plt.plot(ns, v, marker="o", label=f"{label} (peak RSS)")
    plt.xscale("log", base=2); plt.yscale("log")
    plt.xlabel("matrix size n"); plt.ylabel("peak resident memory (MiB)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(FIG, "memory.png"), dpi=150); plt.close()
    print(open(os.path.join(RES, "summary.csv")).read())


if __name__ == "__main__":
    main()
