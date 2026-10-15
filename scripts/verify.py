"""Correctness tests for the three implementations (run before benchmarking).

Each implementation is run in 'mult' mode on the same input files; its output is compared with
the expected result (known answers) and with NumPy's matmul as trusted reference.

Tolerance for random inputs: entries are in [-1, 1), so the standard floating-point error bound of
a length-n dot product gives |error| <= n * u * sum_k |a_ik||b_kj| <= n^2 * u (u = 2^-53).
We use atol = n^2 * eps (eps = 2^-52 = 2u, covering the error of both our loop and the reference),
rtol = 0. Known-answer cases (2x2, identity, zero) are exact in floating point and must match exactly.

Writes results/correctness.csv. Exit code 1 if any test fails.
"""
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IMPLS = {
    "python": ["python3", os.path.join(ROOT, "src/python/matmul.py")],
    "java": ["java", "-cp", os.path.join(ROOT, "src/java/target/benchmarks.jar"), "matmul.MatMul"],
    "c": [os.path.join(ROOT, "build/c/matmul")],
}


def run(lang, a, b, tmp):
    n = a.shape[0]
    pa, pb, pc = (os.path.join(tmp, f"{x}.bin") for x in "ABC")
    a.astype("<f8").tofile(pa)
    b.astype("<f8").tofile(pb)
    subprocess.run(IMPLS[lang] + ["mult", pa, pb, str(n), pc], check=True)
    return np.fromfile(pc, dtype="<f8").reshape(n, n)


def cases():
    rng = np.random.default_rng(7)
    a50 = rng.uniform(-1, 1, (50, 50))
    yield "known_2x2", np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0, 6.0], [7.0, 8.0]]), \
        np.array([[19.0, 22.0], [43.0, 50.0]]), True
    yield "A_times_identity_n50", a50, np.eye(50), a50, True
    yield "identity_times_A_n50", np.eye(50), a50, a50, True
    yield "A_times_zero_n50", a50, np.zeros((50, 50)), np.zeros((50, 50)), True
    for n in (64, 100):                     # a power of two and a non power of two
        a, b = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
        yield f"random_n{n}_vs_numpy", a, b, a @ b, False


def main():
    rows, ok_all = [], True
    with tempfile.TemporaryDirectory() as tmp:
        for name, a, b, expected, exact in cases():
            n = a.shape[0]
            atol = 0.0 if exact else n * n * np.finfo(np.float64).eps
            outputs = {}
            for lang in IMPLS:
                c = run(lang, a, b, tmp)
                outputs[lang] = c
                err = float(np.max(np.abs(c - expected)))
                ok = err <= atol
                ok_all &= ok
                rows.append([name, lang, n, f"{err:.3e}", f"{atol:.3e}", "pass" if ok else "FAIL"])
            same = all(np.array_equal(outputs["python"], outputs[l]) for l in IMPLS)
            rows.append([name, "all_three_bitwise_identical", n, "", "", "yes" if same else "no"])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results/correctness.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "implementation", "n", "max_abs_error", "tolerance", "result"])
        w.writerows(rows)
    for r in rows:
        print(",".join(map(str, r)))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
