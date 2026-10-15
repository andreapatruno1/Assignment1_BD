"""Basic dense matrix multiplication (triple loop, i-j-k order) in pure Python.

Storage: one flat Python list of floats per matrix, row-major (element (i, j) at i*n + j).

Benchmarked with pytest-benchmark (see test_matmul_benchmark.py). Correctness tests use:
  python3 matmul.py mult A.bin B.bin n C.bin
"""
import array
import sys


def multiply(a, b, c, n):
    """C = A * B with the straightforward O(n^3) triple loop."""
    for i in range(n):
        row = i * n
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a[row + k] * b[k * n + j]
            c[row + j] = s


def load(path, n):
    values = array.array("d")
    with open(path, "rb") as f:
        values.fromfile(f, n * n)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()            # plain list of Python floats


def peak_rss_kb():
    """Peak resident set size of this process (VmHWM in /proc/self/status), in kB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return -1


def checksum(c):
    """Plain left-to-right sum (same order in the three languages)."""
    s = 0.0
    for x in c:
        s += x
    return s


def batch_size(n):
    """Very small sizes are timed in batches so that one sample is not too short."""
    return max(1, (1 << 24) // (n * n * n))


def main():
    mode = sys.argv[1]
    if mode == "mult":
        n = int(sys.argv[4])
        a, b = load(sys.argv[2], n), load(sys.argv[3], n)
        c = [0.0] * (n * n)
        multiply(a, b, c, n)
        array.array("d", c).tofile(open(sys.argv[5], "wb"))
        return
    raise SystemExit("usage: matmul.py mult A.bin B.bin n C.bin")

if __name__ == "__main__":
    main()
