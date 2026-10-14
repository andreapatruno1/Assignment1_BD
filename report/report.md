# Assignment 1 — How Matrix Size Affects a Basic Matrix Multiplication in Python, Java and C

<p class="meta">Big Data · Grado en Ciencia e Ingeniería de Datos · ULPGC 2026-2027 · Individual Assignment 1 · Andrea Patruno · October 2026</p>

**Abstract.** We compare the same triple-loop matrix multiplication (i-j-k order, float64, flat row-major arrays) in Python, Java and C, on the same shared inputs and the same machine, for n = 64 … 2048. For each configuration we take 5 timed repetitions of the kernel only and report median and interquartile range, together with the peak resident memory of the process. All three implementations produce bitwise identical results. C and Java finish n = 1024 (8.5 s and 4.0 s), pure Python reaches only n = 512 (14.8 s) inside the 20 s budget. Runtime grows about as n³ only while the matrices fit in cache: from n = 256 to 512 the time grows ×34 (C) instead of ×8 because the column-wise access to B causes cache misses. Python is 50–140× slower than C; Java is 2.4× slower than C at n = 64, close at n = 256–512, and 2.2× faster at n = 1024. The conclusions hold only for these implementations, settings and this machine.

## 1. Introduction and objective

Matrix multiplication is a basic Big Data kernel (graphs, machine learning, scientific computing) and its cost grows quickly with size. Before optimising it (later assignments) we need a controlled baseline.

**Research question.** How does the matrix size n affect the runtime and the memory use of the same basic triple-loop algorithm implemented in Python, Java and C?

- *Factor changed:* matrix size n (and the language).
- *Outcomes measured:* time per multiplication (median and IQR) and peak resident memory.

**Hypotheses (stated before running).**

- H1: the time grows as n³ (×8 when n doubles) for all three languages.
- H2: C is the fastest, Java is close after JIT warm-up, Python is one to two orders of magnitude slower because it is interpreted.
- H3: C uses about 24n² bytes for the three matrices plus a small runtime; Java adds the JVM (tens of MB); Python needs more memory per element because each float is an object.

## 2. Background

**Algorithm.** For C = A·B with n × n matrices, the straightforward method computes every C[i][j] as the dot product of row i of A and column j of B: three nested loops, n³ multiply-add operations, O(n³) time and O(n²) memory [1].

**Memory hierarchy.** With row-major storage, the inner loop reads A[i][k] contiguously but B[k][j] with a stride of n·8 bytes, so each access to B can need a new cache line. When B no longer fits in the cache the cost per operation increases; with power-of-two strides many lines also map to the same cache sets (conflict misses) [2].

**Languages.** C is compiled ahead of time to machine code. Java is compiled to bytecode and then compiled at run time by the JIT, so the first runs are slower (warm-up). CPython interprets bytecode and each float is a separate heap object, so every operation in the inner loop costs much more than one machine instruction.

**Floating-point error.** A length-n dot product computed in floating point has error at most γₙ·Σ|aₖ||bₖ| with γₙ ≈ n·u, u = 2⁻⁵³ [3]. This is used to set the correctness tolerance.

## 3. Methodology

### 3.1 Implementations

The three programs (`src/`) contain the same loop, in the same order, on the same layout:

```
for i in 0..n-1:  for j in 0..n-1:  s = 0;  for k in 0..n-1:  s += A[i*n+k] * B[k*n+j];   C[i*n+j] = s
```

| | Python | Java | C |
|---|---|---|---|
| Storage | flat `list` of Python floats | flat `double[]` | flat `double*` (malloc) |
| Numeric type | float64 (boxed objects) | double (primitive) | double |
| Execution | CPython 3.14.4 interpreter | OpenJDK 21.0.12, JIT (C2), `-Xmx2g` | GCC 15.2.0 `-O2 -std=c11` |
| Timer | `time.perf_counter_ns` [6] | `System.nanoTime` [7] | `clock_gettime(CLOCK_MONOTONIC)` |

No library call replaces the loops. Unavoidable differences: Python stores pointers to float objects (≈ 32 bytes per element instead of 8) and checks types at every operation; Java checks array bounds; C uses `-O2` without `-ffast-math`, so the compiler cannot reorder the floating-point sum [8] and all three perform exactly the same additions in the same order.

### 3.2 Inputs

`scripts/gen_inputs.py` writes A and B once for every n as binary files (float64, little-endian, row-major), and the three programs read the same files. Generator: NumPy PCG64 [4], seed 42 + n, values uniform in [−1, 1); A is drawn first, then B.

### 3.3 Environment

| Item | Value |
|---|---|
| Machine | MSI laptop, Intel Core i7-1255U (10 cores: 2 performance + 8 efficiency, 12 threads), Windows host |
| Execution environment | WSL2, Ubuntu 26.04.1 (kernel 6.6.87.2-microsoft-standard-WSL2), 6 cores / 12 logical processors visible, 10.7 GiB RAM |
| Caches (per core, as reported in WSL2) | L1d 48 KiB, L2 1.25 MiB, L3 12 MiB (shared) |
| Threads | all three programs are single-threaded; no numerical library is used in the timed code |

### 3.4 Correctness (before benchmarking)

`scripts/verify.py` runs the three programs on: a 2×2 case with known result, A·I = A and I·A = A (n = 50), A·0 = 0 (n = 50), and random matrices with n = 64 and n = 100 compared with NumPy `matmul` (reference only). Known-answer cases must match exactly. For random inputs |aᵢₖ|, |bₖⱼ| ≤ 1, so the error bound of §2 gives at most n²u per entry; we use absolute tolerance n²·eps (eps = 2⁻⁵², which also covers the reference's own error) and no relative tolerance.

### 3.5 Measurement protocol

- **Timing boundary:** kernel only. Loading the files and allocating A, B, C are outside the timer; the output array is allocated once and overwritten.
- **Repetitions:** 5 measured repetitions per configuration; warm-up runs excluded: Java 3 (JIT), C 1 and Python 1 (caches).
- **Short runs:** for n = 64 and 128 one sample times a batch of 64 and 8 multiplications, and reports the time per multiplication.
- **Result consumed:** after each sample the program sums all entries of C (checksum); the checksum is saved with every row.
- **Summary:** median, Q1 and Q3 (inclusive method), IQR = Q3 − Q1. All raw values are in `results/raw_times.csv`.
- **Memory:** peak resident set size (VmHWM in `/proc/self/status` [5]) of the process, read by the program itself after its last run. It includes the runtime (interpreter, JVM), so it is reported next to an estimate of the array storage: 3·8n² bytes for C/Java, 3·32n² for Python.
- **Sizes and budgets** (declared before running): n = 64, 128, 256, 512, 1024, 2048. Budget: 20 s for one multiplication, 2 GB peak memory. When 8 × the previous median exceeds 20 s, one multiplication is first attempted with a 20 s timeout; if it times out, larger sizes are not attempted.

## 4. Results

### 4.1 Correctness

All tests passed for the three implementations. The known-answer, identity and zero cases match exactly (error 0). Against NumPy the maximum absolute error is 0 for both n = 64 (tolerance 9.1·10⁻¹³) and n = 100 (tolerance 2.2·10⁻¹²): on this machine the reference produced the same values. The outputs of Python, Java and C are **bitwise identical** in every test, and the benchmark checksums are identical across languages for every n.

### 4.2 Runtime

Table 1 — Time per multiplication (seconds), median of 5 repetitions with IQR. "—" = not attempted after a timeout at a smaller size.

| n | C median | C IQR | Java median | Java IQR | Python median | Python IQR | Java / C | Python / C |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.0000704 | 0.0000037 | 0.000166 | 0.000010 | 0.00987 | 0.00049 | 2.4 | 140 |
| 128 | 0.00102 | 0.00003 | 0.00159 | 0.00004 | 0.0805 | 0.0008 | 1.6 | 79 |
| 256 | 0.00890 | 0.00011 | 0.0111 | 0.0009 | 0.821 | 0.143 | 1.3 | 92 |
| 512 | 0.298 | 0.013 | 0.336 | 0.037 | 14.81 | 0.56 | 1.1 | 50 |
| 1024 | 8.54 | 0.32 | 3.96 | 0.06 | timeout | | 0.46 | — |
| 2048 | timeout | | timeout | | — | | | |

<figure><img src="time.png"><figcaption>Figure 1 — Median time per multiplication vs n (log-log, error bars = IQR). The dotted line is n³ scaled from C at n = 64.</figcaption></figure>

Table 2 — Growth factor of the median when n doubles (n³ model: ×8).

| n → 2n | 64→128 | 128→256 | 256→512 | 512→1024 |
|---|---|---|---|---|
| C | 14.4 | 8.7 | 33.5 | 28.7 |
| Java | 9.6 | 7.0 | 30.2 | 11.8 |
| Python | 8.1 | 10.2 | 18.0 | — |

<figure><img src="per_iter.png"><figcaption>Figure 2 — Median time divided by n³: cost of one inner-loop iteration (one multiply-add) in nanoseconds.</figcaption></figure>

Variability is small: the IQR is below 6 % of the median in most configurations; it is larger for Java at n = 256 and 512 (8 %, 11 %) and Python at n = 256 (17 %).

### 4.3 Memory and limits

Table 3 — Peak resident memory of the process (MiB) and estimated storage of A, B and C (MiB).

| n | C peak RSS | Java peak RSS | Python peak RSS | arrays C/Java (3·8n²) | arrays Python (≈3·32n²) |
|---|---|---|---|---|---|
| 64 | 1.9 | 44.1 | 11.1 | 0.09 | 0.4 |
| 128 | 2.1 | 43.4 | 12.4 | 0.4 | 1.5 |
| 256 | 3.3 | 45.5 | 18.2 | 1.5 | 6 |
| 512 | 7.8 | 54.8 | 40.9 | 6 | 24 |
| 1024 | 25.8 | 92.5 | — | 24 | — |

<figure><img src="memory.png"><figcaption>Figure 3 — Peak resident memory vs n (log-log).</figcaption></figure>

**Limits.** Largest size completed inside the budget: n = 1024 for C and Java, n = 512 for Python. Timeouts (one multiplication not finished in 20 s, `results/limits.csv`): C at n = 2048, Java at n = 2048, Python at n = 1024. The memory budget (2 GB) was never reached; the largest attempted size, 2048, needs 96 MiB of arrays in C/Java, so the limit is time, not memory.

## 5. Discussion

**H1 — cubic growth: only partly confirmed.** The growth is close to ×8 only while the matrices are small: Python ×8.1 (64→128), C ×8.7 and Java ×7.0 (128→256). From 256 to 512 the time grows ×34 (C), ×30 (Java) and ×18 (Python), and from 512 to 1024 ×29 (C) and ×12 (Java). Figure 2 shows why: the cost of one multiply-add is constant only while the data stays in cache, then it rises (C: 0.27–0.53 ns for n ≤ 256, 2.2 ns at n = 512, 8.0 ns at n = 1024). At n = 512 one matrix is 2 MiB, larger than the L2 cache (1.25 MiB), and the inner loop jumps 4 KiB in B at every step: almost every access to B is a cache miss, and the power-of-two stride makes the lines compete for the same cache sets [2]. The first jump in C (×14.4 from 64 to 128) has the same cause on a smaller scale: at n = 64, B (32 KiB) fits in the 48 KiB L1, at n = 128 (128 KiB) it does not. The algorithm is O(n³), but the observed constant depends on the memory hierarchy. We did not measure cache misses with hardware counters, so this explanation is supported by the timing pattern and the cache sizes, not by direct cache measurements.

**H2 — language ranking: confirmed only up to n = 512.** For n ≤ 512 C is fastest, Java is 1.1–2.4× slower and Python 50–140× slower. In Python every inner iteration executes several bytecode instructions, two list lookups and the creation of a new float object (≈ 38–49 ns per iteration for n ≤ 256 against < 1 ns in C). The gap between C and Java shrinks as n grows, because the loop becomes limited by memory accesses rather than by the generated instructions; the extra cost of Java at n = 64 is plausibly due to array bounds checks and to short kernels where fixed overheads are more visible. At n = 1024 Java is 2.2× faster than C (3.96 s vs 8.54 s, IQRs not overlapping). This result was not predicted and our data cannot explain it with certainty. Two plausible causes are: (a) the JIT generates a different inner loop than GCC `-O2` (for example with more unrolling, so more memory accesses are in flight when almost every access to B misses the cache); (b) the i7-1255U has performance and efficiency cores and no CPU affinity was set, so the C process may have run on a slower core. Separating them would require hardware counters and pinning the process to one core.

**H3 — memory: confirmed.** C's peak memory is the arrays plus ≈ 1.8 MiB of process (25.8 MiB at n = 1024 for 24 MiB of arrays). Java has a fixed JVM cost of ≈ 43 MiB, and its peak also includes the temporary byte buffers used to read the files (92.5 MiB at n = 1024). Python starts at ≈ 11 MiB and grows about 4–5× faster per element than C, because each value is a float object (24 bytes) referenced by a list slot (8 bytes); at n = 512 it uses 40.9 MiB against 7.8 MiB for C.

**Limitations.**

- One machine, under WSL2 on a Windows host: background activity can add noise (largest IQR: Python 17 % at n = 256). The results do not describe other machines.
- Hybrid CPU (performance and efficiency cores) and no CPU affinity: different runs may use different core types.
- Peak RSS includes runtime and file-loading buffers; it is not the same quantity as array storage, so the two are reported separately.
- Only 5 repetitions per configuration; the timeouts give a time limit, not a measured time.
- The batching for n ≤ 128 means that those samples reuse warm caches.
- The cache explanation is not verified with hardware counters (e.g. `perf`).

## 6. Conclusions

On this machine, with the same algorithm, inputs and arithmetic, the triple-loop multiplication grows as n³ only while the matrices fit in cache; beyond that, memory access dominates and the growth per doubling is much larger than 8. C and Java reach n = 1024 (8.5 s and 4.0 s); C is fastest up to n = 512, while at n = 1024 Java is faster for reasons we could not isolate. Pure Python is 50–140× slower than C and reaches only n = 512 in the 20 s budget. Memory follows the 8n² (C, Java) or ≈ 32n² (Python) storage plus a fixed runtime cost. These results are limited to these implementations, compiler and runtime settings, and hardware; they do not show that one language is faster in general. The loop order that reads B by columns is the main bottleneck, which is the starting point for the optimisations of Assignment 2.

## 7. References and materials

[1] T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, *Introduction to Algorithms*, 4th ed. Cambridge, MA, USA: MIT Press, 2022.

[2] U. Drepper, "What every programmer should know about memory," Red Hat, Inc., Nov. 2007. [Online]. Available: https://people.freebsd.org/~lstewart/articles/cpumemory.pdf

[3] N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. Philadelphia, PA, USA: SIAM, 2002.

[4] C. R. Harris *et al.*, "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.

[5] Linux man-pages project, "proc(5) — Linux manual page." [Online]. Available: https://man7.org/linux/man-pages/man5/proc.5.html

[6] Python Software Foundation, "time — Time access and conversions," Python 3.14 documentation. [Online]. Available: https://docs.python.org/3.14/library/time.html

[7] Oracle, "System (Java SE 21 & JDK 21)," API documentation. [Online]. Available: https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/System.html

[8] Free Software Foundation, "Options that control optimization," GCC online documentation. [Online]. Available: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html

**Materials.** GitHub repository: https://github.com/andreapatruno1/Assignment1_BD. It contains the source code, the README with exact build and run commands, the input generator, raw measurements (`results/raw_times.csv`), correctness results, environment description and the scripts that produce all tables and figures (`scripts/analyze.py`).

