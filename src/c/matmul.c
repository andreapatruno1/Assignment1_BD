/*
 * Basic dense matrix multiplication (triple loop, i-j-k order) in C.
 * Storage: one flat double array per matrix (malloc), row-major (element (i, j) at i*n + j).
 *
 * Build: gcc -O2 -std=c11 -Wall -o matmul matmul.c   (benchmarked under perf stat, see scripts/run_benchmarks.py)
 * Usage:
 *   ./matmul mult  A.bin B.bin n C.bin       multiply once and write C (correctness tests)
 *   ./matmul bench n warmup reps datadir     benchmark; prints one CSV row per measured repetition
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* C = A * B with the straightforward O(n^3) triple loop. */
static void multiply(const double *a, const double *b, double *c, int n) {
    for (int i = 0; i < n; i++) {
        int row = i * n;
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += a[row + k] * b[k * n + j];
            c[row + j] = s;
        }
    }
}

static double *load(const char *path, int n) {
    double *m = malloc((size_t) n * n * sizeof(double));
    FILE *f = fopen(path, "rb");
    if (!m || !f || fread(m, sizeof(double), (size_t) n * n, f) != (size_t) n * n) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    fclose(f);
    return m;   /* files are little-endian float64; x86-64 is little-endian */
}

/* Peak resident set size of this process (VmHWM in /proc/self/status), in kB. */
static long peak_rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long kb = -1;
    while (f && fgets(line, sizeof line, f))
        if (strncmp(line, "VmHWM:", 6) == 0) { kb = strtol(line + 6, NULL, 10); break; }
    if (f) fclose(f);
    return kb;
}

/* Plain left-to-right sum (same order in the three languages). */
static double checksum(const double *c, int n) {
    double s = 0.0;
    for (long i = 0; i < (long) n * n; i++) s += c[i];
    return s;
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Very small sizes are timed in batches so that one sample is not too short. */
static int batch_size(int n) {
    long b = (1L << 24) / ((long) n * n * n);
    return b < 1 ? 1 : (int) b;
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    if (strcmp(argv[1], "mult") == 0) {
        int n = atoi(argv[4]);
        double *a = load(argv[2], n), *b = load(argv[3], n), *c = calloc((size_t) n * n, sizeof(double));
        multiply(a, b, c, n);
        FILE *f = fopen(argv[5], "wb");
        fwrite(c, sizeof(double), (size_t) n * n, f);
        fclose(f);
        return 0;
    }
    int n = atoi(argv[2]), warmup = atoi(argv[3]), reps = atoi(argv[4]);
    char pa[512], pb[512];
    snprintf(pa, sizeof pa, "%s/A_%d.bin", argv[5], n);
    snprintf(pb, sizeof pb, "%s/B_%d.bin", argv[5], n);
    double *a = load(pa, n), *b = load(pb, n);                       /* outside the timed region */
    double *c = calloc((size_t) n * n, sizeof(double));             /* output allocated outside too */
    int batch = batch_size(n);
    double *time = malloc(reps * sizeof(double)), *chk = malloc(reps * sizeof(double));
    for (int r = -warmup; r < reps; r++) {
        double t0 = now_s();
        for (int i = 0; i < batch; i++) multiply(a, b, c, n);
        double t1 = now_s();
        if (r >= 0) { time[r] = (t1 - t0) / batch; chk[r] = checksum(c, n); }   /* consumes the result */
    }
    long rss = peak_rss_kb();
    for (int r = 0; r < reps; r++)
        printf("c,%d,%d,%d,%.9f,%.12e,%ld\n", n, r, batch, time[r], chk[r], rss);
    return 0;
}
