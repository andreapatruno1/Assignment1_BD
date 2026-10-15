package matmul;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Basic dense matrix multiplication (triple loop, i-j-k order) in Java.
 * Storage: one flat double[] per matrix, row-major (element (i, j) at i*n + j).
 *
 * Benchmarked with JMH (see MatMulBenchmark). Correctness tests use:
 *   java -cp target/benchmarks.jar matmul.MatMul mult A.bin B.bin n C.bin
 */
public class MatMul {

    /** C = A * B with the straightforward O(n^3) triple loop. */
    static void multiply(double[] a, double[] b, double[] c, int n) {
        for (int i = 0; i < n; i++) {
            int row = i * n;
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) s += a[row + k] * b[k * n + j];
                c[row + j] = s;
            }
        }
    }

    static double[] load(String path, int n) throws IOException {
        ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(Paths.get(path))).order(ByteOrder.LITTLE_ENDIAN);
        double[] m = new double[n * n];
        buf.asDoubleBuffer().get(m);
        return m;
    }

    static void save(String path, double[] m) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(m.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        buf.asDoubleBuffer().put(m);
        Files.write(Paths.get(path), buf.array());
    }

    /** Peak resident set size of this process (VmHWM in /proc/self/status), in kB. */
    static long peakRssKb() throws IOException {
        try (BufferedReader r = new BufferedReader(new FileReader("/proc/self/status"))) {
            String line;
            while ((line = r.readLine()) != null)
                if (line.startsWith("VmHWM:")) return Long.parseLong(line.replaceAll("[^0-9]", ""));
        }
        return -1;
    }

    /** Plain left-to-right sum (same order in the three languages). */
    static double checksum(double[] c) {
        double s = 0.0;
        for (double x : c) s += x;
        return s;
    }

    public static void main(String[] args) throws IOException {
        if (args[0].equals("mult")) {
            int n = Integer.parseInt(args[3]);
            double[] a = load(args[1], n), b = load(args[2], n), c = new double[n * n];
            multiply(a, b, c, n);
            save(args[4], c);
            return;
        }
        throw new IllegalArgumentException("usage: MatMul mult A.bin B.bin n C.bin");
    }
}
