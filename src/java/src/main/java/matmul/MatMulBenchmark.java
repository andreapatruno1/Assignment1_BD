package matmul;

import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.*;

/**
 * JMH benchmark of the triple-loop multiplication (kernel only: inputs are loaded in @Setup).
 * Average time per multiplication; 3 warm-up and 5 measured iterations of 1 s each
 * (an iteration contains at least one multiplication, so large n give one multiplication per iteration).
 * The result array is returned, so JMH consumes it (no dead-code elimination).
 *
 * Run (from src/java): java -jar target/benchmarks.jar MatMulBenchmark -p n=512 -p dataDir=../../data
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgs = {"-Xmx2g"})
@State(Scope.Benchmark)
public class MatMulBenchmark {

    @Param({"64", "128", "256", "512", "1024", "2048"})
    public int n;

    @Param({"data"})
    public String dataDir;

    private double[] a, b, c;

    @Setup(Level.Trial)
    public void setup() throws IOException {
        a = MatMul.load(dataDir + "/A_" + n + ".bin", n);
        b = MatMul.load(dataDir + "/B_" + n + ".bin", n);
        c = new double[n * n];
    }

    @Benchmark
    public double[] multiply() {
        MatMul.multiply(a, b, c, n);
        return c;
    }

    /** Appends "n,checksum,peak_rss_kb" to the file given by -Dmatmul.info (if set). */
    @TearDown(Level.Trial)
    public void report() throws IOException {
        String file = System.getProperty("matmul.info");
        if (file == null) return;
        try (FileWriter w = new FileWriter(file, true)) {
            w.write(n + "," + String.format(java.util.Locale.ROOT, "%.12e", MatMul.checksum(c)) + "," + MatMul.peakRssKb() + "\n");
        }
    }
}
