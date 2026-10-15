#!/bin/bash
# Records the execution environment in results/environment.txt.
cd "$(dirname "$0")/.."
mkdir -p results
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "== CPU =="; lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|L1d|L2|L3"
  echo "== memory =="; free -m | head -2
  echo "== OS =="; grep PRETTY_NAME /etc/os-release; uname -r
  echo "== C =="; gcc --version | head -1; echo "flags: -O2 -std=c11 -Wall"
  echo "== Java =="; java -version 2>&1 | grep -v JAVA_TOOL; mvn -v 2>/dev/null | head -1; echo "JMH 1.35, fork JVM flags: -Xmx2g (default JIT and GC)"
  echo "== Python =="; python3 --version; python3 -c "import platform, numpy, matplotlib; print(platform.python_implementation(), 'numpy', numpy.__version__, 'matplotlib', matplotlib.__version__)"
  python3 -c "import pytest, pytest_benchmark; print('pytest', pytest.__version__, 'pytest-benchmark', pytest_benchmark.__version__)"
  echo "== perf =="; P=$(ls /usr/lib/linux-tools/*/perf /usr/lib/linux-tools-*/perf 2>/dev/null | tail -1); ${P:-perf} --version; echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
} > results/environment.txt
cat results/environment.txt
