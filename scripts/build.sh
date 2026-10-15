#!/bin/bash
# Builds the Java JMH benchmark (src/java/target/benchmarks.jar) and the C program (build/c/matmul).
set -e
cd "$(dirname "$0")/.."
mvn -q -B -f src/java/pom.xml package
mkdir -p build/c
gcc -O2 -std=c11 -Wall -o build/c/matmul src/c/matmul.c
echo "build OK"
