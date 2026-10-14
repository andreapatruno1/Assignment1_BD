#!/bin/bash
# Builds the Java and C implementations into build/.
set -e
cd "$(dirname "$0")/.."
mkdir -p build/java build/c
javac --release 11 -d build/java src/java/MatMul.java
gcc -O2 -std=c11 -Wall -o build/c/matmul src/c/matmul.c
echo "build OK"
