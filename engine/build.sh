#!/bin/bash
# Linux build (g++, x86-64 with AVX2 + BMI2).   ./build.sh [NAME]   -> ../bin/NAME   (default NAME=quirky)
#   ACC=1024 ./build.sh w1024     net accumulator width baked into the binary (must match the .nnue; default 512)
#   EXTRA=-DFT_INT8 ./build.sh    extra compiler flags
#   ACC=1024 EXTRA=-DNNUE_LNN6 ./build.sh quirky_h   Round H net (weights/L1024T_lc0.nnue: 16 king buckets + threat inputs)
# Syzygy probing needs Fathom in ../third_party/fathom (cloned automatically on first build).
set -e
cd "$(dirname "$0")"
NAME=${1:-quirky}
[ -d ../third_party/fathom/src ] || git clone -q --depth 1 https://github.com/jdart1/Fathom ../third_party/fathom
mkdir -p ../bin
FLAGS="-I../third_party/fathom/src -O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mpopcnt -mfma -mlzcnt -flto -pthread -DNDEBUG -DUSE_PEXT -w -DNNUE_LNN2 -DACC_WIDTH=${ACC:-512} $EXTRA"
gcc -I../third_party/fathom/src -O3 -mpopcnt -mbmi2 -DNDEBUG -w -c ../third_party/fathom/src/tbprobe.c -o /tmp/tbprobe_$$.o
g++ $FLAGS main.cpp search.cpp nnue.cpp nnue_small.cpp position.cpp book.cpp /tmp/tbprobe_$$.o -o ../bin/$NAME
rm -f /tmp/tbprobe_$$.o
echo "built bin/$NAME"
