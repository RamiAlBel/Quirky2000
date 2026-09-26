#!/bin/bash
# Linux build (g++). ./build.sh             -> ../nnue_engine          (LNN1, 256, the original lite.nnue)
#                    ./build.sh lnn2 1024   -> ../nnue_engine_lnn2_1024 (LNN2 nets of width 1024)
set -e
cd "$(dirname "$0")"
FLAGS="-O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mpopcnt -mfma -mlzcnt -flto -pthread -DNDEBUG -DUSE_PEXT -w"
OUT=../nnue_engine
if [ "$1" = "lnn2" ]; then FLAGS="$FLAGS -DNNUE_LNN2 -DACC_WIDTH=$2"; OUT=../nnue_engine_lnn2_$2; fi
g++ $FLAGS main.cpp search.cpp nnue.cpp position.cpp -o $OUT
echo "Built $OUT"
