#!/bin/bash
# Windows PGO build with LLVM clang (Git Bash):   ./build_pgo_win.sh [NET]   -> ../bin/quirky_pgo.exe
#   NET = the .nnue used for the training run (default ../weights/W512_sf.nnue). The search is unchanged by PGO
#   (same bench node count as a plain build); only the code layout is tuned, about +20-25% nodes/s on a Ryzen 5800X.
# Needs LLVM in "C:\Program Files\LLVM\bin" (clang++, lld, llvm-profdata) and Fathom in ../third_party/fathom.
set -e
cd "$(dirname "$0")"
export PATH="/c/Program Files/LLVM/bin:$PATH"
NET=$(cygpath -m "$(realpath "${1:-../weights/W512_sf.nnue}")")
T=$(mktemp -d)
mkdir -p ../bin
F="-O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mpopcnt -mfma -mlzcnt -DNDEBUG -DUSE_PEXT -w -DNNUE_LNN2 -DACC_WIDTH=${ACC:-512} -I../third_party/fathom/src -flto -fuse-ld=lld $EXTRA"
SRC="main.cpp search.cpp nnue.cpp nnue_small.cpp position.cpp book.cpp $T/tb.o"
clang -O3 -mpopcnt -mbmi2 -DNDEBUG -w -I../third_party/fathom/src -c ../third_party/fathom/src/tbprobe.c -o "$T/tb.o"
clang++ $F -fprofile-instr-generate $SRC -o "$T/gen.exe"
# training run: version G options, bench + one 3 s two-thread search
{
  echo "setoption name EvalFile value $NET"
  for o in UseFinny UseLazyAcc UseContHist UseCaptHist UseSingular UseCorr2 UseDblExt UseAspFH UseDeeper; do
    echo "setoption name $o value 1"
  done
  echo "setoption name SingMargin value 3"
  echo "isready"
  echo "setoption name Hash value 512"
  echo "setoption name Threads value 2"
  echo "bench 13"
  echo "position startpos moves e2e4 c7c5 g1f3 d7d6"
  echo "go movetime 3000"
  sleep 4
  echo "quit"
} | LLVM_PROFILE_FILE="$T/p%p.profraw" "$T/gen.exe" > /dev/null
llvm-profdata merge -o "$T/pgo.profdata" "$T"/p*.profraw
clang++ $F -fprofile-instr-use="$T/pgo.profdata" $SRC -o ../bin/quirky_pgo.exe
rm -rf "$T"
echo "built bin/quirky_pgo.exe"
