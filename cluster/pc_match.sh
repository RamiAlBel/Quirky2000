#!/bin/bash
#SBATCH --partition=titans
#SBATCH --exclude=comp-gpu04,comp-gpu05
#SBATCH --cpus-per-task=16
#SBATCH --mem=12G
#SBATCH --time=03:00:00
# One round (2 games, colours swapped, one opening) at a PC-equivalent clock.
# usage: sbatch -J NAME pc_match.sh ENG_A ENG_B BASE_S INC_S OPENING_IDX OUTDIR
# ENG = E (bundle E: C1 net) or F512|F1024|F4096 (bundle F options = E options + W<width>_sf net).
# The clock is BASE+INC scaled by 10M / (this node's 16-thread W512 nps), so nodes per move match the user's PC
# (about 10M nps with the 512 net on 16 threads).
set -u
X=/scratch/ralbe/chess_nnue/exp2
A=$1; B=$2; BASE=$3; INC=$4; OPEN=$5; OUT=$6
FC=/scratch/ralbe/chess_nnue/tools/fastchess/fastchess
BOOK=/scratch/ralbe/chess_nnue/tools/8moves_v3.pgn
TH=16; HASH=1024; PC_NPS=10000000
T=$OUT/${A}_vs_${B}_o$OPEN; rm -rf $T; mkdir -p $T
for n in $A $B; do
  if [ $n = E ]; then cp $X/bin/W512 $T/$n.bin; cp $X/nets/C_hkb8pair_sfold_s2/C_hkb8pair_sfold_s2.nnue $T/$n.nnue
  else w=${n#F}; cp $X/bin/W$w $T/$n.bin; cp $X/nets/W${w}_sf_s2/W${w}_sf_s2.nnue $T/$n.nnue; fi
done
co=""; for kv in $(sed 's/EvalFile=[^ ]*//' $X/ledger/E_opts.txt); do co+=" option.${kv%%=*}=${kv#*=}"; done
bench() { printf "setoption name EvalFile value $2\nsetoption name Threads value $TH\nsetoption name Hash value 256\nsetoption name UseFinny value 1\nsetoption name UseLazyAcc value 1\nbench 16\nquit\n" | $1 2>&1 | grep -E "^bench:" | sed 's/.* \([0-9]*\) nps.*/\1/'; }
hostname; nps=0
for i in 1 2 3; do v=$(bench $X/bin/W512 $X/nets/W512_sf_s2/W512_sf_s2.nnue); echo "W512 bench $i: $v nps"; [ "${v:-0}" -gt $nps ] && nps=$v; done
for n in $A $B; do echo "$n bench: $(bench $T/$n.bin $T/$n.nnue) nps"; done
read TB TI < <(python3 -c "f=$PC_NPS/$nps; print(round($BASE*f), round($INC*f,2))")
echo "$A vs $B  opening=$OPEN  node W512 nps=$nps  scale=$(python3 -c "print(round($PC_NPS/$nps,3))")  tc=$TB+$TI (PC $BASE+$INC)  threads=$TH hash=$HASH"
$FC -engine cmd=$T/$A.bin name=$A option.EvalFile=$T/$A.nnue -engine cmd=$T/$B.bin name=$B option.EvalFile=$T/$B.nnue \
  -each tc=$TB+$TI option.Threads=$TH option.Hash=$HASH $co \
  -rounds 1 -games 2 -repeat -openings file=$BOOK format=pgn order=sequential start=$OPEN \
  -concurrency 1 -recover -pgnout file=$T/games.pgn nodes=true nps=true -config outname=$T/config.json \
  | grep --line-buffered -vE "^(Warning|Info|Position|Moves);|^Started game"
echo "== done"
