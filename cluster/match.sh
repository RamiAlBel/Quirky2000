#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# Head-to-head of two sweep nets (each with its own width engine, bundle E options), TC 60+0.5 (1 min + 0.5 s/move).
# usage: sbatch match.sh CHAMP CHALL   (names like W512_sf; nets/NAME_s2/NAME_s2.nnue, bin/W<width>)
# SPRT [-3,+3] may stop it early; the winner is decided from the game results afterwards (ladder.py).
set -u
X=/scratch/ralbe/chess_nnue/exp2
A=$1; B=$2; TC=${TC:-60+0.5}; ROUNDS=${ROUNDS:-400}
FC=/scratch/ralbe/chess_nnue/tools/fastchess/fastchess
BOOK=/scratch/ralbe/chess_nnue/tools/8moves_v3.pgn
T=$X/runs/m_${A}_vs_${B}${MTAG:-}; rm -rf $T; mkdir -p $T
for n in $A $B; do w=${n#W}; w=${w%%_*}; cp $X/bin/W$w $T/$n.bin; cp $X/nets/${n}_s2/${n}_s2.nnue $T/$n.nnue; done
co=""; for kv in $(sed 's/EvalFile=[^ ]*//' $X/ledger/E_opts.txt); do co+=" option.${kv%%=*}=${kv#*=}"; done
echo "$A vs $B tc=$TC rounds<=$ROUNDS"; hostname
$FC -engine cmd=$T/$A.bin name=$A option.EvalFile=$T/$A.nnue -engine cmd=$T/$B.bin name=$B option.EvalFile=$T/$B.nnue \
  -each tc=$TC option.Threads=1 option.Hash=64 $co \
  -rounds $ROUNDS -games 2 -repeat -openings file=$BOOK format=pgn order=random \
  -sprt elo0=-3 elo1=3 alpha=0.05 beta=0.05 -concurrency 11 -recover -ratinginterval 100 \
  -pgnout file=$T/games.pgn -config outname=$T/config.json \
  | grep --line-buffered -vE "^(Warning|Info|Position|Moves);|^Started game|^Finished game"
echo "== done"
