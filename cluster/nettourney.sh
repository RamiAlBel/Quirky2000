#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# Round robin of nets: sbatch -J NAME nettourney.sh NAME TC ROUNDS "common opts" name=BIN:NET ...
#   BIN relative to exp2/bin, NET absolute or relative to exp2; opts = Name=Value UCI options for every player
set -u
X=/scratch/ralbe/chess_nnue/exp2
NAME=$1; TC=$2; ROUNDS=$3; COMMON=$4; shift 4
FC=/scratch/ralbe/chess_nnue/tools/fastchess/fastchess
BOOK=/scratch/ralbe/chess_nnue/tools/8moves_v3.pgn
T=$X/runs/${NAME}_${SLURM_JOB_ID:-local}
mkdir -p $T
co=""; for kv in $COMMON; do co+=" option.${kv%%=*}=${kv#*=}"; done
ENG=()
for spec in "$@"; do
  n=${spec%%=*}; rest=${spec#*=}; b=${rest%%:*}; net=${rest#*:}
  [[ $net = /* ]] || net=$X/$net
  cp $X/bin/$b $T/$n.bin; cp $net $T/$n.nnue
  printf "%-12s " $n; printf "setoption name EvalFile value $T/$n.nnue\n$(for kv in $COMMON; do printf "setoption name ${kv%%=*} value ${kv#*=}\\\\n"; done)bench 13\nquit\n" | $T/$n.bin 2>&1 | grep -E "^bench:|nnue:"
  ENG+=(-engine cmd=$T/$n.bin name=$n option.EvalFile=$T/$n.nnue)
done
$FC "${ENG[@]}" -each tc=$TC option.Threads=1 option.Hash=32 $co \
  -tournament roundrobin -rounds $ROUNDS -games 2 -repeat -openings file=$BOOK format=pgn order=random \
  -concurrency 11 -recover -ratinginterval 300 -report penta=true -pgnout file=$T/games.pgn \
  | grep --line-buffered -vE "^(Warning|Info|Position|Moves);|^Started game|^Finished game"
echo "== done"
