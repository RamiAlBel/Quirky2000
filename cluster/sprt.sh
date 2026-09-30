#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# SPRT of a candidate against the baseline (t25p + current accepted options).
# usage: sbatch -J NAME sprt.sh NAME "BASE opts" "NEW opts" [TC=6+0.06] [ELO1=8] [BASE_BIN=bin/base] [NEW_BIN=bin/dev]
#   opts = space-separated Name=Value UCI options; EvalFile defaults to nets/t25p.nnue
set -u
X=/scratch/ralbe/chess_nnue/exp2
NAME=$1; BOPTS=$2; NOPTS=$3; TC=${4:-6+0.06}; ELO1=${5:-8}
BB=${6:-$X/bin/base}; NB=${7:-$X/bin/dev}
FC=/scratch/ralbe/chess_nnue/tools/fastchess/fastchess
BOOK=/scratch/ralbe/chess_nnue/tools/8moves_v3.pgn
OPEN=${OPENINGS-"-openings file=$BOOK format=pgn order=random"}  # OPENINGS="" -> every game from the start position
EACH=$([[ $TC == *=* ]] && echo "$TC" || echo "tc=$TC")  # "nodes=N" -> fixed-node games
T=$X/runs/${NAME}_${SLURM_JOB_ID:-local}
mkdir -p $T; cp $BB $T/base_bin; cp $NB $T/new_bin
opts() { local o="option.EvalFile=$X/nets/t25p.nnue"; for kv in $1; do o+=" option.${kv%%=*}=${kv#*=}"; done; echo $o; }
echo "$NAME  base[$BOPTS] new[$NOPTS] tc=$TC elo1=$ELO1"; hostname
$FC -engine cmd=$T/new_bin name=new $(opts "$NOPTS") -engine cmd=$T/base_bin name=base $(opts "$BOPTS") \
  -each $EACH option.Threads=${THREADS:-1} option.Hash=${HASH:-32} \
  -rounds 10000 -games 2 -repeat $OPEN \
  -sprt elo0=0 elo1=$ELO1 alpha=0.05 beta=0.05 -concurrency ${CONC:-11} -recover -ratinginterval 200 \
  -pgnout file=$T/games.pgn -config outname=$T/config.json \
  | grep --line-buffered -vE "^(Warning|Info|Position|Moves);|^Started game|^Finished game"
echo "== done"
