#!/bin/bash
# Round G SPRT: candidate vs baseline = bundle E options + W512_sf net, both on bin/G (LNN5 + ExpertRules build).
# usage: ./gsprt.sh NAME NEWNET ["extra new opts"] [TC=6+0.06] [ELO1=5]     NEWNET = nets/<x>/<x>.nnue path or a net name
#   env: OPENINGS (fastchess -openings ... string, see sprt.sh), PART (default titans)
cd /scratch/ralbe/chess_nnue/exp2
X=/scratch/ralbe/chess_nnue/exp2
NAME=$1; NET=$2; EXTRA=${3:-}; TC=${4:-6+0.06}; ELO1=${5:-5}
[ -f "$NET" ] || NET=$X/nets/$NET/$NET.nnue
[ -f "$NET" ] || { echo "no net $NET"; exit 1; }
BASEOPTS="$(sed "s#EvalFile=[^ ]*#EvalFile=$X/nets/W512_sf_s2/W512_sf_s2.nnue#" ledger/E_opts.txt)"
NEWOPTS="$(sed "s#EvalFile=[^ ]*#EvalFile=$NET#" ledger/E_opts.txt) $EXTRA"
sbatch --export=ALL -p ${PART:-titans} --exclude=comp-gpu04,comp-gpu05 --cpus-per-task=24 --mem=48G -J g_$NAME \
  -o logs/sprt_G_$NAME.out sprt.sh G_$NAME "$BASEOPTS" "$NEWOPTS" $TC $ELO1 $X/bin/G $X/bin/G | awk '{print $4}'
