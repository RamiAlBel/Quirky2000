#!/bin/bash
# wave 2: 200 superbatches x 100M positions (Jan-Jun 2024 test80), one GPU each.
# w2_thr / w2_nothr = the full-length threats on/off pair (needed whatever the pilot SPRT says).
B=/scratch/ralbe/chess_nnue/exp2/bullet
D=$B/../data/lc0/bp
export QK_DATA=$D/01-jan.binpack,$D/02-feb.binpack,$D/03-mar.binpack,$D/04-apr.binpack,$D/05-may.binpack,$D/06-jun.binpack
export QK_SB=200 QK_SAVE=20
sub() { local n=$1; shift; env "$@" QK_NAME=$n sbatch -J $n -o $B/logs/$n.out $B/h_train.sh; }
[ $# -eq 0 ] && set -- w2_thr w2_nothr
for n in "$@"; do
  case $n in
    w2_thr) sub w2_thr ;;
    w2_nothr) sub w2_nothr QK_THREATS=0 ;;
    *) echo "unknown run $n" ;;
  esac
done
