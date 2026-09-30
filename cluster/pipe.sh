#!/bin/bash
# t25p recipe for a net variant: stage 1 = full training with acc512's head (16->32, crelu, scale 400),
# stage 2 = t25p head (8->out, dual, 8 buckets, scale 350) fitted on the frozen FT + 1 polish epoch.
# usage: ./pipe.sh NAME [train_v4 flags...]   -> nets/NAME (stage 1), nets/NAME_s2 (final)
NAME=$1; shift
cd /scratch/ralbe/chess_nnue/exp2
STAGE2=" " sbatch --export=ALL -J $NAME -o logs/train_$NAME.out train.sh $NAME --h1 16 --h2 32 --nb 1 --hid_act crelu --cp_scale 400 "$@" | awk '{print $4}'
