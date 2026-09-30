#!/bin/bash
# Round G: specialised weight sets on the W512_sf recipe (512, pairwise, 20 epochs, sf only, then the small head).
# Each net differs from W512_sf in the listed factor only. usage: ./g_sweep.sh NAME "extra train_v4 flags" [...]
cd /scratch/ralbe/chess_nnue/exp2
while [ $# -ge 2 ]; do
  n=$1; f=$2; shift 2
  STAGE2=" " TLIM=00:45:00 MAXMIN=38 sbatch --export=ALL -J $n -o logs/train_$n.out train.sh $n \
    --h1 16 --h2 32 --nb 1 --hid_act crelu --cp_scale 400 --features hkb --ft_act pair --epochs 20 --acc 512 \
    --data sf:1 $f | awk -v n=$n '{print n, $4}'
done
