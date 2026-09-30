#!/bin/bash
# Round G experts: fine-tune the final W512_sf_s2 net (all weights, low LR) on one category of the sfo data.
# usage: ./g_expert.sh NAME SUBSET EPOCHS EPOCH_POS   (SUBSET as training/subsets.py, all: = the control)
cd /scratch/ralbe/chess_nnue/exp2
n=$1; sub=$2; ep=$3; pos=$4
TLIM=00:45:00 MAXMIN=38 sbatch --export=ALL -J $n -o logs/train_$n.out train.sh $n \
  --init W512_sf_s2 --subset $sub --data sfo:1 --features hkb --kb 8 --ft_act pair --acc 512 \
  --h1 8 --h2 0 --nb 8 --hid_act dual --cp_scale 350 --lr 1e-4 --lr_end 3e-6 --epochs $ep --epoch_pos $pos | awk -v n=$n '{print n, $4}'
