#!/bin/bash
# 3 widths x 3 label sources, C1 recipe (hkb8 + pairwise, 20 epochs, stage 2 = small head), one source per net.
#   sf  = Lichess games with Stockfish %eval (data/jul/sf.bin), edb = Lichess eval DB (deep SF, data/evaldb/edb.bin),
#   lc0 = Leela test80 Jun 2024 self-play, lc0 evals (data/lc0/lc0.bin; waits for the extraction job $1)
# usage: ./width_sweep.sh EXTRACT_JOBID
cd /scratch/ralbe/chess_nnue/exp2
for src in sf edb lc0; do
  dep=""; [ $src = lc0 ] && dep="--dependency=afterok:$1"
  for w in 512 1024 4096; do
    n=W${w}_$src
    if [ $w = 4096 ]; then export TLIM=03:00:00 MAXMIN=170; else export TLIM=00:45:00 MAXMIN=38; fi
    STAGE2=" " sbatch --export=ALL --time=$TLIM $dep -J $n -o logs/train_$n.out train.sh $n \
      --h1 16 --h2 32 --nb 1 --hid_act crelu --cp_scale 400 \
      --features hkb --kb 8 --ft_act pair --epochs 20 --acc $w --data $src:1 | awk -v n=$n '{print n, $4}'
  done
done
