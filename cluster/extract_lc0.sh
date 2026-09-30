#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
# Leela test80 binpack -> REC4 (data/lc0/lc0.bin). cp = lc0 score x 0.39 (C1-calibrated onto the Stockfish cp scale,
# training/calib_teacher.py: best fit 0.31 on lc0 vs 0.80 on sf.bin), result = Leela self-play game result.
F=${F:-test80-2024-06-jun-2tb7p.min-v2.v6.binpack.zst}
D=/scratch/ralbe/chess_nnue/exp2/data/lc0
cd $D && hostname && date
[ -f month.binpack ] || zstd -d -T4 -o month.binpack $F
/scratch/ralbe/chess_nnue/exp2/lc0conv/binpack2rec4 month.binpack lc0.bin.tmp 0.39 1.0 2000000000 > lc0_stats.json && mv lc0.bin.tmp lc0.bin
cat lc0_stats.json; ls -la lc0.bin && rm month.binpack && echo EXTRACT_OK; date
