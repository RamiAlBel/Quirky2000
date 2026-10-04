#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
# Round H: download Leela test80 2024 Jan-Jun binpacks (v6) and decompress to data/lc0/bp/*.binpack for bullet.
D=/scratch/ralbe/chess_nnue/exp2/data/lc0
cd $D && hostname && date
for M in 01-jan 02-feb 03-mar 04-apr 05-may 06-jun; do
  F=test80-2024-$M-2tb7p.min-v2.v6.binpack.zst
  [ -f bp/$M.binpack ] && continue
  [ -f $F ] || curl -sSL -C - --retry 20 --retry-delay 30 -o $F.part "https://huggingface.co/datasets/linrock/test80-2024/resolve/main/$F" && mv $F.part $F
  zstd -d -T2 -o bp/$M.binpack.tmp $F && mv bp/$M.binpack.tmp bp/$M.binpack && echo "OK $M $(stat -c %s bp/$M.binpack)"
  [ $M != 06-jun ] && rm -f $F   # keep the original Jun .zst (used by older scripts)
  date
done
echo DL_DONE
