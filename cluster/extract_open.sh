#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=46
#SBATCH --mem=96G
#SBATCH --time=06:00:00
# July 2026 eval games -> data/open/sfo.bin + sfo.meta (opening/game sidecar, training/extract_open.py)
X=/scratch/ralbe/chess_nnue/exp2
hostname
zstd -dc $X/data/lichess_2026-07.pgn.zst | /scratch/ralbe/miniconda3/envs/cirr_segm/bin/python $X/training/extract_open.py $X/data/open 44
