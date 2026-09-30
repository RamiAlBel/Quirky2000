#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=46
#SBATCH --mem=120G
#SBATCH --time=04:00:00
X=/scratch/ralbe/chess_nnue/exp2
zstd -dc $X/data/lichess_2026-07.pgn.zst | /scratch/ralbe/miniconda3/envs/cirr_segm/bin/python $X/training/opp_tree.py $X/books/opp2000.npz 30
