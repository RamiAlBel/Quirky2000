#!/bin/bash
#SBATCH --partition=titans
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp-gpu08
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
X=/scratch/ralbe/chess_nnue/exp2
hostname
zstd -dc $X/data/lichess_2026-07.pgn.zst | /scratch/ralbe/miniconda3/envs/cirr_segm/bin/python $X/training/extract_v4.py $X/data/jul 30
