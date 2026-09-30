#!/bin/bash
#SBATCH --partition=titans
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp-gpu08
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
# usage: sbatch stage2.sh SRC DST [stage2 flags]
cd /scratch/ralbe/chess_nnue/exp2/training
/scratch/ralbe/miniconda3/envs/cirr_segm/bin/python stage2.py "$@" 2>&1 | grep -v Warning
