#!/bin/bash
#SBATCH --partition=titans
#SBATCH --exclude=comp-gpu[04-05,11]
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
cd /scratch/ralbe/chess_nnue/exp2/training; hostname
OMP_NUM_THREADS=16 /scratch/ralbe/miniconda3/envs/cirr_segm/bin/python buckets.py "$@" 2>&1 | grep -v Warning
