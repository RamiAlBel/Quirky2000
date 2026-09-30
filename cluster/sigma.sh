#!/bin/bash
#SBATCH --partition=titans
#SBATCH --exclude=comp-gpu[04-05,11]
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# sigma head on CPU (no GPU needed): sbatch -J sig_X -o logs/sigma_X.out sigma.sh NET OUT [sigma.py flags]
cd /scratch/ralbe/chess_nnue/exp2/training; hostname
OMP_NUM_THREADS=16 /scratch/ralbe/miniconda3/envs/cirr_segm/bin/python sigma.py "$@" 2>&1 | grep -v Warning
