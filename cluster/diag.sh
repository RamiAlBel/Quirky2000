#!/bin/bash
#SBATCH --partition=titans
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp-gpu08
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
# usage: sbatch -o logs/diag_TAG.out diag.sh NET...   (after optional --dependency) -> per-phase val MSE
cd /scratch/ralbe/chess_nnue/exp2/training
/scratch/ralbe/miniconda3/envs/cirr_segm/bin/python phase_diag.py "$@"
echo "== done diag"
