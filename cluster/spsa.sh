#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# usage: sbatch -J NAME -o logs/spsa_NAME.out spsa.sh CONFIG.json STATE.json N PAIRS TC "fixed opts"
#   runs with a private binary copy (so rebuilding bin/dev mid-run is safe); resumable via STATE.json
X=/scratch/ralbe/chess_nnue/exp2
T=$X/runs/spsa_${SLURM_JOB_ID:-local}; mkdir -p $T; cp $X/bin/dev $T/engine
hostname
/scratch/ralbe/miniconda3/envs/cirr_segm/bin/python -u $X/spsa.py "$1" "$2" "$3" "$4" 11 "$5" "$6" $T/engine
echo "== done spsa"
