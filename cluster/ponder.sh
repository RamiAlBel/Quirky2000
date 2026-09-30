#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# usage: sbatch ponder.sh GAMES "A opts" "B opts"   (A ponders; 6+0.06; 5 games at once = <=10 busy threads)
hostname
B=/tmp/ponder_bin_$$; cp /scratch/ralbe/chess_nnue/exp2/bin/dev $B  # private copy: bin/dev gets rebuilt
/scratch/ralbe/miniconda3/envs/cirr_segm/bin/python /scratch/ralbe/chess_nnue/exp2/ponder_match.py $1 5 6 0.06 "$2" "$3" 1 $B
