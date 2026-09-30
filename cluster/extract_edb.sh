#!/bin/bash
#SBATCH --partition=titans
#SBATCH --exclude=comp-gpu[01-05,07-08,10-12]
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=08:00:00
# Lichess eval DB -> data/evaldb/edb.{bin,move,gap,depth}; waits for the background download to finish
X=/scratch/ralbe/chess_nnue/exp2; F=$X/data/evaldb/lichess_db_eval.jsonl.zst
while [ "$(stat -c %s $F)" != 22086532809 ]; do sleep 60; done
hostname
zstd -dc $F | /scratch/ralbe/miniconda3/envs/cirr_segm/bin/python $X/training/extract_evaldb.py $X/data/evaldb 23
