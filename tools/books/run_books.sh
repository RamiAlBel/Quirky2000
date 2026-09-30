#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=46
#SBATCH --mem=120G
#SBATCH --time=06:00:00
# Static opening-book comparison for our bot (training/book_rollout.py, evaldb_lookup.py, book_score.py).
# usage: sbatch books/run_books.sh TAG NAME=PATH[:best|:weighted] ...  -> books/TAG.md
# Every .bin in /scratch/ralbe/chess_nnue/books is added automatically (best-move pick).
set -e
X=/scratch/ralbe/chess_nnue/exp2; PY=/scratch/ralbe/miniconda3/envs/cirr_segm/bin/python; T=$X/books
TAG=$1; shift
specs="$@"
for f in /scratch/ralbe/chess_nnue/books/*.bin; do [ -f "$f" ] && specs+=" $(basename "$f" .bin | tr ' ' '_')=$f:best"; done
echo "books: $specs"
cd $X/training
$PY book_rollout.py $T/opp2000.npz $T/$TAG.rollouts.pkl $specs
$PY book_score.py $T/$TAG.rollouts.pkl keys $T/$TAG.keys.npy
zstd -dc $X/data/evaldb/lichess_db_eval.jsonl.zst | $PY evaldb_lookup.py $T/$TAG.keys.npy $T/$TAG.evals.pkl 44
$PY book_score.py $T/$TAG.rollouts.pkl score $T/$TAG.evals.pkl $T/$TAG.md
