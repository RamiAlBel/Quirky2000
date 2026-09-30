#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
# Download one month of Leela test80 self-play (Stockfish binpack format, filtered by linrock) from HuggingFace.
F=${F:-test80-2024-06-jun-2tb7p.min-v2.v6.binpack.zst}
D=/scratch/ralbe/chess_nnue/exp2/data/lc0
cd $D && hostname && date
curl -sSL -C - --retry 20 --retry-delay 30 -o $F "https://huggingface.co/datasets/linrock/test80-2024/resolve/main/$F"
ls -la $F; zstd -t $F && echo DOWNLOAD_OK; date
