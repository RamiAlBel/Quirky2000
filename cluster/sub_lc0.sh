#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
# lc0.bin (2B, file order) -> lc0s.bin: every 7th block of 100k records (~286M, similar to sf.bin 356M / edb.bin 280M)
# spread over the whole month; the last block (validation tail of the file) is kept last.
D=/scratch/ralbe/chess_nnue/exp2/data/lc0
/scratch/ralbe/miniconda3/envs/cirr_segm/bin/python - <<PY
import numpy as np
R=70; B=100_000; K=7
src=np.memmap("$D/lc0.bin",mode="r",dtype=np.uint8).reshape(-1,R)
n=len(src); nb=n//B
with open("$D/lc0s.bin.tmp","wb") as f:
    for b in range(0,nb,K):
        f.write(src[b*B:(b+1)*B].tobytes())
import os; os.replace("$D/lc0s.bin.tmp","$D/lc0s.bin")
print("records", os.path.getsize("$D/lc0s.bin")//R)
PY
echo SUB_OK
