#!/bin/bash
#SBATCH --partition=titans
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp-gpu08
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# compute phase + hard strata, then queue the stratified-sampling variants (hkb8, July+old data)
X=/scratch/ralbe/chess_nnue/exp2; PY=/scratch/ralbe/miniconda3/envs/cirr_segm/bin/python
cd $X/training
$PY strata.py phase 2>&1 | grep -v Warning
$PY strata.py hard hkb8_sfold_s2 8 2>&1 | grep -v Warning
for v in "phase05:phase:0.5" "phase10:phase:1.0" "hard05:hard:0.5"; do
  n=hkb8_sfold_${v%%:*}; st=${v#*:}
  STAGE2=" " sbatch --export=ALL -J $n -o $X/logs/train_$n.out $X/train.sh $n --features hkb --kb 8 --data sf:3.5,old:1 --strata $st
done
