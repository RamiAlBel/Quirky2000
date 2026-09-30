#!/bin/bash
#SBATCH --partition=titans
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp-gpu08
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# after the teacher T1024 is trained: label the eval-less July positions, prune T1024 to 512/384 lanes (+ stage 2
# with a longer polish), then queue the distillation run hkb8_sft (SF + teacher-labelled positions).
set -x
X=/scratch/ralbe/chess_nnue/exp2; PY=/scratch/ralbe/miniconda3/envs/cirr_segm/bin/python
cd $X/training
$PY label_teacher.py T1024 2>&1 | grep -v Warning
for w in 512 384; do
  $PY prune.py T1024 P$w $w 2>&1 | grep -v Warning
  $PY stage2.py P$w P${w}_s2 --polish_pos 300000000 2>&1 | grep -v Warning
done
[ -f $X/data/jul/noeval_teacher.i16 ] && STAGE2=" " sbatch --export=ALL -J hkb8_sft -o $X/logs/train_hkb8_sft.out $X/train.sh hkb8_sft --features hkb --kb 8 --data sf:1,teacher:0.25
