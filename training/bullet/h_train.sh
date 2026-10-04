#!/bin/bash
#SBATCH --partition=titans
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp-gpu08
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
# one bullet training run; config via QK_* env (see bullet/examples/quirky/main.rs). Resumes from the newest checkpoint.
B=/scratch/ralbe/chess_nnue/exp2/bullet
export LD_LIBRARY_PATH=/opt/cuda/cuda-12.6/lib64:$LD_LIBRARY_PATH
export QK_OUT=${QK_OUT:-$B/ckpt/$QK_NAME}
mkdir -p $QK_OUT
hostname; nvidia-smi --query-gpu=name --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES%%,*}; date
env | grep ^QK_ | sort
last=$(ls -d $QK_OUT/$QK_NAME-* 2>/dev/null | sed 's/.*-//' | sort -n | tail -1)
if [ -n "$last" ] && [ -d $QK_OUT/$QK_NAME-$last/optimiser_state ]; then
  export QK_RESUME=$QK_OUT/$QK_NAME-$last QK_START=$((last + 1))
  echo "resume from $QK_RESUME"
fi
cp $B/bullet/target-cuda/release/examples/quirky $QK_OUT/quirky.bin 2>/dev/null
$QK_OUT/quirky.bin
date; echo TRAIN_DONE
