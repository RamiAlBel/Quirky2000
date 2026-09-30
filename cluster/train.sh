#!/bin/bash
#SBATCH --partition=titans
#SBATCH --gres=gpu:1
#SBATCH --exclude=comp-gpu08
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:45:00
# usage: [STAGE2="stage2 flags"] sbatch -J NAME -o logs/train_NAME.out train.sh NAME [train_v4 flags...]
#   STAGE2 set (may be " ") -> after training, run stage2.py NAME NAME_s2 $STAGE2 in the same job
# Trains in <=45 min chunks (short jobs backfill past the long GPU array jobs) and resubmits itself
# until train_v4.py finishes (wide nets: TLIM=hh:mm:ss MAXMIN=minutes for longer chunks,
# pass --time=$TLIM to the first sbatch too) (it exits 3 when it stopped for time; resume is automatic).
PY=/scratch/ralbe/miniconda3/envs/cirr_segm/bin/python
X=/scratch/ralbe/chess_nnue/exp2
NAME=$1; shift
cd $X/training
hostname
$PY train_v4.py --name $NAME --max_minutes ${MAXMIN:-38} --loaders 10 "$@" 2>&1 | grep -v Warning
rc=${PIPESTATUS[0]}
if [ $rc = 3 ]; then
  sbatch --time=${TLIM:-00:45:00} -J $NAME -o $X/logs/train_$NAME.out --open-mode=append --export=ALL $X/train.sh $NAME "$@"
elif [ $rc != 0 ]; then echo "FAILED rc=$rc"
elif [ -n "$STAGE2" ]; then  # t25p recipe: swap in the small head, fit it, polish (see stage2.py)
  $PY stage2.py $NAME ${NAME}_s2 $STAGE2 2>&1 | grep -v Warning
  [ ${PIPESTATUS[0]} = 0 ] || echo "FAILED stage2"
fi
