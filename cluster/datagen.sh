#!/bin/bash
#SBATCH --partition=titans
#SBATCH --exclude=comp-gpu[11]
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --nice=100
# self-play data: 16 processes x GAMES games at NODES nodes/move with the bundle-D options -> data/selfplay/g<task>_<proc>.bin
X=/scratch/ralbe/chess_nnue/exp2; GAMES=${1:-25000}; NODES=${2:-5000}
hostname
c="setoption name EvalFile value $X/nets/C_hkb8pair_sfold_s2/C_hkb8pair_sfold_s2.nnue\nsetoption name Hash value 16\n"
for kv in UseFinny=1 UseLazyAcc=1 UseContHist=1 UseCaptHist=1 UseSingular=1 SingMargin=3 RfpMargin=81 RfpImp=24 NmpEvalDiv=196 LmpBase=4 FutBase=99 FutMul=114 SeeQuiet=26 SeeNoisy=99 HistDiv=5715 LmrBase=85 LmrDiv=206 AspDelta=17; do
  c+="setoption name ${kv%%=*} value ${kv#*=}\n"; done
for p in $(seq 0 15); do
  seed=$(( SLURM_ARRAY_TASK_ID * 100 + p + 1000 ))
  printf "${c}datagen $X/data/selfplay/g${SLURM_ARRAY_TASK_ID}_$p.bin $GAMES $NODES $seed\nquit\n" | $X/bin/datagen > $X/logs/datagen_${SLURM_ARRAY_TASK_ID}_$p.log 2>&1 &
done
wait
