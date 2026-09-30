#!/bin/bash
#SBATCH --partition=cyclopes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
# Interleaved single-core bench of several configs of one binary: sbatch speed.sh BIN NET REPS "cfgA" "cfgB" ...
# cfg = space-separated Name=Value UCI options, e.g. "UseFinny=1 UseLazyAcc=1"
BIN=$1; NET=$2; REPS=$3; shift 3
hostname; lscpu | grep "Model name"
CPU=$(taskset -pc $$ | awk -F": " "{print \$2}" | cut -d, -f1 | cut -d- -f1); echo "pinned to cpu $CPU"
for r in $(seq $REPS); do
  for cfg in "$@"; do
    cmds="setoption name EvalFile value $NET\n"
    for kv in $cfg; do cmds+="setoption name ${kv%%=*} value ${kv#*=}\n"; done
    printf "%-40s " "[$cfg]"
    printf "${cmds}bench 13\nquit\n" | taskset -c $CPU $BIN | grep "^bench:"
  done
done
