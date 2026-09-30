#!/bin/bash
# E (previous version) vs F512 / F1024 / F4096 at PC-equivalent blitz (3+2) and rapid (10+5), 10 games each:
# 5 openings x 2 colours, one Slurm job per opening.
X=/scratch/ralbe/chess_nnue/exp2
OPENINGS="13453 15614 41876 65866 67086"
for mode in rapid blitz; do
  if [ $mode = blitz ]; then B=180; I=2; TL=01:30:00; else B=600; I=5; TL=03:30:00; fi
  for f in F512 F1024 F4096; do
    out=$X/runs/pc_$mode; mkdir -p $out
    for o in $OPENINGS; do
      sbatch -J pc_${mode}_${f}_$o --time=$TL -o $X/logs/pc_${mode}_E_vs_${f}_o$o.out $X/pc_match.sh E $f $B $I $o $out
    done
  done
done
