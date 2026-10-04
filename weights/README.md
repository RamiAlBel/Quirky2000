# Weights: sf-trained nets, three widths

Final nets (after stage 2) of the width sweep in [`../experiments/04-width-x-source-sweep.md`](../experiments/04-width-x-source-sweep.md).
All three: 8 king buckets, pairwise activation, 20 epochs on the Lichess/Stockfish `sf` set (356M positions), small
8-neuron head with 8 piece-count output sets. The accumulator width is compiled into the engine, so build the matching one:

| file | width | size | val_mse | build | nodes/s (1 thread, bench 13) | sha256 |
|---|---|---|---|---|---|---|
| `W512_sf.nnue` | 512 | 5.9 MB | 0.04013 | `engine/build.sh` | 937k | `15d49b69930d...` |
| `W1024_sf.nnue` | 1024 | 11.8 MB | 0.03826 | `ACC=1024 engine/build.sh w1024` | 749k | `5a0b1c3743d4...` |
| `W4096_sf.nnue` | 4096 | 47.2 MB | 0.03748 | `ACC=4096 engine/build.sh w4096` | 320k | `604392fbced1...` |

```
printf 'setoption name EvalFile value weights/W1024_sf.nnue\nposition startpos\ngo movetime 1000\n' | bin/w1024
```

`val_mse` is measured on a Stockfish-labelled held-out set (lower is better). It is not a strength measure: the wider nets
fit better but search fewer nodes per second. For playing strength see [`../experiments/results/`](../experiments/results).
The net used in the tested configuration (bundle E) is `../assets/C1.nnue` (512 wide, `sf` + older set).
A different width than the build's is rejected on loading.

## Round H: Leela-trained net with threat inputs

| file | inputs | width | head | size | build | nodes/s (1 thread, bench 11) | sha256 |
|---|---|---|---|---|---|---|---|
| `L1024T_lc0.nnue` | HalfKA 16 king buckets + threat features (attacker, victim, squares) | 1024 | 32 -> 32 -> 1, 8 output buckets | 86.7 MB | `ACC=1024 EXTRA=-DNNUE_LNN6 engine/build.sh quirky_h` | 450k | `419100300039...` |

Trained with [bullet](https://github.com/jw1912/bullet) (`training/bullet/`) on six months of Leela test80 self-play
(Jan-Jun 2024, 20B positions, 200 superbatches), target = mix of the Leela score and the game result (result weight
0.2 -> 0.4). On version G's search it beats `W512_sf` by **+34 +/- 17 Elo** (1071 games, 8+0.08, UHO openings).
LNN6 format: only an `-DNNUE_LNN6` build loads it. Details in
[`../experiments/07-round-h.md`](../experiments/07-round-h.md).

```
printf 'setoption name EvalFile value weights/L1024T_lc0.nnue\nposition startpos\ngo movetime 1000\n' | bin/quirky_h
```
