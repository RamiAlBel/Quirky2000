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
