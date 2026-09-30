# Quirky2000

A UCI chess engine with a hand-built NNUE evaluation, plus the full record of the experiments that shaped it.
C++17, one binary, AVX2/BMI2. Search is alpha-beta with modern pruning; the evaluation is a quantized
neural net trained on Lichess games and Stockfish/Leela evaluations. Every search or net idea was added as a UCI
switch (default off), then tested against the current best with SPRT before it was accepted.

> The engine currently reports itself as `LiteNNUE-native` in the UCI `id name` line (its original working name).

## Layout

| path | what |
|---|---|
| `engine/` | the engine (C++). `build.sh` builds it, `build.bat` is the old MSVC build |
| `training/` | PyTorch net training: data extraction, `nnue4.py` (net + export), `train_v4.py`, `stage2.py`, checks |
| `cluster/` | SLURM scripts used on the DTU cluster (training chunks, SPRT, tournaments, ladder). Paths at the top of each script are cluster-specific |
| `tools/` | SPSA tuner, PGN scoring, ponder match, Leela binpack to training-record converter |
| `experiments/` | **the record of what was tried**: index, per-area write-ups, raw ledger, option sets |
| `docs/` | `ENGINE_DETAILS.txt` (what the engine does, in plain words), `CHANGES_E.txt` |
| `assets/` | the current net (`C1.nnue`, 5.9 MB) and an opening book (`book_jul2200.bin`) |

## Build and run

```
engine/build.sh                      # -> bin/quirky (needs g++, an AVX2 + BMI2 CPU; clones Fathom for Syzygy)
ACC=1024 engine/build.sh w1024       # the net's accumulator width is compiled in and must match the .nnue file
```

```
$ bin/quirky
setoption name EvalFile value assets/C1.nnue
position startpos
go movetime 1000
```

The tested configuration ("bundle E") is the set of UCI options in `experiments/options/E_opts.txt`
(`EvalFile` there is a cluster path; use `assets/C1.nnue`). Bench with those options: depth 13, 574,687 nodes.
Everything the engine does, and the Elo each switch was worth, is in `docs/ENGINE_DETAILS.txt`.

## Strength, as measured

All numbers are self-play SPRT (fastchess, 1 thread, 8moves_v3 openings, colours swapped), not rating-list results.

| step | Elo | time control |
|---|---|---|
| all accepted engine switches + C1 net vs the first net (t25p, all switches off) | +354 | 6+0.06 |
| same | +254 | 40+0.4 |
| bundle D vs C1 | +26 | 6+0.06 |
| E1 (D + three new search switches) vs D | +49 | 6+0.06 |
| bundle E (E1 + two more switches) vs E1 | +19 (790 games, stopped early) | 6+0.06 |

## What was tried

See [`experiments/README.md`](experiments/README.md). Short version:

- **Accepted:** king-bucket features, pairwise activation, two-stage training, refresh cache + lazy accumulator,
  continuation/capture history, singular + double extensions, correction history v2, time management (incl. node-share),
  pondering, own opening book, Syzygy, SPSA-tuned constants.
- **Rejected:** threat features (25% slower), wider/narrower nets than 512, pruning/distillation, an uncertainty head,
  ProbCut, razoring, phase-balanced or hard-example sampling, most extra history tables.
- **In progress:** a 3 x 3 sweep of net width (512 / 1024 / 4096) against training-label source
  (Lichess games with Stockfish evals / Lichess eval database / Leela self-play).

## License

GPL-3.0, see `LICENSE`. Third-party code is not vendored: Fathom (Syzygy probing, cloned by `build.sh`) and the
nnue-pytorch binpack headers (only needed to rebuild `tools/binpack2rec4.cpp`).
