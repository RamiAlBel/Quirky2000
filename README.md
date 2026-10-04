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
| `weights/` | the sf-trained nets of the width sweep: 512, 1024 and 4096 wide (`weights/README.md` says how to build for each) |
| `assets/` | version E's net (`C1.nnue`, 5.9 MB) and its own opening book (`book_jul2200.bin`) |
| `deploy/G/` | version G for the Windows lichess-bot PC: exe, net, config block, install steps |

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

## Versions

| version | what changed | options |
|---|---|---|
| C1 | king-bucket net `C1.nnue` + all accepted search switches (+354 vs the first net) | |
| D | SPSA-tuned search constants (+26 vs C1) | `D_opts.txt` |
| E1 | D + correction history v2, double extensions, node-share time management (+49 vs D) | `E1_opts.txt` |
| E | E1 + aspiration fail-high and deeper/shallower re-search (+19 vs E1). Net `assets/C1.nnue`, own book | `E_opts.txt` |
| F | E's options with the sf-trained 512 net `weights/W512_sf.nnue` | `F_opts.txt` |
| **G (current)** | F + the Cerebellum opening book (`UseBook=1 BookBest=1 BookDepth=255`), +102 vs E's book | `G_opts.txt` |
| H-lc0T (testing) | G's search with the Leela-trained threat net `weights/L1024T_lc0.nnue` (+34 +/- 17 vs G at 8+0.08); build `ACC=1024 EXTRA=-DNNUE_LNN6 engine/build.sh quirky_h`. See `experiments/07-round-h.md` | `H_lc0T_opts.txt` |

Option files are in `experiments/options/`. `EvalFile`/`SyzygyPath` there are cluster paths.
Bench (depth 13) with E's options: 574,687 nodes with `assets/C1.nnue`, 378,755 with `weights/W512_sf.nnue` (F, G).

**Running G on the lichess-bot PC:** [`deploy/G/README.md`](deploy/G/README.md) (Windows exe, net, config block,
where to get the Cerebellum book and how to check it).
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
- **Width x source sweep (done):** 512 / 1024 / 4096 wide nets on Stockfish-labelled Lichess games, the Lichess eval
  database and Leela self-play. Stockfish labels won, and 1024/4096 were not stronger than 512 in games (`experiments/results/`).
- **Specialised weights (stopped, no benefit):** more king buckets, factorizer, phase/colour weight sets, opening experts
  (`experiments/05-specialised-weights.md`).
- **Opening books:** Cerebellum chosen out of 12 books (`experiments/06-opening-books.md`).

## License

GPL-3.0, see `LICENSE`. Third-party code is not vendored: Fathom (Syzygy probing, cloned by `build.sh`) and the
nnue-pytorch binpack headers (only needed to rebuild `tools/binpack2rec4.cpp`).
