# Net width x training-label source (in progress)

Started 2026-09-29. Question: which accumulator width (512 / 1024 / 4096) and which label source works best?
Nine nets, all with the C1 recipe (8 king buckets, pairwise, 20 epochs, then the small head), nothing else changed.

| source | what it is | size used |
|---|---|---|
| `sf` | positions from Lichess games (July 2026), labelled with the Stockfish eval Lichess attached, plus the game result | 356M |
| `edb` | Lichess evaluation database: board positions analysed by Stockfish, mostly at high depth; no games, so no results | 280M |
| `lc0` | Leela Chess Zero self-play (linrock's `test80-2024-06` binpack): Leela's score and the self-play result | 286M of 2B extracted |

Notes on the Leela data (`tools/binpack2rec4.cpp`, `cluster/extract_lc0.sh`, `cluster/sub_lc0.sh`):
- Binpack scores are about 2.6x Stockfish centipawns, so labels are `score x 0.39`. That factor is a compromise:
  fitting the net C1's evals to the two sources gave 0.31 for Leela and 0.80 for `sf`.
- Filters: no missing/placeholder scores, ply >= 8, no captures, promotions or checks.
- The 286M subset takes every 7th block of 100k positions over the whole month, so it is comparable in size to the other
  two sources and not only the first days.

## Strength (self-play matches, 60+0.5, 1 thread, about 380 games each)

Nets play as *winner stays on*: the champion meets the latest finished net, the loser is out
(`cluster/ladder.py`, `cluster/match.sh`). The full, current table is generated into `runs/RESULTS.md`
(`cluster/report.py`).

| A | B | games | A +W =D -L | Elo(A) | winner |
|---|---|---|---|---|---|
| 512 edb | 512 lc0 | 371 | +94 =212 -65 | +27 +-23 | edb |
| 512 sf | 512 edb | 371 | +105 =234 -32 | +69 +-21 | **sf** |
| 512 sf | 4096 edb | 299 | +137 =155 -7 | +162 +-26 | **512 sf** (match stopped by hand) |

At 512 the order is **sf > edb > lc0** (sf vs lc0 not played directly).

The 4096 result is mostly speed plus the weaker source: on the same benchmark (depth 13, 1 thread) the 512 engine runs at
about 970k nodes/s and the 4096 engine at about 411k, and edb already lost to sf by 69 Elo at the same width. Every game
ended normally (no illegal moves, crashes or time forfeits). The width effect alone is measured by the sf-only pairs below.

Short indications, not results (matches stopped early, error bars 50 Elo or more):

| A | B | games | A +W =D -L | Elo(A) |
|---|---|---|---|---|
| 1024 sf | 1024 edb | 51 | +10 =37 -4 | +41 +-49 |
| 1024 sf | 1024 lc0 | 50 | +11 =34 -5 | +42 +-54 |

Running when this was written: **1024 sf vs 512 sf** (45 games so far, 1024 at -31 +-47, i.e. no difference yet).
Still training: 4096 lc0. 4096 sf is finished (weights in `../weights/`, results in `results/`). Time control 60+0.5 is only 1.7M nodes per move; a test at a
real node budget (10M nodes/s, 16 threads: about 15M nodes per move in bullet, 65M in blitz) is planned to see whether the
speed cost of the wide nets shrinks at a realistic budget.

Bullet with the thread count of the target PC (1+0, 16 threads per engine), 10 games each, indications only:
1024 sf vs 512 sf +2 =7 -1; **4096 sf vs 512 sf +0 =7 -3** (about 3x slower per node at 16 threads). See `results/README.md`.

## Validation error (Stockfish-labelled held-out set; favours the Stockfish-trained nets)

| source | 512 | 1024 | 4096 |
|---|---|---|---|
| sf | 0.04013 | 0.03826 | **0.03748** |
| edb | 0.04766 | 0.04704 | 0.04589 |
| lc0 | 0.05502 | 0.05515 | pending |

Wider nets fit a little better for `sf` and `edb`; `lc0` gains nothing from 1024. The validation set is labelled by
Stockfish, so it biases towards the `sf` and `edb` nets; the matches are the real comparison.
