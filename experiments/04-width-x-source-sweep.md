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

At 512 the order is **sf > edb > lc0** (sf vs lc0 not played directly).

Still running when this was written: 512 sf vs 4096 edb (275 games, sf ahead by a large margin, which includes the
speed cost of a 4096-wide net at a fixed time control), and the three 1024-wide pairings. Results for the rest
will be added here.

## Validation error (Stockfish-labelled held-out set; favours the Stockfish-trained nets)

| source | 512 | 1024 | 4096 |
|---|---|---|---|
| sf | 0.04013 | 0.03826 | pending |
| edb | 0.04766 | 0.04704 | 0.04589 |
| lc0 | 0.05502 | 0.05515 | pending |

Wider nets fit a little better for `sf` and `edb`; `lc0` gains nothing from 1024. The validation set is labelled by
Stockfish, so it biases towards the `sf` and `edb` nets; the matches are the real comparison.
