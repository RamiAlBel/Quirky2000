# Nets

Validation error = MSE of `tanh(cp/400)` on a held-out set, with the engine-like quantized first layer (lower is
better). Run-to-run noise is about 1%. Source: [`LEDGER.md`](LEDGER.md), `options/netT1_final.txt`.

## Architecture that won (C1)

```
sparse features (own king bucket x piece x square, 8 buckets, mirrored) -> 2 x 512 accumulator
  -> pairwise product (halves multiplied) -> 512 -> 8 neurons (output set chosen by piece count) -> [x, x^2] -> 1
```

Trained in two stages (`cluster/pipe.sh`, `training/stage2.py`): stage 1 trains the features under a wider head
(16 -> 32, crelu); stage 2 swaps in the small fast head, fits it on the frozen features, then one polish epoch. A
one-stage run of the small head alone learned worse (val 0.05168 vs 0.04889): the 8-neuron head throttles feature learning.

C1 = king buckets (8) + pairwise + July 2026 Lichess games with Stockfish evals (356M positions) + an older 100M set,
20 epochs. Validation 0.03922. Against the first net (t25p) in SPRT: **+185.7 +-36.0 Elo**.

## What each factor did

Everything below differs from a control by one factor unless noted.

| factor | result | verdict |
|---|---|---|
| king-bucket features 8 / 16 / 32 | val 0.04589 / 0.04699 / 0.04728 (control 0.04889); tournament: 8 buckets +47 vs the field, t25p -50 | **8 buckets** |
| pairwise activation | val about equal, +7% nodes/s (881k vs 828k) | **accept** |
| more data (July 356M + old 100M) | val 0.04097 -> 0.03922 with the other changes | **accept** |
| training length | 10 -> 20 epochs: stage 1 0.04259 -> 0.04078; 40 epochs 0.03828 (20: 0.03890); 80 epochs 0.03820, saturating; 80 vs 20 in SPRT +3.1 +-3.1 (no verdict, 20000-game cap) | **20 epochs** |
| threat features (attacked-by-opponent) | val 0.03769 (better) but -25% nodes/s; SPRT -29 +-16 (794 games) | rejected: speed |
| width 256 / 384 / 768 / 1024 | 256: val 0.04204, SPRT -10.4 +-9.3 (H0); 384: 0.04036, -4.0 +-6.6 (H0); 768 and 1024 not stronger at this training budget | **512** |
| head 16 neurons instead of 8 | -0.5% error, -4.5% nodes/s; SPRT -0.3 +-6.0 | neutral, keep 8 |
| PSQT shortcut | val 0.04936 | rejected |
| int8 feature weights | val 0.04942 | rejected |
| EMA weights | val 0.04960 | no help |
| loss power 2.5 | val 0.05060 (metric is MSE-biased) | rejected |
| pruning a 1024 net to 512 / 384 | 0.04633 / 0.04758, worse than 512 from scratch | rejected |
| distillation from a 1024 teacher | 0.04162 vs 0.04097 | no gain with a weak teacher |
| hard-example sampling | 0.04806 vs 0.04097 | rejected |
| phase-balanced sampling | val equal (0.03920 vs 0.03922); SPRT -2.55 +-5.77 (5588 games) | rejected |
| output-bucket schemes (8/16 by piece count, +queens, non-pawn material) | 0.03925 / 0.03933 / 0.03950, current 0.03922 | no difference: the features already encode the phase |

## Head search (before C1)

The head on top of a frozen 512-wide feature layer was searched with Optuna (layer sizes, activations, output
buckets, scale, quantization-aware training). The winner "t25p" (8 neurons, dual activation, 8 piece-count buckets,
1 polish epoch) beat the plain head by +51 Elo at 10+0.1 and +71 +-14 at 40+0.4. **Elo followed evaluation speed,
not small validation gains**: a head with the best validation error (32 neurons) was -79 Elo.

An earlier width sweep of the accumulator (256 / 512 / 1024 / 2048) at 10+0.1 had already picked 512.

## Data

| source | size | labels |
|---|---|---|
| Lichess games, July 2026, with Stockfish `%eval` (`training/extract_v4.py`) | 356M positions | Stockfish cp + game result |
| Lichess games, Aug 2026 | 100M | Stockfish cp |
| Lichess eval database (`training/extract_evaldb.py`) | 410M lines -> 280M quiet positions | deep Stockfish cp, no game results |
| Leela self-play, `test80-2024-06` (`tools/binpack2rec4.cpp`) | 2B extracted, 286M used | Leela score x 0.39 to match Stockfish cp |
| own engine self-play (`datagen` command) | 9M | 5000-node search score; not used for a net |

See [`04-width-x-source-sweep.md`](04-width-x-source-sweep.md) for the comparison of these sources.
