# Rejected and neutral ideas

"Rejected" means SPRT accepted H0 (no gain) or the result was clearly negative. "Neutral" means stopped without a
verdict with a small Elo estimate; these are not adopted, because each switch adds code and time.

## Search

| idea | result |
|---|---|
| correction history v1 (`UseCorrHist`) | -13.5 +-13.4 (2568 games, H0). A bug was found (it learned from tablebase scores) and fixed, but it was still -47.7 +-17.3, because the old table was an unbounded sum that saturated. Replaced by v2 (+24, accepted) |
| singular extensions, first version (depth 8+, margin 2 x depth) | -16.3 +-11.2 (H0). Retried with margin 3: +21.7, accepted |
| razoring | +2.3 +-4.7 after 8802 games, neutral |
| ProbCut | +1.9 +-5.3 after 6798 games, neutral |
| uncertainty ("sigma") head for pruning margins and LMR | -9 to -44 depending on variant, including at fixed node counts, so it is not only a speed cost. Sigma from the eval head's hidden layer: no better than a table. Rejected |
| flag mode against opponents low on the clock | -2 (1917 games) |
| ttPv flag in the transposition table | 0.0 (3136 games) |
| capture futility | +3.8 (2821 games) |
| RFP returning (eval + beta) / 2 | +3.6 (3383 games) |
| LMR extra reduction when the TT move is a capture | +2.7 (3119 games) |
| 4-ply continuation history | +1.0 (about 1700 games) |
| history pruning of quiets | -1.3 |
| pawn history | +0.6 |
| prior-move bonus | -0.4 |
| two-net scheme (128-wide net for lopsided positions) | built, never SPRT-tested; the 128-wide net was worse on validation (0.05152) |

## Nets

See the table in [`01-nets.md`](01-nets.md): threat features (-29 Elo, 25% slower), widths other than 512,
pruning and distillation, PSQT shortcut, int8 feature weights, EMA, loss power 2.5, hard-example and phase-balanced
sampling, 16-neuron head, alternative output-bucket rules, 80 epochs.

## Specialised weights (Round G, stopped 2026-09-30)

See [`05-specialised-weights.md`](05-specialised-weights.md). 16 king buckets -8.5 +-5.8 (H0), 32 king buckets
-11.2 +-6.7 (H0), factorizer no gain after stage 2, opening expert nets neutral (-2.2 .. +2.0 at about 8k games each)
and 1.d4-other -7.0 (H0). Searched, no benefit; stopped by decision.

## Dropped by decision

GPU net variants with a WDL blend (0.2 / 0.4) and a net on the Lichess eval database were cancelled before they
started on 2026-09-28 in order to stop work on unpromising ideas. The eval-database net was picked up again in the
sweep in [`04-width-x-source-sweep.md`](04-width-x-source-sweep.md).
