# Experiments

Every idea is a switch (a UCI option for the engine, a flag for the trainer) that is off by default. Off means
identical to the previous version, which is checked with bench node counts. An idea is accepted only if it beats
the current baseline in a test; otherwise it is rejected or left as neutral.

## How things are tested

- **Elo:** SPRT with [fastchess](https://github.com/Disservin/fastchess), the engine against itself with one switch
  changed. 6s+0.06s per game, 1 thread, `8moves_v3` openings with colours swapped, H0 = 0 Elo, H1 = +8 (later +6),
  alpha = beta = 0.05. The important winners were re-checked at 40+0.4. Scripts: `cluster/sprt.sh`.
- **Nets:** validation error on a held-out set (MSE of `tanh(cp/400)`, with the engine's quantized first layer, lower
  is better) and then a round-robin tournament or SPRT. Validation error is a filter, not a verdict: **Elo tracks the
  speed of the net as much as its accuracy** (see `01-nets.md`). Scripts: `cluster/nettourney.sh`, `training/xcheck4.py`
  (engine output equals PyTorch, incremental accumulator equals full recompute).
- **Search constants:** SPSA (`tools/spsa.py`, `cluster/spsa.sh`), 800 iterations x 22 games.

## Write-ups

| file | contents |
|---|---|
| [`01-nets.md`](01-nets.md) | net architecture and training experiments (features, activations, width, data, epochs, heads) |
| [`02-engine-and-search.md`](02-engine-and-search.md) | accepted engine/search/time/book changes with their Elo |
| [`03-rejected-and-neutral.md`](03-rejected-and-neutral.md) | everything that did not help, and what it cost |
| [`04-width-x-source-sweep.md`](04-width-x-source-sweep.md) | current experiment: net width x training-label source |
| [`05-specialised-weights.md`](05-specialised-weights.md) | king buckets 16/32, factorizer, phase/colour weight sets, opening expert nets: stopped, no benefit |
| [`06-opening-books.md`](06-opening-books.md) | opening book shoot-out (12 books, round robin + joint rating fit): Cerebellum chosen, about +60 Elo over no book |
| [`07-round-h.md`](07-round-h.md) | Round H (Coda-inspired): Leela-trained threat net with bullet (+34 vs W512_sf), search features, eval error by game phase |
| [`LEDGER.md`](LEDGER.md) | the raw log, in the order things happened (cluster paths and job ids left in) |
| [`options/`](options) | the exact UCI option sets of bundles D, E1 and E |

## Reading the numbers

`+24.1 +-9.9` means an estimated Elo gain with its 95% error. "H1" means SPRT accepted the gain, "H0" that it
accepted "no gain". A result stopped without a verdict is marked neutral. Numbers with few games (below about 1000)
are indications, not measurements.
