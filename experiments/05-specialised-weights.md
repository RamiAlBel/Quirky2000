# Specialised weights: king buckets, factorizer, opening experts (stopped, no benefit)

Round G, 2026-09-30. Question: do several sets of weights used in different situations beat the single 512-wide sf net
(`weights/W512_sf.nnue`, bundle E options)? King buckets and opening experts were **searched and gave no benefit**;
work on them was stopped on 2026-09-30. Phase and colour weight sets are covered at the end.

Everything uses the W512_sf recipe (512, pairwise, 8 king buckets, 20 epochs on `sf`, then the small head) with one
factor changed. Validation: MSE of `tanh(cp/400)` with the quantized first layer (lower is better). SPRTs:
`cluster/gsprt.sh`, 6+0.06, H0 = 0, H1 = +5, both sides on the same binary.

## What was built

- **More king buckets:** `--kb 16` / `--kb 32` (HalfKA buckets), no engine change.
- **Factorizer:** `--fact 1`, a shared piece-square table added to every bucket during training and folded into the
  rows at export (`nnue4.py: folded_ft, load_folded`).
- **Weight sets per phase / colour:** `--sets phase2|phase3|phase4|color`, one feature-transformer copy per band.
  The engine reads them as LNN5 (`engine/nnue.cpp`, commit "LNN5 weight sets"), refreshing the accumulator when a
  capture changes the band. Checked with `training/xcheck4.py`.
- **Opening experts:** W512_sf fine-tuned (all weights, lr 1e-4, 2 x 30M positions) on one opening family, taken from the
  first two plies (`training/subsets.py`, data with game/ECO sidecar from `training/extract_open.py`). The engine
  loads the family's net for the whole game (`ExpertRules`, commit "ExpertRules"). Each SPRT used books of its own
  family.

## Results

| net | stage-1 val | final val | Elo (SPRT) | decision |
|---|---|---|---|---|
| control (W512_sf recipe, new seed) | 0.03967 | 0.04027 | – | reference (run-to-run noise about 0.35%) |
| 16 king buckets | 0.03969 | 0.03997 | **-8.5 +-5.8** (3940 games, H0) | rejected |
| 32 king buckets | 0.03978 | 0.04014 | **-11.2 +-6.7** (3132 games, H0) | rejected |
| 8 buckets + factorizer | 0.03923 | 0.04022 | – | no gain |
| 16 buckets + factorizer | 0.03939 | 0.04030 (0.04044 with longer head fit) | – | no gain |
| 32 buckets + factorizer | 0.03933 | 0.04177 | – | worse |

| opening expert | own-family val vs control | Elo (SPRT, about 8k games) |
|---|---|---|
| 1.e4 e5 | -0.7% | +1.5 [-2.0, +5.0] |
| Sicilian | -0.7% | -2.0 [-5.9, +1.8] |
| 1.d4 d5 | -0.4% | +2.0 [-1.5, +5.6] |
| 1.d4 Nf6 | +0.5% | -2.2 [-6.6, +2.2] |
| flank | -0.2% | +0.8 [-3.7, +5.2] |
| 1.e4 other | -0.1% | 0.0 [-4.6, +4.6] |
| 1.d4 other | +0.5% | **-7.0 +-5.6** (4750 games, H0) |

Every expert was worse than the control outside its own family, and inside it the gain (at most 0.7% error) is too
small to show in games.

## Why the extra king buckets lose

The net's error is not where more king buckets help: on held-out sf positions they are better only with 2-8 pieces
(0.041 vs 0.043), not at 9-24 pieces where most of the error is. The same data is spread over 2-4x as many feature
weights, and the bigger feature table costs about 1.4% speed.

## Factorizer: a bug, found late

The factorizer's stage-1 lead (up to -1.1%) vanished in stage 2. Cause: a folded row (row + shared table) can reach
+-2, about 1% of weights are above the +-1 clip, and stage 2 clamped them back to +-1 after every step. Fixed in this
commit (`Net4.ft_clip`; stage 2 allows 2 x `FT_CLIP` for folded factorizer nets). The export already handled +-2. The
re-runs with the fix were cancelled when the line was stopped, so **the fixed factorizer was not measured**. Unlike
extra buckets, it costs nothing at inference, so it is the one idea here that could be worth a retry.

## Phase / colour weight sets

| net | stage-1 val | final val | decision |
|---|---|---|---|
| 2 phase sets (>= 17 / <= 16 pieces) | 0.04007 | 0.04020 | no gain |
| 2 phase sets + factorizer | 0.03978 | 0.04029 | no gain (factorizer bug above) |
| 3 phase sets + factorizer | 0.04005 | 0.04054 | no gain (factorizer bug above) |
| 4 phase sets + factorizer | 0.04010 | 0.04093 | worse, memorises (train loss 0.029 vs 0.034) |
| colour sets + factorizer | 0.04162 | 0.04179 | worse |

## Files

`training/nnue4.py` (`sets`, `fact`, `ft_clip`), `training/train_v4.py` (`--sets --fact --init --subset`),
`training/stage2.py`, `training/subsets.py`, `training/eval_cats.py` (error by pieces / king square / opening / rating),
`training/extract_open.py`, `training/xcheck4.py` (LNN5 checks), `cluster/g_sweep.sh`, `cluster/g_expert.sh`,
`cluster/gsprt.sh`, `cluster/extract_open.sh`. Raw log: the "Round G" section of [`LEDGER.md`](LEDGER.md).
