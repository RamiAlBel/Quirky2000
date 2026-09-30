# Opening books: Cerebellum is the best book

2026-09-30. Question: can a downloaded opening book make the engine stronger than its own book? Decision: use
**Cerebellum Light (3Merge, 2020-09-16)** with `UseBook=1 BookBest=1 BookDepth=255`.

## The books

- **Cerebellum Light 3Merge** (BrainFish project, CC BY-NC-SA 4.0): Polyglot, 11.1M entries for 11.0M positions,
  built from Stockfish analysis. Weight 255 marks the best move, 127 an alternative. Not included in this repository;
  download it from the BrainFish / Cerebellum page.
- **Ten books from the chessengines.blogspot download page:** optimus32, m11_2, sf211, morphius129c, chiron15,
  perfect2021, aaricia2012, ranomi170527 (the morphius file had to be re-sorted by key to be a valid Polyglot book).
- **Our own book** (`own_jul2200`, from July 2026 Lichess games of 2200+ players). It was tested both as deployed
  (weighted random choice, 20 plies) and as best-move with no depth limit.
- **No book.**

Engine change (commit "BookBest"): `BookBest=1` always plays the highest-weight move instead of a weighted random one,
and `BookDepth` now goes up to 255 plies. Both are off by default, and bench is unchanged.

## Method

Every engine is the same binary with bundle E options and the `W512_sf` net; only the book options differ
(`tools/books/engines_s1.txt`). A fastchess round robin at 6+0.06, 1 thread (`tools/books/rr.sh`, 4 shards) plays
every pair with colours swapped. It starts from `seeds4ply.pgn`, the 200 most common 4-ply lines of 2000+ Lichess games,
which cover 76% of those games. The books take over after the seed moves. A joint Bradley-Terry fit of all games
(`tools/books/rr_fit.py`, Ordo-style, draws = half points) gives each book one rating. The 95% intervals come from 300
bootstrap resamples of the games, and P(best) is the share of resamples in which a book has the top rating. This
compares all books with each other in one run instead of one SPRT per pair.

## Results

**SPRT, Cerebellum vs our deployed book** (from the start position, 6+0.06, [0, 5]):
**H1, +102.3 +-15.2** (553 games, +171 =370 -12).

**Stage 1: all 12 books** (1458 games, own deployed book = 0):

| # | book | Elo | 95% interval | P(best) |
|---|---|---|---|---|
| 1 | cerebellum | +51 | [+19, +86] | 55% |
| 2 | optimus32 | +43 | [+12, +75] | 23% |
| 3 | own_jul2200 (best move) | +37 | [+7, +74] | 11% |
| 4 | m11_2 | +34 | [-3, +68] | 7% |
| 5 | sf211 | +26 | [-11, +59] | 4% |
| 6 | morphius129c | +17 | [-14, +49] | 0% |
| 7 | no book | +15 | [-18, +51] | 0% |
| 8 | chiron15 | +14 | [-15, +44] | 0% |
| 9 | perfect2021 | +13 | [-15, +43] | 0% |
| 10 | aaricia2012 | +10 | [-20, +39] | 0% |
| 11 | ranomi170527 | +2 | [-35, +41] | 0% |
| 12 | own_jul2200 (deployed) | 0 | | 0% |

**Stage 2: the top 6** (2442 games, no book = 0):

| # | book | Elo | 95% interval | P(best) |
|---|---|---|---|---|
| 1 | **cerebellum** | **+63** | [+48, +79] | 70% |
| 2 | optimus32 | +57 | [+41, +72] | 26% |
| 3 | m11_2 | +49 | [+33, +65] | 4% |
| 4 | own_jul2200 (best move) | +5 | [-11, +19] | 0% |
| 5 | no book | 0 | | |
| 6 | sf211 | -3 | [-19, +13] | 0% |

Cerebellum's head-to-head results in stage 2: vs optimus32 +21 =110 -30, vs m11_2 +29 =121 -14,
vs own_jul2200 +39 =114 -10, vs no book +39 =120 -8, vs sf211 +36 =120 -6.

**Cerebellum vs no book, from `8moves_v3` openings** (the book takes over only after 8 moves):
+12.6 +-9.1 after 1600 games (stopped without a verdict).

## What we learned

- **The book is worth about 60 Elo.** The three strong books (Cerebellum, Optimus32, m11_2) are 50-60 Elo above no
  book, and Cerebellum is +102 over our deployed book.
- **Our own book, as deployed, was no better than no book** (stage 1: no book +15 [-18, +51] over it). The weighted random choice from human games follows popular
  moves, not good ones. Best-move mode helps (+37 in stage 1) but is still far behind the engine-analysed books.
- **Most of the gain comes from the first moves.** Starting after 8 fixed moves, Cerebellum gains only about +13, so the
  value is in choosing the opening, not in the later book moves.
- **Cerebellum and Optimus32 are close.** Cerebellum leads the joint fit mainly because it beats the weaker books by
  more. Head to head it lost to Optimus32 (+21 =110 -30, noisy). We chose Cerebellum because it is ahead in the fit
  and is the one book tested directly against ours.
- **Self-play only.** Every engine had the same net, so these numbers measure the book against a copy of our engine.
  Against Lichess opponents, who play other lines, the gain can be different.

## Files

`engine/book.cpp` (BookBest, BookDepth), `tools/books/` (round robin, fit, engine lists), `training/opp_tree.py`,
`training/book_rollout.py`, `training/evaldb_lookup.py`, `training/book_score.py` (a static comparison: simulated games
against a Lichess opponent model, scored with eval-DB Stockfish evals; not finished before the decision). Games:
`results/matches/book_roundrobin_stage{1,2}_*.pgn.xz`, `results/matches/cerebellum_vs_ownbook_sprt_6+0.06.pgn.xz`.
Raw log: [`LEDGER.md`](LEDGER.md), "Round G" section.
