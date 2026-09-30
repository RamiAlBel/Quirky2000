# Width x source sweep - results

**Current champion: W512_sf**  (eliminated: W512_lc0, W512_edb, W4096_edb)

## Matches (TC 60+0.5, 1 thread; winner stays on)

| A | B | games | A +W =D -L | Elo(A) +/- | winner |
|---|---|---|---|---|---|
| W512_edb | W512_lc0 | 371 | +94 =212 -65 | +27 +/- 23 | W512_edb |
| W512_sf | W512_edb | 371 | +105 =234 -32 | +69 +/- 21 | W512_sf |
| W512_sf | W4096_edb | 299 | +137 =155 -7 | +162 +/- 26 | W512_sf |

## Other head-to-head matches (not part of the ladder path; short ones are indications only)

| A | B | games | A +W =D -L | Elo(A) +/- | status |
|---|---|---|---|---|---|
| W1024_sf | W1024_edb | 51 | +10 =37 -4 | +41 +/- 49 | stopped early |
| W1024_sf | W1024_lc0 | 50 | +11 =34 -5 | +42 +/- 54 | stopped early |
| W1024_sf | W512_sf | 174 | +11 =137 -26 | -30 +/- 24 | running |

## Nets (val_mse on the Stockfish-labelled validation set, lower = better; not a strength measure)

| net | epochs done | final val_mse (after stage 2) |
|---|---|---|
| W512_sf | 20/20 | 0.04013 |
| W1024_sf | 20/20 | 0.03826 |
| W4096_sf | 20/20 | 0.03748 |
| W512_edb | 20/20 | 0.04766 |
| W1024_edb | 20/20 | 0.04704 |
| W4096_edb | 20/20 | 0.04589 |
| W512_lc0 | 20/20 | 0.05502 |
| W1024_lc0 | 20/20 | 0.05515 |
| W4096_lc0 | 20/20 | 0.05366 |
