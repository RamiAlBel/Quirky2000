# Engine and search: accepted changes

SPRT against the previous best with one switch changed (6+0.06, 1 thread) unless a different time control is
given. Option names are the UCI switches. Details of what each does: `../docs/ENGINE_DETAILS.txt`.

## Round 1: from the first net, one switch at a time

| id | change (switch) | Elo |
|---|---|---|
| S1 + S2 | refresh cache (`UseFinny`) + lazy accumulator (`UseLazyAcc`); results identical, +7% nodes/s | +14.9 +-9.1 (2334 games) |
| S3 | continuation history (`UseContHist`) | +25.7 +-12.5 (1312) |
| S9 | time management: soft/hard limit, stretched while the best move changes (`UseTM`) | +55.4 +-22.7 (902) |
| capthist | capture history (`UseCaptHist`) | +7.7 +-5.9 (5684) |
| B1 | Syzygy 3-4-5 through Fathom (`SyzygyPath`) | +8.65 +-6.4 |
| B2 | own opening book from July 2200+ games, Polyglot format (`UseBook`) | +21.5 +-10.9 (from the start position) |
| B3 | pondering: on ponderhit the clock counts from the start of pondering | +107.5 [+91, +124] (750 games, wall-clock match) |
| singular | singular extensions with margin 3 (`UseSingular`, `SingMargin`) | +21.7 +-10.5 (1350) |
| net C1 | new net vs first net t25p | +185.7 +-36.0 |
| all together | C1 + the switches above vs t25p all off | **+353.5 +-50.9 at 6+0.06; +254.1 +-36.5 at 40+0.4** |

SPSA tuning of 16 search constants (800 iterations x 22 games) produced bundle D's values (`options/D_opts.txt`).
Bundle D vs C1: +26.4 +-11.8 (6+0.06), +22.7 +-9.4 (40+0.4).

## Round E: on top of bundle D

| id | change (switch) | Elo |
|---|---|---|
| E-corr2 | correction history v2: running weighted average keyed by pawn structure and non-pawn pieces (`UseCorr2`) | +24.1 +-9.9 (1548) |
| E-dblext | double singular extension, non-PV, at most 6 per line (`UseDblExt`) | +20.7 +-8.6 (1566) |
| E-nodetm | soft time limit scaled by the best move's share of nodes (`UseNodeTM`) | +14.9 +-7.1 (2188) |
| E-aspfh | aspiration fail high: re-search one ply shallower (`UseAspFH`) | +11.3 [+4, +18] (3549) |
| E-deeper | late-move re-search one ply deeper/shallower depending on the score (`UseDeeper`) | +12.5 [+5, +20] (3533) |

Stacked: the first three (E1) vs D: **+49.2 +-13.9** (732 games, H1). Later checks, stopped early on request:
E vs E1 +18.9 [+5.9, +32.0] (790 games); E vs D with 8 threads at 10+0.1 +66.2 [+34.5, +99.0] (85 games);
E1 vs D at 40+0.4 +45.5 [+29.2, +61.9] (369 games). All clearly positive, so **bundle E** (`options/E_opts.txt`) is the
recommended configuration.
