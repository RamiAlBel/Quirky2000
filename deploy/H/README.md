# Version H (provisional): install on the lichess-bot PC (Windows)

H = version G (same search settings, same Cerebellum book) + the new **Leela-trained threat net** `L1024T_lc0.nnue`
+ seven new search switches: `UseThreatHist UseCuckoo UseCont6 UsePawnHist UseHindsight UseTtBlend UseQsBlend`.

**Provisional.** The net is tested: +34.2 +- 16.6 Elo vs W512_sf on G's search (1071 games, 8+0.08, 1 thread).
The switches are the ones that scored above 0 in a rough test of each, one at a time, on this net
(5+0.05, about 210 games each, error bars about +-30 Elo, so none of them is proven yet):

| switch | Elo (1 at a time, vs H net with G's search) |
|---|---|
| `UseCuckoo` | +33 +- 32 |
| `UsePawnHist` | +22 +- 31 |
| `UseTtBlend` + `UseQsBlend` (tested together) | +20 +- 30 |
| `UseThreatHist` | +17 +- 29 |
| `UseCont6` | +15 +- 32 |
| `UseHindsight` | +8 +- 34 |

Left out (negative in the same test): `UseCorr5` (-20), `UseFhBlend` (-11), and `UseHistNorm`: Cont6 + PawnHist + HistNorm
together scored -23 +- 31, a warning that these switches may not simply add up. All seven together have not been tested,
and the search constants have not been re-tuned (SPSA) for this net yet. Expect updates. Any switch can be set to 0
in the config to turn it off.

## What is in this folder

| file | what |
|---|---|
| `quirky_H.exe` | the engine, Windows x64, needs an AVX2 + BMI2 CPU. Plain zig cross-compile (x86_64-windows-gnu), **not PGO** |
| `L1024T_lc0.nnue` | the net (same file as `weights/L1024T_lc0.nnue`, 87 MB) |
| `config_engine_block.yml` | the lichess-bot `engine:` settings for H |

Not included: `Cerebellum3Merge.bin`, the same book as G (see `deploy/G/README.md`, step 2). Copy it from your G folder.

## Install

Same as G: copy this folder to the bot PC (for example `...\Native NNUE Chess\H\`), put `Cerebellum3Merge.bin` next to
`quirky_H.exe`, and set the `engine:` block of `lichess-bot\config.yml` as in `config_engine_block.yml`. Keep a
backup of the G config so you can switch back.

## Check it before going live

In a `cmd` window in the H folder, start `quirky_H.exe` and paste:

```
setoption name EvalFile value L1024T_lc0.nnue
setoption name UseFinny value 1
setoption name UseLazyAcc value 1
setoption name UseContHist value 1
setoption name UseCaptHist value 1
setoption name UseTM value 1
setoption name UseSingular value 1
setoption name SingMargin value 3
setoption name RfpMargin value 81
setoption name RfpImp value 24
setoption name NmpEvalDiv value 196
setoption name LmpBase value 4
setoption name FutBase value 99
setoption name FutMul value 114
setoption name SeeQuiet value 26
setoption name SeeNoisy value 99
setoption name HistDiv value 5715
setoption name LmrBase value 85
setoption name LmrDiv value 206
setoption name AspDelta value 17
setoption name UseCorr2 value 1
setoption name UseDblExt value 1
setoption name UseNodeTM value 1
setoption name UseAspFH value 1
setoption name UseDeeper value 1
setoption name UseThreatHist value 1
setoption name UseCuckoo value 1
setoption name UseCont6 value 1
setoption name UsePawnHist value 1
setoption name UseHindsight value 1
setoption name UseTtBlend value 1
setoption name UseQsBlend value 1
bench 13
```

It must print `bench: 320326 nodes`. Then do the book check from `deploy/G/README.md`.

## Flag mode (games without increment)

A practical rule for 1+0 / 3+0, **not tested in games** (self-play cannot show it: an engine opponent never flags).
When neither side has an increment, the opponent has less than 30 s and we have at least 1.3x their time, a draw
counts as -50 cp for us. The engine then avoids threefold repetition and other drawing lines unless it is clearly
worse (below -50 cp a draw is still welcome), and keeps the game going so the opponent can run out of time.
With an increment it never switches on. In the lichess-bot log it shows as `info string flag mode: ...`.
Settings: `UseFlag FlagNoInc FlagOppMs FlagRatio FlagContempt FlagTimePct` in `config_engine_block.yml`;
set `UseFlag: 0` to turn it off. It does not change the bench.

## Speed

The net is 2x wider than W512_sf and has threat inputs, so nodes/s is about half of G's. The +34 Elo above already
includes that cost, at 8+0.08. At very short time controls (bullet) the gain may be smaller.

This exe is not PGO-built (G's exe is, worth about +20% nps). For a PGO build on the bot PC (Git Bash, LLVM installed):

```
cd engine
ACC=1024 EXTRA=-DNNUE_LNN6 ./build_pgo_win.sh ../weights/L1024T_lc0.nnue
```

That writes `bin/quirky_pgo.exe`. It gives the same bench node count; only nps changes.
