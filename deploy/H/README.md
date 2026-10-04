# Version H (provisional): install on the lichess-bot PC (Windows)

H = version G (same search settings, same Cerebellum book) + the new **Leela-trained threat net** `L1024T_lc0.nnue`
+ two search features that passed SPRT on G: `UseThreatHist` and `UseCuckoo`.

**Provisional.** The net is tested: +34.2 +- 16.6 Elo vs W512_sf on G's search (1071 games, 8+0.08, 1 thread).
The two features were accepted on the old net (W512_sf: ThreatHist +4.7, Cuckoo +3.9). Their re-test on this net is
still running, and the search constants have not been re-tuned (SPSA) for this net yet. Expect updates.

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
bench 13
```

It must print `bench: 339009 nodes`. Then do the book check from `deploy/G/README.md`.

## Speed

The net is 2x wider than W512_sf and has threat inputs, so nodes/s is about half of G's. The +34 Elo above already
includes that cost, at 8+0.08. At very short time controls (bullet) the gain may be smaller.

This exe is not PGO-built (G's exe is, worth about +20% nps). For a PGO build on the bot PC (Git Bash, LLVM installed):

```
cd engine
ACC=1024 EXTRA=-DNNUE_LNN6 ./build_pgo_win.sh ../weights/L1024T_lc0.nnue
```

That writes `bin/quirky_pgo.exe`. It gives the same bench node count; only nps changes.
