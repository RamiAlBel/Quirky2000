# Version G: install on the lichess-bot PC (Windows)

G = version E's search settings + the `W512_sf` net (that pair is version F) + the **Cerebellum** opening book.
Nothing else changed from E.

## What is in this folder

| file | what |
|---|---|
| `quirky_G.exe` | the engine, Windows x64, needs an AVX2 + BMI2 CPU. Since 2026-10-02 the faster PGO build (see below), built on Windows with LLVM clang by `engine/build_pgo_win.sh` from `engine/` in this repo |
| `W512_sf.nnue` | the net (same file as `weights/W512_sf.nnue`) |
| `config_engine_block.yml` | the lichess-bot `engine:` settings for G |

**Not included: the Cerebellum book** (178 MB, over GitHub's file limit, and licensed CC BY-NC-SA 4.0).
You download it yourself (step 2).

## Install

1. Copy this folder (`deploy/G`) to the bot PC, for example to `...\Native NNUE Chess\G\`.
2. **Download Cerebellum Light (3Merge, 2020-09-16)** from the BrainFish / Cerebellum download page
   (zip `Cerebellum_Light_3Merge_200916`, which also holds a README and the licence). Unzip it and put **`Cerebellum3Merge.bin` in the same folder as
   `quirky_G.exe`**. Keep that file name, or change `BookFile` in the config to match.
   Check that you have the same file (PowerShell, in that folder):
   ```
   Get-FileHash Cerebellum3Merge.bin
   ```
   It must be `5B2F1794C20F095E23364A966F682B1C235AFD46EC1F1ACBC6E4BF72F1F92BFE` (177,973,984 bytes).
3. Back up `lichess-bot\config.yml`. In its `engine:` block, set `dir`, `name`, `working_dir` and the `uci_options`
   list as in `config_engine_block.yml`. Put your own folder path in `dir` and `working_dir`. Keep everything else
   (`ponder: true`, `online_egtb`, lichess-bot's own book off).
4. Start `Lichess Bot.bat` as usual.

The engine loads the book when `BookFile` is set. That takes about a second and needs about 180 MB of RAM on top of the hash.

## Check it before going live

In a `cmd` window in the G folder, start `quirky_G.exe` and type (or paste):

```
setoption name EvalFile value W512_sf.nnue
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
bench 13
```

1. **Bench.** It must print `bench: 378755 nodes` (nps depends on the PC). A different number means the exe or the net is not the right one.
2. **Book.** Then type:
   ```
   setoption name BookFile value Cerebellum3Merge.bin
   setoption name UseBook value 1
   setoption name BookBest value 1
   setoption name BookDepth value 255
   isready
   position startpos moves e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6
   go wtime 60000 btime 60000
   ```
   It must print `info string book Cerebellum3Merge.bin loaded`, then `readyok`, and then answer at once with
   `bestmove c1e3` (the book move). If it prints `ERROR could not load book`, the file is not next to the exe or has a
   different name.
3. Type `quit`.

## How the book is used (the same way it was tested)

- Every move, the engine looks up the current position in the book. With `BookBest=1` it plays the move with the
  highest weight. Cerebellum gives weight 255 to its best move and 127 to alternatives.
- With `BookDepth=255` it keeps using the book as long as the game stays in it. Once a position is not in the book, it
  searches normally. If the game comes back into the book later (a transposition), it uses the book again.
- It does not use the book while pondering, so pondering starts after the book ends.
- It works for both colours and does not depend on how the opponent opened. The book covers positions, not move orders.

## How much it is worth

All results are self-play at 6+0.06 with 1 thread, and every engine used the same net (`experiments/06-opening-books.md`):
Cerebellum vs our old book (the one E used) **+102 +-15 Elo** (SPRT), Cerebellum vs no book about +63. Against human-like
Lichess opponents the gain may be different.

F vs E (the net change alone) at PC-equivalent blitz/rapid: `experiments/results/README.md`.

## Faster build since 2026-10-02 (`quirky_G.exe` is now this build)

`quirky_G.exe` was replaced by a faster build of the same engine: same options, same net, same book, nothing to
change in `config.yml`. **The bot runs this build since 2026-10-02.** The first G exe (cross-compiled with zig) is in git history.
The bench check above prints the same `bench: 378755 nodes` (the search is identical; only nps changes).

What changed:
- **PGO** (profile-guided optimisation, LLVM clang): the compiler is given a profile of a real search and lays out the
  code for it. No change to the search. Build: `engine/build_pgo_win.sh` (Git Bash, LLVM in `C:\Program Files\LLVM`).
- **TT prefetch** (`engine/search.cpp`): right after a move (or null move) is made, the child's hash-table cluster is
  prefetched, so the probe at the start of the child node waits less on memory. No change to the search
  (`-DNO_TT_PREFETCH` turns it off).

Speed, Ryzen 7 5800X, version G options, 6 positions that were not in the PGO training run, `go depth 17` (1 thread)
or `go depth 20` (4 threads); 1-thread node counts identical in every build:

| build | 1 thread, 512 MB hash | 4 threads, 4 GB hash |
|---|---|---|
| plain clang build (same source as `quirky_G.exe`) | 1.23M nps | 3.76M nps |
| + TT prefetch | +1% | +2% |
| PGO | +18% | +23% |
| **PGO + TT prefetch (`quirky_G.exe` now)** | **+21%** | **+27%** |

Cache misses on the hash table turned out to be a small cost: the prefetch only adds 1-3%. The gain is PGO.

### Is speed worth Elo? Speed-odds match (stopped early)

Quirky G against itself at 10+0.1, 1 thread each, 64 MB hash, no book; random 6-ply openings within +-80 cp, each
played with both colours. The "fast" side gets 2x the clock and increment, which is what a 2x faster engine would get.

| fast side gets | games | score | Elo of the fast side |
|---|---|---|---|
| 2x time | 90 | 70.0% | **+147** (roughly +80 .. +230) |

Stopped by request after 90 games, so the error bar is wide; a 1.25x run (about what PGO gives) was planned but not
played. Taken at face value, speed is worth a lot for Quirky at blitz: if Elo scales with log(speed), +21-27% speed is
about +40-50 Elo in self-play. That is an estimate from the 2x result, not a measurement.

### Tried and rejected: skipping zero inputs in the first layer

Stockfish-style sparse L1 (only multiply the groups of 4 L1 inputs that are not all zero; `-DSPARSE_L1`, off by
default, bit-identical results). With `W512_sf` 40% of the L1 input bytes are non-zero, but 85% of the 4-byte groups
contain a non-zero byte, so almost nothing is skipped: the sparse eval took 125 ns vs 92 ns dense, -5 to -10% nps.
Reordering neurons by activity would only bring the groups to about 78% (estimate). It would need a net trained
for sparse activations. `-DL1_STATS` prints the sparsity and writes per-neuron activity to `l1_idx.txt`.
