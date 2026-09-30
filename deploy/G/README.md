# Version G: install on the lichess-bot PC (Windows)

G = version E's search settings + the `W512_sf` net (that pair is version F) + the **Cerebellum** opening book.
Nothing else changed from E.

## What is in this folder

| file | what |
|---|---|
| `quirky_G.exe` | the engine, Windows x64, needs an AVX2 + BMI2 CPU. Cross-compiled on Linux with zig from `engine/` in this repo |
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
