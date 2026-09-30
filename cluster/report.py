#!/usr/bin/env python3
"""writes runs/RESULTS.md: ladder matches, champion, training status + val_mse of all 9 nets."""
import json, math, os
X = "/scratch/ralbe/chess_nnue/exp2"
st = json.load(open(f"{X}/runs/ladder.json")) if os.path.exists(f"{X}/runs/ladder.json") else {"matches": [], "champ": None, "out": []}
def elo(w, l, d):
    n = w + l + d; s = (w + d / 2) / n if n else .5
    return -400 * math.log10(1 / s - 1) if 0 < s < 1 else float("nan")
def err(w, l, d):
    n = w + l + d
    if not n: return float("nan")
    s = (w + d / 2) / n; v = (w + d / 4) / n - s * s if n else 0  # per-game variance of score
    return 1.96 * 400 / math.log(10) * math.sqrt(max(v, 1e-9) / n) / (s * (1 - s))
L = ["# Width x source sweep - results", "", f"**Current champion: {st['champ']}**  (eliminated: {', '.join(st['out']) or '-'})", "",
     "## Matches (TC 60+0.5, 1 thread; winner stays on)", "", "| A | B | games | A +W =D -L | Elo(A) +/- | winner |", "|---|---|---|---|---|---|"]
for m in st["matches"]:
    w, l, d = m["a_wins"], m["b_wins"], m["draws"]
    L.append(f"| {m['a']} | {m['b']} | {w+l+d} | +{w} ={d} -{l} | {elo(w,l,d):+.0f} +/- {err(w,l,d):.0f} | {m['winner']} |")
L += ["", "## Nets (val_mse on the Stockfish-labelled validation set, lower = better; not a strength measure)", "",
      "| net | epochs done | final val_mse (after stage 2) |", "|---|---|---|"]
for src in ("sf", "edb", "lc0"):
    for w in (512, 1024, 4096):
        n = f"W{w}_{src}"; ep = "-"; v = "-"
        try: ep = json.load(open(f"{X}/nets/{n}/history.json"))["hist"][-1]["epoch"]
        except Exception: pass
        try: v = "%.5f" % json.load(open(f"{X}/nets/{n}_s2/history.json"))["best"]
        except Exception: pass
        L.append(f"| {n} | {ep}/20 | {v} |")
open(f"{X}/runs/RESULTS.md", "w").write("\n".join(L) + "\n")
