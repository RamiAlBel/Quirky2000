#!/usr/bin/env python3
"""Score of engine 'new' vs 'base' from fastchess PGNs: pgnscore.py runs/X*/games.pgn ..."""
import math, re, sys
for f in sys.argv[1:]:
    txt = open(f, errors="replace").read()
    w = d = l = 0
    for wh, bl, res in re.findall(r'\[White "(\w+)"\]\s*\[Black "(\w+)"\]\s*\[Result "([^"]+)"\]', txt):
        if res == "1/2-1/2": d += 1
        elif res in ("1-0", "0-1"):
            if (res == "1-0") == (wh == "new"): w += 1
            else: l += 1
    n = w + d + l
    if not n: print(f, "no games"); continue
    s = (w + d / 2) / n
    var = (w * (1 - s) ** 2 + l * s ** 2 + d * (0.5 - s) ** 2) / n
    elo = lambda p: -400 * math.log10(1 / min(max(p, 1e-6), 1 - 1e-6) - 1)
    e, se = elo(s), 1.96 * math.sqrt(var / n)
    print(f"{f.split('/')[-2]:28s} n={n:5d} W{w} D{d} L{l}  {e:+6.1f} [{elo(s - se):+6.1f},{elo(s + se):+6.1f}]")
