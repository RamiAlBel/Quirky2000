"""Summarise the E vs F PC-budget matches: python pc_report.py -> markdown tables (from runs/pc_{blitz,rapid}, logs/pc_*)."""
import glob, math, re, os
X = "/scratch/ralbe/chess_nnue/exp2"
GAME = re.compile(r'\[White "([^"]+)"\]\s*\[Black "([^"]+)"\]\s*\[Result "(1-0|0-1|1/2-1/2)"\]')

def elo(w, d, l):
    n = w + d + l
    if n == 0: return "", ""
    s = (w + d / 2) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    e = lambda p: -400 * math.log10(1 / p - 1) if 0 < p < 1 else float("inf") * (1 if p >= 1 else -1)
    se = math.sqrt(var / n)
    return f"{e(s):+.0f}", f"{(e(min(s + 1.96 * se, .999)) - e(max(s - 1.96 * se, .001))) / 2:.0f}"

med = lambda v: f"{sorted(v)[len(v) // 2] / 1e6:.0f}M" if v else ""

for mode, pc in (("blitz", "3+2"), ("rapid", "10+5")):
    print(f"\n### {mode.capitalize()} (PC {pc})\n")
    print("| A | B | games | A +W =D -L | score | Elo(A) +- 95% | cluster clock (range) | median nodes/move E / F (PC target E ~" + ("65M" if mode == "blitz" else "200M") + ") |")
    print("|---|---|---|---|---|---|---|---|")
    for f in ("F512", "F1024", "F4096"):
        w = d = l = 0; tcs = []; nodes = {"E": [], f: []}
        for pg in sorted(glob.glob(f"{X}/runs/pc_{mode}/E_vs_{f}_o*/games.pgn")):
            txt = open(pg).read()
            for g in GAME.findall(txt):
                r = {"1-0": 1, "0-1": 0, "1/2-1/2": .5}[g[2]]
                r = r if g[0] == "E" else 1 - r
                w += r == 1; d += r == .5; l += r == 0
            for game in txt.split("[Event ")[1:]:
                side = [re.search(r'\[White "([^"]+)"\]', game).group(1), re.search(r'\[Black "([^"]+)"\]', game).group(1)]
                for i, m in enumerate(re.findall(r"\{[^}]*?n=(\d+)", game)):
                    nodes[side[i % 2]].append(int(m))
        for lg in glob.glob(f"{X}/logs/pc_{mode}_E_vs_{f}_o*.out"):
            m = re.search(r"tc=([\d.]+)\+([\d.]+)", open(lg).read())
            if m: tcs.append((float(m.group(1)), float(m.group(2))))
        n = w + d + l
        e, ci = elo(w, d, l)
        tc = f"{min(t[0] for t in tcs):.0f}-{max(t[0] for t in tcs):.0f} s + {min(t[1] for t in tcs):.1f}-{max(t[1] for t in tcs):.1f}" if tcs else ""
        print(f"| E | {f} | {n} | +{w} ={d} -{l} | {(w + d / 2) / max(n, 1):.0%} | {e} +- {ci} | {tc} | {med(nodes['E'])} / {med(nodes[f])} |")
