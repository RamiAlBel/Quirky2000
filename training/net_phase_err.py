"""
Static-eval error of several nets against independent reference evals, as a function of game phase.

  python net_phase_err.py OUT_DIR

Reference sets (REC4 records, quiet positions: not in check, best move not a capture/promotion):
  lc0 : held-out Leela test80 Jul 2024 binpacks (training used Jan-Jun), cp = 0.39 x binpack score, has ply
  sf  : Lichess eval DB, deepest Stockfish analysis (depth >= MIN_DEPTH), no ply (phase by piece count only)
Each net's engine cp is rescaled by one least-squares factor per (net, set) so unit conventions do not count as error.
Error bars: 95% bootstrap CI, resampling whole games for lc0 (positions of one game are correlated).
"""
import os
import subprocess
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from extract_v4 import REC4

X = "/scratch/ralbe/chess_nnue/exp2"
NETS = [  # label, binary, EvalFile
    ("W512_sf (yours)", f"{X}/bin/Hbase", "/scratch/ralbe/chess_nnue/Quirky2000/weights/W512_sf.nnue"),
    ("lc0 no-threats", f"{X}/bin/H6q", f"{X}/nets/h/w2_nothr-200.nnue"),
    ("lc0 threats", f"{X}/bin/H6q", f"{X}/nets/h/w2_thr-200.nnue"),
]
LC0 = [f"{X}/data/heldout/test80-jul2024/training-run1-test80-20240710-0417.no-db.rec4",
       f"{X}/data/heldout/test80-jul2024/training-run1-test80-20240723-1417.no-db.rec4"]
EDB, EDB_DEPTH, N_EDB, MIN_DEPTH = f"{X}/data/evaldb/edb.bin", f"{X}/data/evaldb/edb.depth", 150000, 30
CPMAX, NBOOT = 1000, 300
PIECE_BINS = [(29, 32), (25, 28), (21, 24), (17, 20), (13, 16), (9, 12), (2, 8)]
PLY_BINS = [(8, 19), (20, 39), (40, 59), (60, 79), (80, 99), (100, 139), (140, 255)]
SYM = "PNBRQKpnbrqk"


def fen_of(r):
    board = ["."] * 64
    for c in r["pc"]:
        if c != 0xFFFF:
            board[c % 64] = SYM[c // 64]
    rows = []
    for rank in range(7, -1, -1):
        s, e = "", 0
        for f in range(8):
            p = board[rank * 8 + f]
            if p == ".":
                e += 1
            else:
                s += (str(e) if e else "") + p
                e = 0
        rows.append(s + (str(e) if e else ""))
    return "/".join(rows) + (" w" if r["stm"] else " b") + " - - 0 1"


def load_sets():
    parts = [np.fromfile(f, dtype=REC4) for f in LC0]
    lc0 = np.concatenate(parts)
    # records keep game order: a new game starts where ply goes down (or at a file boundary)
    newg = np.r_[True, np.diff(lc0["ply"].astype(int)) < 0]
    newg[np.cumsum([len(p) for p in parts])[:-1]] = True
    gid = np.cumsum(newg)
    edb = np.memmap(EDB, dtype=REC4, mode="r")
    dep = np.memmap(EDB_DEPTH, dtype="u1", mode="r")
    rng = np.random.default_rng(1)
    cand = rng.choice(len(edb), N_EDB * 4, replace=False)
    cand = np.sort(cand[dep[cand] >= MIN_DEPTH][:N_EDB])
    e = np.array(edb[cand])
    return {"lc0": (lc0, gid), "sf": (e, np.arange(len(e)))}


def run_net(binary, net, fens):
    cmds = ["uci", f"setoption name EvalFile value {net}"]
    for f in fens:
        cmds += [f"position fen {f}", "eval"]
    cmds.append("quit")
    out = subprocess.run([binary], input="\n".join(cmds) + "\n", capture_output=True, text=True).stdout
    cps = [int(l.rsplit("cp(quant)", 1)[1]) for l in out.splitlines() if "cp(quant)" in l]
    assert len(cps) == len(fens), (binary, net, len(cps), len(fens))
    return np.array(cps, dtype=np.float64)


def boot_ci(vals, groups, rng):
    """mean and 95% CI of vals, bootstrap over groups"""
    u, inv = np.unique(groups, return_inverse=True)
    s, n = np.bincount(inv, vals), np.bincount(inv)
    m = s.sum() / n.sum()
    idx = rng.integers(0, len(u), (NBOOT, len(u)))
    bm = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return m, lo, hi


def wp(cp):
    return 1 / (1 + 10 ** (-cp / 400))


def main(out):
    os.makedirs(out, exist_ok=True)
    sets = load_sets()
    res = {}
    for sname, (recs, gid) in sets.items():
        cache = f"{out}/{sname}_evals.npz"
        if os.path.exists(cache):
            z = np.load(cache)
            evals = {k: z[k] for k in z.files}
        else:
            fens = [fen_of(r) for r in recs]
            evals = {lab: run_net(b, n, fens) for lab, b, n in NETS}
            np.savez(cache, **evals)
        ref = recs["cp"].astype(np.float64)
        pieces = (recs["pc"] != 0xFFFF).sum(1)
        res[sname] = dict(ref=ref, pieces=pieces, ply=recs["ply"].astype(int), gid=gid, evals=evals)
    rng = np.random.default_rng(7)
    lines = []
    for sname, d in res.items():
        ok = np.abs(d["ref"]) < CPMAX
        d["err"], d["werr"], d["scale"] = {}, {}, {}
        for lab, ev in d["evals"].items():
            m = ok & (np.abs(ev) < 2 * CPMAX)
            a = (ev[m] * d["ref"][m]).sum() / (ev[m] ** 2).sum()
            d["scale"][lab] = a
            d["err"][lab] = np.abs(np.clip(a * ev, -CPMAX, CPMAX) - d["ref"])
            d["werr"][lab] = 100 * np.abs(wp(a * ev) - wp(d["ref"]))
            raw = np.abs(np.clip(ev, -CPMAX, CPMAX) - d["ref"])[ok].mean()
            lines.append(f"{sname:4s} {lab:16s} n={ok.sum():6d}  scale {a:.3f}  cp-MAE fitted {d['err'][lab][ok].mean():6.1f}"
                         f"  raw {raw:6.1f}  WP-err {d['werr'][lab].mean():5.2f}%")
        d["ok"] = ok
    tables = {}
    for sname, d in res.items():
        for axis, bins, key in [("pieces", PIECE_BINS, "pieces"), ("ply", PLY_BINS, "ply")]:
            if axis == "ply" and sname == "sf":
                continue
            rows = []
            base = list(d["evals"])[0]
            for lo, hi in bins:
                sel = d["ok"] & (d[key] >= lo) & (d[key] <= hi)
                row = {"bin": f"{lo}-{hi}", "n": int(sel.sum())}
                for lab in d["evals"]:
                    row[lab] = boot_ci(d["err"][lab][sel], d["gid"][sel], rng)
                    row["wp " + lab] = boot_ci(d["werr"][lab][sel], d["gid"][sel], rng)
                    row["d " + lab] = boot_ci(d["err"][lab][sel] - d["err"][base][sel], d["gid"][sel], rng)
                rows.append(row)
            tables[(sname, axis)] = rows
    with open(f"{out}/summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
        for (sname, axis), rows in tables.items():
            f.write(f"\n[{sname} by {axis}] cp-MAE (95% CI); delta vs {NETS[0][0]}\n")
            for r in rows:
                f.write(f"{r['bin']:>8s} n={r['n']:6d} " + "  ".join(
                    f"{lab.split()[0]}{'-' + lab.split()[1] if lab.startswith('lc0') else ''}: {r[lab][0]:5.1f} [{r[lab][1]:5.1f},{r[lab][2]:5.1f}]"
                    for lab, _, _ in NETS) + "  | " + "  ".join(
                    f"d {lab}: {r['d ' + lab][0]:+5.1f} [{r['d ' + lab][1]:+.1f},{r['d ' + lab][2]:+.1f}]" for lab, _, _ in NETS[1:]) + "\n")
    np.save(f"{out}/tables.npy", {f"{k[0]}|{k[1]}": v for k, v in tables.items()}, allow_pickle=True)
    print(open(f"{out}/summary.txt").read())


if __name__ == "__main__":
    main(sys.argv[1])
