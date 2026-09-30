"""Fit E[result] ~ tanh(cp / k) on a REC4 file (records with results only); prints k.
Used to put lc0 binpack scores on the same cp scale as the Lichess/Stockfish data (same eval -> same expected score).
usage: python calib_scale.py file.bin [n_sample]"""
import sys
import numpy as np
from extract_v4 import REC4

d = np.memmap(sys.argv[1], mode="r", dtype=REC4)
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
rng = np.random.default_rng(0)
ix = np.sort(rng.integers(0, len(d), min(n, len(d))))
s = d[ix]
cp = s["cp"].astype(np.float64); r = s["result"].astype(np.float64)
m = np.abs(cp) < 1500
cp, r = cp[m], r[m]
ks = np.arange(50, 1500, 5.0)
err = [np.mean((np.tanh(cp / k) - r) ** 2) for k in ks]
k = ks[int(np.argmin(err))]
print(f"{sys.argv[1]}: n={len(cp)} best k={k:.0f} mse={min(err):.4f}  |cp| median {np.median(np.abs(cp)):.0f}  "
      f"draw rate {np.mean(r == 0):.3f}  mean ply {s['ply'].mean():.0f}")
