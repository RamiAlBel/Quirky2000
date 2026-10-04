#!/usr/bin/env python3
"""float reference forward pass of a bullet quirky raw.bin, from a feature dump (QK_MODE=dump / engine `features`).
usage: ref6.py raw.bin dump.txt  -> prints 400*out per FEN"""
import sys
import numpy as np
NPSQ, NTHR, NOB, L1, L2, L3 = 12288, 59808, 8, 1024, 32, 32
raw = np.fromfile(sys.argv[1], dtype="<f4")
shapes = [("psqt", L1, NPSQ), ("fac", L1, 768), ("thrw", L1, NTHR), ("thrb", L1, 1), ("w1", NOB * L2, L1),
          ("b1", NOB * L2, 1), ("w2", NOB * L3, 2 * L2), ("b2", NOB * L3, 1), ("w3", NOB, L3), ("b3", NOB, 1)]
W, o = {}, 0
for n, r, c in shapes:
    W[n] = raw[o:o + r * c].reshape(c, r).T; o += r * c
lines = [l.split() for l in open(sys.argv[2])]
fen = None
for l in lines:
    if l[0] == "FEN": fen = " ".join(l[1:]); continue
    idx = [int(x) for x in l[1:]]
    acc = W["thrb"][:, 0].astype(np.float64).copy()
    for i in idx:
        acc += W["psqt"][:, i] + W["fac"][:, i % 768] if i < NPSQ else W["thrw"][:, i - NPSQ]
    a = np.clip(acc, 0, 1); h = a[:L1 // 2] * a[L1 // 2:]
    if l[0] == "STM": stm = h; npieces = sum(1 for i in idx if i < NPSQ); continue
    x = np.concatenate([stm, h]); b = (npieces - 2) // 4
    y = W["w1"][b * L2:(b + 1) * L2] @ x + W["b1"][b * L2:(b + 1) * L2, 0]
    hid = np.clip(np.concatenate([y, y * y]), 0, 1)
    z = np.clip(W["w2"][b * L3:(b + 1) * L3] @ hid + W["b2"][b * L3:(b + 1) * L3, 0], 0, 1)
    out = W["w3"][b] @ z + W["b3"][b, 0]
    print(f"REF {fen} | {400 * out:.3f}")
