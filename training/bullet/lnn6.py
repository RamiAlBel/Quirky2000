#!/usr/bin/env python3
"""bullet raw.bin (examples/quirky, all f32, column-major) -> engine LNN6 net.

usage: lnn6.py CKPT_DIR_or_raw.bin OUT.nnue [--l1 1024 --l2 32 --l3 32 --dual 1]

LNN6 layout (little endian):
  "LNN6" i32 version=2, l1, npsq, nthr, l2, l3, nob, dual, zero pad to 64 bytes
  (v2: every block below starts 64-byte aligned, the engine mmaps psq/thr; v1 had no padding)
  i16 ftb[l1]                       FT bias, Q0=255
  i16 psq[npsq][l1]                 psqt + factoriser, Q0=255
  i8  thr[nthr][l1]                 threat rows, Q0=255 (trainer clips to +-127/255)
  i8  w1[nob][l1/4][l2][4]          L1, chunked for the sparse affine, scale s1[b][o] = 127/max|w1[b][o]| per output
  f32 s1[nob][l2]
  f32 b1[nob][l2]
  f32 w2[nob][in2][l3]              in2 = 2*l2 if dual else l2 (input-major)
  f32 b2[nob][l3]
  f32 w3[nob][l3]
  f32 b3[nob]
"""
import argparse, os, struct
import numpy as np

NPSQ, NTHR, NOB, Q0, Q1 = 16 * 768, 59808, 8, 255, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--l1", type=int, default=1024)
    ap.add_argument("--l2", type=int, default=32)
    ap.add_argument("--l3", type=int, default=32)
    ap.add_argument("--dual", type=int, default=1)
    ap.add_argument("--nothr", action="store_true", help="net trained with QK_THREATS=0: zero the (untrained) threat rows")
    a = ap.parse_args()
    path = os.path.join(a.src, "raw.bin") if os.path.isdir(a.src) else a.src
    raw = np.fromfile(path, dtype="<f4")
    l1, l2, l3 = a.l1, a.l2, a.l3
    in2 = 2 * l2 if a.dual else l2
    # (name, rows, cols) in save order; bullet stores (rows=out, cols=in) column-major
    shapes = [("psqt", l1, NPSQ), ("fac", l1, 768), ("thrw", l1, NTHR), ("thrb", l1, 1),
              ("w1", NOB * l2, l1), ("b1", NOB * l2, 1), ("w2", NOB * l3, in2), ("b2", NOB * l3, 1),
              ("w3", NOB, l3), ("b3", NOB, 1)]
    need = sum(r * c for _, r, c in shapes)
    assert raw.size == need, f"{path}: {raw.size} floats, expected {need} (wrong --l1/--l2/--l3/--dual?)"
    W, off = {}, 0
    for name, r, c in shapes:
        W[name] = raw[off:off + r * c].reshape(c, r).T  # [out][in]
        off += r * c

    def q(x, s, lo, hi, dt, name):
        y = np.round(x * s)
        n = int(((y < lo) | (y > hi)).sum())
        if n:
            print(f"warning: {name}: {n} values clipped")
        return np.clip(y, lo, hi).astype(dt)

    psq = W["psqt"] + np.tile(W["fac"], (1, 16))                    # [l1][npsq]
    ftb = q(W["thrb"][:, 0], Q0, -32767, 32767, "<i2", "ftb")
    psq = q(psq.T, Q0, -32767, 32767, "<i2", "psq")                 # [npsq][l1]
    thr = q(W["thrw"].T * (0 if a.nothr else 1), Q0, -127, 127, "i1", "thr")  # [nthr][l1]
    w1f = W["w1"].reshape(NOB, l2, l1)
    s1 = (127.0 / np.maximum(np.abs(w1f).max(axis=2), 1e-9)).astype("<f4")  # [nob][l2]
    w1 = q(w1f * s1[:, :, None], 1, -127, 127, "i1", "w1")
    w1 = w1.reshape(NOB, l2, l1 // 4, 4).transpose(0, 2, 1, 3)
    b1 = W["b1"][:, 0].astype("<f4")
    w2 = W["w2"].reshape(NOB, l3, in2).transpose(0, 2, 1).astype("<f4")
    b2 = W["b2"][:, 0].astype("<f4")
    w3 = W["w3"].astype("<f4")
    b3 = W["b3"][:, 0].astype("<f4")
    with open(a.out, "wb") as f:
        f.write((b"LNN6" + struct.pack("<8i", 2, l1, NPSQ, NTHR, l2, l3, NOB, a.dual)).ljust(64, b"\0"))
        for arr in (ftb, psq, thr, np.ascontiguousarray(w1), s1, b1, np.ascontiguousarray(w2), b2, w3, b3):
            f.write(arr.tobytes())
    acc_max = np.abs(psq.astype(np.int64)).max()
    print(f"wrote {a.out}: l1={l1} l2={l2} l3={l3} dual={a.dual} psq|max|={acc_max} thr|max|={np.abs(thr).max()} s1 min/med/max={s1.min():.0f}/{np.median(s1):.0f}/{s1.max():.0f}")


if __name__ == "__main__":
    main()
