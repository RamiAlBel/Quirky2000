"""
Label the eval-less July positions with a teacher net: python label_teacher.py TEACHER_NAME
Reads exp2/nets/TEACHER/last.pt (EMA weights if present), writes data/jul/noeval_teacher.i16 (int16 centipawns,
side to move, clamped +-1500, same order as noeval.bin) and prints the teacher's val MSE on the SF-labelled
July val slice as a sanity check.
"""
import os, sys, time
import numpy as np
import torch
import nnue4 as N

name = sys.argv[1]
X = N.X
dev = torch.device("cuda")
s = torch.load(f"{X}/nets/{name}/last.pt", map_location="cpu", weights_only=False)
net = N.Net4(**s["cfg"])
net.load_state_dict(s["ema"] or s["net"])
net = net.to(dev).eval()
cfg = s["cfg"]
scale = cfg["cp_scale"]
data = np.memmap(f"{N.JUL}/noeval.bin", mode="r", dtype=N.REC4)
dst = np.memmap(f"{N.JUL}/noeval_teacher.i16.tmp", mode="w+", dtype="<i2", shape=(len(data),))
t0 = time.time()
B = 1 << 16
with torch.no_grad():
    for i in range(0, len(data), B):
        r = np.array(data[i:i + B])
        us, them, n = (torch.from_numpy(x).to(dev) for x in N.features(r["pc"], r["stm"], cfg["featset"], cfg["nkb"]))
        cp = (net(us, them, n) * scale).clamp(-1500, 1500).round().short().cpu().numpy()
        dst[i:i + len(r)] = cp
        if (i // B) % 200 == 0:
            print(f"{i/1e6:.0f}M / {len(data)/1e6:.0f}M  ({(time.time()-t0)/60:.1f} min)", flush=True)
dst.flush()
del dst
os.replace(f"{N.JUL}/noeval_teacher.i16.tmp", f"{N.JUL}/noeval_teacher.i16")
# sanity: teacher vs Stockfish on the SF val slice
sf = N.Source(f"{N.JUL}/sf.bin", rec4=True)
r = np.array(sf.data[sf.n_train:sf.n_train + 200_000])
us, them, n = (torch.from_numpy(x).to(dev) for x in N.features(r["pc"], r["stm"], cfg["featset"], cfg["nkb"]))
with torch.no_grad():
    pred = torch.tanh(net(us, them, n) * scale / 400).cpu().numpy()
print(f"DONE teacher {name}: labelled {len(data):,} positions; July SF val MSE {np.mean((pred - np.tanh(r['cp'] / 400)) ** 2):.5f}")
