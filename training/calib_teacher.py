"""Scale for an REC4 file's cp labels relative to a reference net (C1, trained on Stockfish cp):
fit tanh(s * cp / 400) ~ tanh(net / 400) over a sample; prints s and sign agreement.
usage: python calib_teacher.py file.bin [net_dir] [n]"""
import sys
import numpy as np
import torch
import nnue4 as N

path = sys.argv[1]
nd = sys.argv[2] if len(sys.argv) > 2 else "/scratch/ralbe/chess_nnue/exp2/nets/C_hkb8pair_sfold_s2"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 500_000
ck = torch.load(f"{nd}/last.pt", map_location="cpu", weights_only=False)
net = N.Net4(**ck["cfg"]); net.load_state_dict(ck["net"]); net.eval()
src = N.Source(path, rec4=True)
d = N.Data([src], [1], ck["cfg"]["featset"], ck["cfg"]["nkb"])
ix = np.sort(np.random.default_rng(0).integers(0, len(src.data), n))
preds, cps = [], []
with torch.no_grad():
    for i in range(0, n, 32768):
        us, them, nn, cp, _, _ = d.make(src, ix[i:i + 32768])
        preds.append(net(us, them, nn).numpy().ravel() * ck["cfg"]["cp_scale"]); cps.append(cp.numpy())
p, c = np.concatenate(preds), np.concatenate(cps)
tp = np.tanh(p / 400)
ss = np.arange(0.05, 3.0, 0.01)
err = [np.mean((np.tanh(s * c / 400) - tp) ** 2) for s in ss]
s = ss[int(np.argmin(err))]
m = (np.abs(p) > 50) & (c != 0)
print(f"{path}: best scale {s:.2f} (tanh-mse {min(err):.4f}; at 1.0: {err[int(round((1.0-0.05)/0.01))]:.4f})  "
      f"sign agreement {np.mean(np.sign(p[m]) == np.sign(c[m])):.3f}  corr {np.corrcoef(tp, np.tanh(c/400))[0,1]:.3f}  "
      f"median |cp| {np.median(np.abs(c)):.0f} vs net {np.median(np.abs(p)):.0f}")
