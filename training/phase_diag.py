# Where are the nets weak? val MSE by piece-count band and by |eval|, plus training-data mix per band.
import sys, numpy as np, torch
import nnue4 as N
dev = torch.device("cuda")
old = N.SOURCES["old"](); jul = N.SOURCES["sf"]()
r = np.array(old.data[old.n_train:old.n_train + old.val_n])
npc = (r["pc"] != 0xFFFF).sum(1)
bands = [(2, 8), (9, 16), (17, 24), (25, 32)]
tgt = np.tanh(r["cp"] / 400)
print("band      share(val)  share(July)")
jn = (np.array(jul.data[:2_000_000])["pc"] != 0xFFFF).sum(1)
for lo, hi in bands:
    print(f"{lo:2d}-{hi:2d} pcs  {np.mean((npc>=lo)&(npc<=hi)):9.3f}  {np.mean((jn>=lo)&(jn<=hi)):9.3f}")
for name in sys.argv[1:]:
    s = torch.load(f"{N.X}/nets/{name}/last.pt", map_location="cpu", weights_only=False)
    net = N.Net4(**s["cfg"]); net.load_state_dict(s["ema"] or s["net"]); net = net.to(dev).eval(); net.fq = True
    us, them, n = N.features(r["pc"], r["stm"], s["cfg"]["featset"], s["cfg"]["nkb"])
    preds = []
    with torch.no_grad():
        for i in range(0, len(r), 65536):
            t = [torch.from_numpy(x[i:i + 65536]).to(dev) for x in (us, them, n)]
            preds.append(torch.tanh(net(*t) * s["cfg"]["cp_scale"] / 400).cpu().numpy())
    e = (np.concatenate(preds) - tgt) ** 2
    ab = np.abs(r["cp"])
    line = f"{name:18s} all {e.mean():.5f} | " + " ".join(f"{lo}-{hi}:{e[(npc>=lo)&(npc<=hi)].mean():.5f}" for lo, hi in bands)
    line += " | |cp|<100:%.5f 100-400:%.5f >400:%.5f" % (e[ab < 100].mean(), e[(ab >= 100) & (ab < 400)].mean(), e[ab >= 400].mean())
    print(line, flush=True)
