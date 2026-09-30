"""
Per-position strata for stratified sampling (train_v4 --strata):
  python strata.py phase            -> <source>.phase  uint8 band of piece count (0: 2-8, 1: 9-16, 2: 17-24, 3: 25-32)
  python strata.py hard NET [bins]  -> <source>.hard   uint8 error-quantile bin of NET's squared error (0 = easiest)
                                       + <source>.hard.json with each bin's mean error
Written for the "old" and "sf" sources, next to their .bin files.
"""
import json, sys, time
import numpy as np
import torch
import nnue4 as N

mode = sys.argv[1]
CH = 1 << 20


def path(src_name):
    return {"old": N.OLD_DATA, "sf": f"{N.JUL}/sf.bin"}[src_name]


for name in ("old", "sf"):
    src = N.SOURCES[name]()
    n = len(src.data)
    t0 = time.time()
    if mode == "phase":
        out = np.memmap(path(name) + ".phase", mode="w+", dtype="u1", shape=(n,))
        for i in range(0, n, CH):
            npc = (np.array(src.data[i:i + CH]["pc"]) != 0xFFFF).sum(1)
            out[i:i + len(npc)] = np.digitize(npc, [9, 17, 25])
        out.flush()
        print(f"{name}: phase strata for {n:,} positions ({(time.time()-t0)/60:.1f} min)", flush=True)
    else:
        net_name = sys.argv[2]
        bins = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        dev = torch.device("cuda")
        s = torch.load(f"{N.X}/nets/{net_name}/last.pt", map_location="cpu", weights_only=False)
        cfg = s["cfg"]
        net = N.Net4(**cfg); net.load_state_dict(s["ema"] or s["net"]); net = net.to(dev).eval()
        err = np.empty(n, np.float32)
        with torch.no_grad():
            for i in range(0, n, CH):
                r = np.array(src.data[i:i + CH])
                us, them, pcs = (torch.from_numpy(x).to(dev) for x in N.features(r["pc"], r["stm"], cfg["featset"], cfg["nkb"]))
                p = torch.tanh(net(us, them, pcs) * cfg["cp_scale"] / 400).cpu().numpy()
                err[i:i + len(r)] = (p - np.tanh(r["cp"] / 400)) ** 2
        edges = np.quantile(err[:src.n_train], np.linspace(0, 1, bins + 1)[1:-1])
        b = np.digitize(err, edges).astype("u1")
        out = np.memmap(path(name) + ".hard", mode="w+", dtype="u1", shape=(n,))
        out[:] = b; out.flush()
        means = [float(err[:src.n_train][b[:src.n_train] == k].mean()) for k in range(bins)]
        json.dump(dict(net=net_name, bins=bins, mean_err=means), open(path(name) + ".hard.json", "w"), indent=1)
        print(f"{name}: hard strata from {net_name}: bin mean errors {np.round(means, 4)} ({(time.time()-t0)/60:.1f} min)", flush=True)
print("DONE")
