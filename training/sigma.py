"""
Uncertainty head ("sigma") on a frozen net: predicts how wrong the net's static eval is, in cp.

  target e = min(|eval_net(pos) - cp_label(pos)|, 1000)   eval_net = the engine-like quantized eval of NET
  sigma head: same shape as the eval head (piece-count buckets, L1 on the frozen FT activations, dual) -> 1
  sigma_cp = 100 * softplus(out)    loss = MSE(sigma_cp/100, e/100)

Kill criterion (printed at the end): val MSE of sigma vs a baseline table mean(e | pieces, |eval| bin) fitted on
training data. sigma is only worth wiring into the search if it clearly beats that table.

usage: python sigma.py NET_DIR_NAME OUT_NAME [--data edb:1] [--h1 8] [--pos 40000000] [--epochs 3]
Writes exp2/nets/OUT/{OUT.sig (SIG1), sigma.pt, report.json}.
"""
import argparse, json, os, struct, time
import numpy as np
import torch
import nnue4 as N

p = argparse.ArgumentParser()
p.add_argument("net"); p.add_argument("out")
p.add_argument("--data", default="edb:1")
p.add_argument("--h1", type=int, default=8)
p.add_argument("--nb", type=int, default=8)
p.add_argument("--pos", type=int, default=40_000_000)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--emax", type=float, default=1000)
p.add_argument("--tap", type=int, default=0, help="1: sigma reads the eval head's hidden layer (no extra L1, ~free)")
args = p.parse_args()
X = "/scratch/ralbe/chess_nnue/exp2"
log = lambda *a: print(*a, flush=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s = torch.load(f"{X}/nets/{args.net}/last.pt", map_location="cpu", weights_only=False)
cfg = s["cfg"]
net = N.Net4(**cfg)
net.load_state_dict(s["ema"] or s["net"])
net = net.to(dev).eval()
net.fq = True
for q in net.parameters():
    q.requires_grad_(False)


def ft_input(us, them):
    bag = lambda idx, w: torch.nn.functional.embedding_bag(idx, w, mode="sum", padding_idx=net.pad)
    a, b = bag(us, net.ft) + net.ftb, bag(them, net.ft) + net.ftb

    def act(x):
        x = torch.clamp(x, 0.0, 1.0)
        if cfg["ft_act"] == "screlu":
            return x * x
        if cfg["ft_act"] == "pair":
            h = x.shape[1] // 2
            return x[:, :h] * x[:, h:]
        return x
    x = torch.cat([act(a), act(b)], 1)
    return torch.round(x * 127) / 127  # engine feeds uint8 [0,127]


def eval_hidden(x, n):
    """eval head hidden activations (engine-like quantized L1), the input of the tap sigma"""
    bk = N.bucket_of(n, cfg["nb"])
    w = net.l1.w
    s_ = 127 / w.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    h = torch.clamp(net.l1(x, bk, torch.round(w * s_) / s_), 0.0, 1.0)
    return torch.cat([h, h * h], 1)


class SigmaTap(torch.nn.Module):
    def __init__(self, hin, nb, h=8):
        super().__init__()
        self.l1 = N.Bucketed(nb, hin + 2, h)
        self.out = N.Bucketed(nb, 2 * h, 1)
        self.nb = nb

    def forward(self, x, n, ev):
        bk = N.bucket_of(n, self.nb)
        a = torch.clamp(ev.abs() / 400, max=4.0).unsqueeze(1)
        z = torch.clamp(self.l1(torch.cat([eval_hidden(x, n), a, a * a], 1), bk), 0.0, 1.0)
        return torch.nn.functional.softplus(self.out(torch.cat([z, z * z], 1), bk).squeeze(1))


class Sigma(torch.nn.Module):
    def __init__(self, l1_in, h1, nb):
        super().__init__()
        self.l1 = N.Bucketed(nb, l1_in, h1)
        self.out = N.Bucketed(nb, 2 * h1 + 2, 1)  # + |eval|/400 and its square (clipped to 4)
        self.nb = nb

    def forward(self, x, n, ev):
        bk = N.bucket_of(n, self.nb)
        h = torch.clamp(self.l1(x, bk), 0.0, 1.0)
        a = torch.clamp(ev.abs() / 400, max=4.0).unsqueeze(1)
        return torch.nn.functional.softplus(self.out(torch.cat([h, h * h, a, a * a], 1), bk).squeeze(1))  # sigma / 100 cp


spec = [(k, float(w)) for k, w in (x.split(":") for x in args.data.split(","))]
srcs = [N.SOURCES[k]() for k, _ in spec]
data = N.Data(srcs, [w for _, w in spec], cfg["featset"], cfg["nkb"])
vs = srcs[0]
val = data.make(vs, np.arange(vs.n_train, vs.n_train + vs.val_n))
l1_in = cfg["acc"] if cfg["ft_act"] == "pair" else 2 * cfg["acc"]
sig = (SigmaTap(2 * cfg["h1"], args.nb) if args.tap else Sigma(l1_in, args.h1, args.nb)).to(dev)
out = f"{X}/nets/{args.out}"
os.makedirs(out, exist_ok=True)
log(f"sigma on {args.net} ({cfg}) data {spec}, val {vs.val_n} of {os.path.basename(vs.path)}")


@torch.no_grad()
def batch_targets(us, them, n, cp):
    ev = net(us, them, n) * cfg["cp_scale"]
    e = torch.clamp((ev - cp).abs(), max=args.emax) / 100
    return ft_input(us, them), ev, e


# baseline table: mean e by (pieces 2..32, |eval| bin of 50 cp up to 750)
TAB = torch.zeros(33, 16, device=dev, dtype=torch.float64)
CNT = torch.zeros(33, 16, device=dev, dtype=torch.float64)
tab_idx = lambda n, ev: (n.long().clamp(0, 32), (ev.abs() / 50).long().clamp(max=15))

opt = torch.optim.Adam(sig.parameters(), lr=args.lr)
steps = args.pos // 16384
total = steps * args.epochs
for ep in range(args.epochs):
    t0 = time.time()
    run = 0.0
    for step, b in enumerate(data.batches(steps, 16384, seed=77 + ep, loaders=10)):
        g = ep * steps + step
        for grp in opt.param_groups:
            grp["lr"] = float(args.lr * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * g / total))))
        us, them, n, cp = (t.to(dev, non_blocking=True) for t in b[:4])
        x, ev, e = batch_targets(us, them, n, cp)
        if ep == 0:
            i, j = tab_idx(n, ev)
            TAB.index_put_((i, j), e.double(), accumulate=True)
            CNT.index_put_((i, j), torch.ones_like(e, dtype=torch.float64), accumulate=True)
        loss = (sig(x, n, ev) - e).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            sig.l1.w.clamp_(-N.L1_CLIP, N.L1_CLIP)
        run += loss.item()
    log(f"epoch {ep + 1}/{args.epochs}  train mse {run / steps:.4f}  ({(time.time() - t0) / 60:.1f} min)")

# ---- kill criterion on held-out data ----
tab = (TAB / CNT.clamp_min(1)).float()
glob = (TAB.sum() / CNT.sum()).float()
tab = torch.where(CNT > 50, tab, glob)
P, B, E, NP, EV = [], [], [], [], []
with torch.no_grad():
    for i in range(0, len(val[0]), 16384):
        us, them, n, cp = (t[i:i + 16384].to(dev) for t in val[:4])
        x, ev, e = batch_targets(us, them, n, cp)
        P.append(sig(x, n, ev)); ti, tj = tab_idx(n, ev); B.append(tab[ti, tj]); E.append(e); NP.append(n); EV.append(ev)
P, B, E, NP, EV = (torch.cat(t).float() for t in (P, B, E, NP, EV))


def rank(t):
    r = torch.empty_like(t); r[t.argsort()] = torch.arange(len(t), device=t.device, dtype=t.dtype); return r


spear = lambda a, b: float(torch.corrcoef(torch.stack([rank(a), rank(b)]))[0, 1])
mse = lambda a: float((a - E).pow(2).mean())
rep = dict(val_n=len(E), mean_err_cp=float(E.mean() * 100), mse_const=float(E.var()), mse_table=mse(B), mse_sigma=mse(P),
           spearman_table=spear(B, E), spearman_sigma=spear(P, E))
rep["gain_vs_table_pct"] = 100 * (1 - rep["mse_sigma"] / rep["mse_table"])
# decile calibration: mean predicted vs mean actual error per sigma decile
q = torch.quantile(P[:200000], torch.linspace(0, 1, 11, device=dev))
rep["deciles"] = [[round(float(P[(P >= q[k]) & (P <= q[k + 1])].mean() * 100), 1),
                   round(float(E[(P >= q[k]) & (P <= q[k + 1])].mean() * 100), 1)] for k in range(10)]
# within a fixed piece-count / |eval| cell, does sigma still separate small from large errors?
cell = (NP >= 20) & (NP <= 26) & (EV.abs() < 100)
if cell.sum() > 1000:
    rep["cell_20-26pcs_eval<100_spearman_sigma"] = spear(P[cell], E[cell])
log(json.dumps(rep, indent=1))
json.dump(rep, open(f"{out}/report.json", "w"), indent=1)
torch.save(dict(sig=sig.state_dict(), args=vars(args), net=args.net), f"{out}/sigma.pt")

if args.tap:  # SIG2: "SIG2" | i32 nb, h, hin | f32 l1.w[nb][h][hin+2], l1.b[nb][h], out.w[nb][2h], out.b[nb]
    with open(f"{out}/{args.out}.sig", "wb") as f:
        f.write(b"SIG2" + struct.pack("<3i", args.nb, 8, 2 * cfg["h1"]))
        for t in (sig.l1.w, sig.l1.b, sig.out.w, sig.out.b):
            f.write(t.detach().cpu().float().numpy().astype("<f4").tobytes())
    log(f"DONE -> {out}/{args.out}.sig (tap)")
    raise SystemExit
# ---- export SIG1: "SIG1" | i32 nb, h1, l1_in | f32 l1.w[nb][h1][l1_in], l1.b[nb][h1], out.w[nb][2*h1+2], out.b[nb]
with open(f"{out}/{args.out}.sig", "wb") as f:
    f.write(b"SIG1" + struct.pack("<3i", args.nb, args.h1, l1_in))
    for t in (sig.l1.w, sig.l1.b, sig.out.w, sig.out.b):
        f.write(t.detach().cpu().float().numpy().astype("<f4").tobytes())
# a few reference values for the engine cross-check
with torch.no_grad():
    us, them, n = (t[:8].to(dev) for t in val[:3])
    ev = net(us, them, n) * cfg["cp_scale"]
    ref = dict(sigma_cp=(sig(ft_input(us, them), n, ev) * 100).tolist(), eval_cp=ev.tolist())
json.dump(ref, open(f"{out}/ref.json", "w"))
log(f"DONE -> {out}/{args.out}.sig")
