"""
Output-bucket schemes on a frozen feature transformer: refit the t25p-style head (h1 8, dual) with a different
rule for picking the head, same budget for every scheme, and compare quantized val MSE (overall and per phase).

  schemes: pc8 (current: piece count, 8), pc16, pcq8 (4 piece-count bands x queens on/off),
           npm8 (non-pawn material, 8 bands), pcq16 (8 piece-count bands x queens on/off)
usage: python buckets.py NET SCHEME [--pos 20000000] [--epochs 2] [--data sf:3.5,old:1]
"""
import argparse, json, os, time
import numpy as np
import torch
import nnue4 as N

p = argparse.ArgumentParser()
p.add_argument("net"); p.add_argument("scheme")
p.add_argument("--pos", type=int, default=20_000_000)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--data", default="sf:3.5,old:1")
args = p.parse_args()
X = "/scratch/ralbe/chess_nnue/exp2"
log = lambda *a: print(*a, flush=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

NB = {"pc8": 8, "pc16": 16, "pcq8": 8, "npm8": 8, "pcq16": 16}[args.scheme]
NPM = torch.tensor([0, 3, 3, 5, 9, 0] * 2, dtype=torch.float32)  # per piece type (colour*6+type)


def bucket(pc):
    """pc: int64 [B,32] piece codes (0xFFFF pad) -> bucket index"""
    valid = pc != 0xFFFF
    n = valid.sum(1)
    code = torch.where(valid, pc // 64, torch.zeros_like(pc))
    queens = ((code % 6 == 4) & valid).any(1).long()
    if args.scheme == "pc8":
        return (n - 1) * 8 // 32
    if args.scheme == "pc16":
        return ((n - 1) * 16 // 32).clamp(0, 15)
    if args.scheme == "pcq8":
        return ((n - 1) * 4 // 32).clamp(0, 3) * 2 + queens
    if args.scheme == "pcq16":
        return ((n - 1) * 8 // 32).clamp(0, 7) * 2 + queens
    if args.scheme == "npm8":  # 0..62 non-pawn material (both sides), 8 bands
        m = torch.where(valid, NPM[code], torch.zeros_like(code, dtype=torch.float32)).sum(1)
        return (m * 8 / 63).long().clamp(0, 7)
    raise ValueError(args.scheme)


s = torch.load(f"{X}/nets/{args.net}/last.pt", map_location="cpu", weights_only=False)
cfg = dict(s["cfg"])
base = N.Net4(**cfg); base.load_state_dict(s["ema"] or s["net"])
net = N.Net4(**{**cfg, "nb": NB})
with torch.no_grad():
    net.ft.copy_(base.ft); net.ftb.copy_(base.ftb)
    for dst, src in ((net.l1, base.l1), (net.out, base.out)):  # warm start every bucket from the nearest old one
        for b in range(NB):
            dst.w[b].copy_(src.w[min(b * cfg["nb"] // NB, cfg["nb"] - 1)]); dst.b[b].copy_(src.b[min(b * cfg["nb"] // NB, cfg["nb"] - 1)])
net = net.to(dev)
net.ft.requires_grad_(False); net.ftb.requires_grad_(False)

# the head's own bucket choice replaces bucket_of(n): monkey-patch through a per-batch override
_cur = {}
orig_bucket_of = N.bucket_of
N.bucket_of = lambda n, nb: _cur["bk"] if "bk" in _cur else orig_bucket_of(n, nb)

spec = [(k, float(w)) for k, w in (x.split(":") for x in args.data.split(","))]
srcs = [N.SOURCES[k]() for k, _ in spec]
wts = np.array([w for _, w in spec]) / sum(w for _, w in spec)
vs = N.SOURCES["old"]()
vidx = np.arange(vs.n_train, vs.n_train + vs.val_n)


def make(src, idx):
    r = np.array(src.data[idx])
    us, them, n = N.features(r["pc"], r["stm"], cfg["featset"], cfg["nkb"])
    return (torch.from_numpy(us), torch.from_numpy(them), torch.from_numpy(n),
            torch.from_numpy(r["cp"].astype(np.float32)), bucket(torch.from_numpy(r["pc"].astype(np.int64))))


def val():
    net.eval(); net.fq = True
    tot, per = 0.0, np.zeros((4, 2))
    with torch.no_grad():
        for i in range(0, len(vidx), 16384):
            us, them, n, cp, bk = (t.to(dev) for t in make(vs, vidx[i:i + 16384]))
            _cur["bk"] = bk
            e = (torch.tanh(net(us, them, n) * (cfg["cp_scale"] / N.REF_SCALE)) - torch.tanh(cp / N.REF_SCALE)) ** 2
            tot += e.sum().item()
            band = torch.bucketize(n, torch.tensor([9, 17, 25], device=dev))  # 2-8 / 9-16 / 17-24 / 25-32
            for k in range(4):
                per[k] += (e[band == k].sum().item(), (band == k).sum().item())
    net.fq = False; net.train(); _cur.pop("bk", None)
    return tot / len(vidx), (per[:, 0] / np.maximum(per[:, 1], 1)).round(5).tolist()


head = [q for k, q in net.named_parameters() if not k.startswith(("ft", "psqt"))]
opt = torch.optim.Adam(head, lr=args.lr)
steps = args.pos // 16384
total = steps * args.epochs
rng = np.random.default_rng(1)
log(f"{args.scheme}: nb {NB}, {args.epochs} x {args.pos / 1e6:.0f}M, data {spec}, start val {val()}")
for ep in range(args.epochs):
    t0 = time.time()
    for st in range(steps):
        g = ep * steps + st
        for grp in opt.param_groups:
            grp["lr"] = float(args.lr * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * g / total))))
        parts = [make(src, np.sort(rng.integers(0, src.n_train, c))) for src, c in zip(srcs, rng.multinomial(16384, wts)) if c]
        us, them, n, cp, bk = (torch.cat(z).to(dev) for z in zip(*parts))
        _cur["bk"] = bk
        loss = (torch.tanh(net(us, them, n)) - torch.tanh(cp / cfg["cp_scale"])).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); net.clip()
    _cur.pop("bk", None)
    v = val()
    log(f"  epoch {ep + 1}: val_mse(q) {v[0]:.5f}  by pieces 2-8/9-16/17-24/25-32 {v[1]}  ({(time.time() - t0) / 60:.1f} min)")
out = f"{X}/nets/buckets"
os.makedirs(out, exist_ok=True)
json.dump(dict(scheme=args.scheme, nb=NB, val=v[0], bands=v[1], args=vars(args)), open(f"{out}/{args.scheme}.json", "w"))
torch.save(dict(net=net.state_dict(), cfg={**cfg, "nb": NB}, scheme=args.scheme), f"{out}/{args.scheme}.pt")
log(f"DONE {args.scheme} {v[0]:.5f}")
