"""
Structured pruning of feature-transformer lanes: python prune.py SRC DST WIDTH [--data sf:1]
Importance of accumulator lane j = (std of its clipped activation over real positions) x (L2 norm of the L1
weights reading it, summed over both perspective halves and all buckets). The top WIDTH lanes are kept (FT
columns, bias and the matching L1 input columns); psqt is unchanged. The result is saved as nets/DST/last.pt
(train_v4-style checkpoint, same head as SRC) — run stage2.py DST DST_s2 afterwards to fit the small head.
Only crelu/screlu FT activations (pairwise nets pair lane j with j+ACC/2; not supported here).
"""
import argparse, os
import numpy as np
import torch
import nnue4 as N

p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("dst"); p.add_argument("width", type=int)
p.add_argument("--data", default="sf:1"); p.add_argument("--n", type=int, default=1_000_000)
args = p.parse_args()
X = N.X

s = torch.load(f"{X}/nets/{args.src}/last.pt", map_location="cpu", weights_only=False)
cfg = s["cfg"]
assert cfg["ft_act"] != "pair", "pairwise nets not supported"
net = N.Net4(**cfg)
net.load_state_dict(s["ema"] or s["net"])
acc, W = cfg["acc"], args.width
assert W < acc and W % 64 == 0

# activation statistics on real positions
src = N.SOURCES[args.data.split(":")[0]]()
idx = np.sort(np.random.default_rng(0).integers(0, src.n_train, args.n))
r = np.array(src.data[idx])
us, them, n = (torch.from_numpy(x) for x in N.features(r["pc"], r["stm"], cfg["featset"], cfg["nkb"]))
with torch.no_grad():
    bag = lambda i: torch.nn.functional.embedding_bag(i, net.ft, mode="sum", padding_idx=net.pad) + net.ftb
    a = torch.cat([torch.clamp(bag(us[k:k + 65536]), 0, 1) for k in range(0, len(us), 65536)])
    b = torch.cat([torch.clamp(bag(them[k:k + 65536]), 0, 1) for k in range(0, len(them), 65536)])
    act_std = (torch.cat([a, b]).std(0))                                   # (acc,)
    w = net.l1.w.detach()                                                  # (nb, h1, 2*acc)
    wnorm = (w[:, :, :acc].pow(2).sum((0, 1)) + w[:, :, acc:].pow(2).sum((0, 1))).sqrt()
    imp = act_std * wnorm
keep = torch.sort(torch.argsort(imp, descending=True)[:W]).values
dead = int((act_std < 1e-3).sum())
print(f"{args.src}: {acc} lanes, {dead} dead (std<1e-3); keeping {W}; kept importance share "
      f"{imp[keep].sum() / imp.sum():.3f}", flush=True)

cfg2 = dict(cfg, acc=W)
small = N.Net4(**cfg2)
with torch.no_grad():
    small.ft.copy_(net.ft[:, keep]); small.ftb.copy_(net.ftb[keep])
    if net.psqt is not None:
        small.psqt.copy_(net.psqt)
    cols = torch.cat([keep, keep + acc])
    small.l1.w.copy_(net.l1.w[:, :, cols]); small.l1.b.copy_(net.l1.b)
    for name in ("l2", "out"):
        la, lb = getattr(net, name), getattr(small, name)
        if la is not None:
            lb.w.copy_(la.w); lb.b.copy_(la.b)
os.makedirs(f"{X}/nets/{args.dst}", exist_ok=True)
torch.save(dict(net=small.state_dict(), ema=None, cfg=cfg2, args=dict(s["args"], acc=W, pruned_from=args.src),
                hist=[], epoch=0), f"{X}/nets/{args.dst}/last.pt")
print(f"DONE pruned {args.src} -> {args.dst} ({W} lanes)")
