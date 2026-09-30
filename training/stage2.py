"""
Stage 2 of the t25p recipe for a train_v4 net: swap in a small head, fit it on the frozen feature transformer,
then fine-tune everything for one epoch at a low learning rate (the FT overfits beyond that).

usage: python stage2.py SRC_NAME DST_NAME [--h1 8 --h2 0 --nb 8 --hid_act dual --cp_scale 350] [--data old:1]
Reads exp2/nets/SRC/last.pt, writes exp2/nets/DST/{DST.nnue, last.pt, history.json}.
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import nnue4 as N
SOURCES = N.SOURCES

p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("dst")
p.add_argument("--h1", type=int, default=8); p.add_argument("--h2", type=int, default=0)
p.add_argument("--nb", type=int, default=8); p.add_argument("--hid_act", default="dual")
p.add_argument("--cp_scale", type=float, default=350)
p.add_argument("--data", default=None, help="default: the source run's data mix")
p.add_argument("--wdl", type=float, default=None)
p.add_argument("--head_epochs", type=int, default=3); p.add_argument("--head_pos", type=int, default=40_000_000)
p.add_argument("--head_lr", type=float, default=2e-3)
p.add_argument("--polish_pos", type=int, default=100_000_000); p.add_argument("--polish_lr", type=float, default=1e-4)
args = p.parse_args()
X = "/scratch/ralbe/chess_nnue/exp2"
log = lambda *a: print(*a, flush=True)


def main():
    dev = torch.device("cuda")
    s = torch.load(f"{X}/nets/{args.src}/last.pt", map_location="cpu", weights_only=False)
    src_args = s["args"]
    cfg, sd = N.load_folded(s["ema"] or s["net"], s["cfg"])  # factorizer folded into the FT
    cfg.update(h1=args.h1, h2=args.h2, nb=args.nb, hid_act=args.hid_act, cp_scale=args.cp_scale)
    net = N.Net4(**cfg)
    if s["cfg"].get("fact"):
        net.ft_clip = 2 * N.FT_CLIP  # folded rows reach +-2; clamping them to +-1 threw away the factorizer's gain
    with torch.no_grad():
        net.ft.copy_(sd["ft"]); net.ftb.copy_(sd["ftb"])
        if net.psqt is not None:
            net.psqt.copy_(sd["psqt"])
    net = net.to(dev)
    data_spec = args.data or src_args["data"]
    wdl = src_args.get("wdl", 0.0) if args.wdl is None else args.wdl
    spec = [(k, float(w)) for k, w in (x.split(":") for x in data_spec.split(","))]
    st = src_args.get("strata") or ""
    strata = (st.split(":")[0], float(st.split(":")[1])) if st else None  # keep stage 1's sampling
    srcs = [SOURCES[k]() for k, _ in spec]
    if src_args.get("subset"):
        import subsets
        for sc in srcs:
            subsets.restrict(sc, src_args["subset"])
    data = N.Data(srcs, [w for _, w in spec], cfg["featset"], cfg["nkb"], strata=strata, sets=cfg.get("sets", "none"))
    vs = SOURCES["old"]()
    data.val = data.make(vs, np.arange(vs.n_train, vs.n_train + vs.val_n))
    out = f"{X}/nets/{args.dst}"
    os.makedirs(out, exist_ok=True)
    log(f"stage2 {args.src} -> {args.dst}: {cfg}  data {spec} wdl {wdl}")

    @torch.no_grad()
    def evaluate():
        net.eval(); net.fq = True
        tot = 0.0
        for i in range(0, len(data.val[0]), 16384):
            us, them, n, cp = (t[i:i + 16384].to(dev) for t in data.val[:4])
            pred = torch.tanh(net(us, them, n) * (cfg["cp_scale"] / N.REF_SCALE))
            tot += torch.nn.functional.mse_loss(pred, torch.tanh(cp / N.REF_SCALE), reduction="sum").item()
        net.fq = False; net.train()
        return tot / len(data.val[0])

    def run(params, epochs, pos, lr, seed, tag, on_epoch=None):
        opt = torch.optim.Adam(params, lr=lr)
        steps = pos // 16384
        total = steps * epochs
        for ep in range(epochs):
            t0 = time.time()
            for step, b in enumerate(data.batches(steps, 16384, seed=seed + ep, loaders=10)):
                g = ep * steps + step
                for grp in opt.param_groups:
                    grp["lr"] = float(lr * 0.03 + 0.5 * (lr - lr * 0.03) * (1 + np.cos(np.pi * g / total)))
                us, them, n, cp, res, has = (t.to(dev, non_blocking=True) for t in b)
                target = torch.tanh(cp / cfg["cp_scale"])
                if wdl:
                    lam = wdl * has
                    target = (1 - lam) * target + lam * res
                loss = (torch.tanh(net(us, them, n)) - target).pow(2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); net.clip()
            v = evaluate()
            log(f"  {tag} epoch {ep+1}/{epochs}  val_mse(q) {v:.5f}  ({(time.time()-t0)/60:.1f} min)")
            if on_epoch:
                on_epoch(v)

    head = [q for k, q in net.named_parameters() if not k.startswith(("ft", "psqt"))]
    for q in (net.ft, net.ftb) + ((net.psqt,) if net.psqt is not None else ()):
        q.requires_grad_(False)
    run(head, args.head_epochs, args.head_pos, args.head_lr, 500, "head")
    for q in net.parameters():
        q.requires_grad_(True)
    best = {"v": evaluate(), "sd": copy.deepcopy(net.state_dict())}
    hist = [dict(stage="head", val_mse=best["v"])]

    def keep(v):
        hist.append(dict(stage="polish", val_mse=v))
        if v < best["v"]:
            best["v"], best["sd"] = v, copy.deepcopy(net.state_dict())
    run(list(net.parameters()), 1, args.polish_pos, args.polish_lr, 900, "polish", keep)
    net.load_state_dict(best["sd"])
    N.export_lnn4(net, f"{out}/{args.dst}.nnue")
    torch.save(dict(net=net.state_dict(), ema=None, cfg=cfg, args={**src_args, **vars(args), "data": data_spec, "wdl": wdl},
                    hist=hist, epoch=0), f"{out}/last.pt")
    json.dump(dict(src=args.src, cfg=cfg, hist=hist, best=best["v"]), open(f"{out}/history.json", "w"), indent=1)
    log(f"DONE {args.dst}: best val_mse(q) {best['v']:.5f}")


if __name__ == "__main__":
    main()
