"""
Train a full NNUE variant (see nnue4.py) from scratch; every experiment knob is a flag.

  --data "sf:1,teacher:0.5"   mixture of sources (old = the 100M Aug positions, sf = July Stockfish-labelled,
                              teacher = July eval-less positions labelled by a teacher net)
  --wdl L                     target = (1-L) tanh(cp/scale) + L * game result (sources without results use L=0)
  --loss_pow P                mean |tanh(out) - target|^P
  --ema D                     export/validate an exponential moving average of the weights (0 = off)
Budget: --epochs x --epoch_pos samples (default 10 x 100M, as acc512). Val: last 1M of the old data,
MSE of tanh(cp/400) with the engine-like quantized L1 (comparable across every variant).
Outputs: exp2/nets/<name>/{last.pt, <name>.nnue (LNN4), history.json}
"""
import argparse
import copy
import json
import os
import time
import numpy as np
import torch
import nnue4 as N

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--data", default="old:1")
p.add_argument("--acc", type=int, default=512)
p.add_argument("--features", default="halfkp", choices=N.FEATSETS)
p.add_argument("--kb", type=int, default=32, choices=[8, 16, 32])
p.add_argument("--psqt", type=int, default=0)
p.add_argument("--ft_act", default="crelu", choices=N.FT_ACTS)
p.add_argument("--h1", type=int, default=8)
p.add_argument("--h2", type=int, default=0)
p.add_argument("--nb", type=int, default=8)
p.add_argument("--hid_act", default="dual", choices=N.HID_ACTS)
p.add_argument("--cp_scale", type=float, default=350)
p.add_argument("--wdl", type=float, default=0.0)
p.add_argument("--strata", default="", help="phase:TAU or hard:ALPHA (needs strata.py files)")
p.add_argument("--ft8", type=int, default=0, help="train FT weights on the int8 grid (engine -DFT_INT8)")
p.add_argument("--loss_pow", type=float, default=2.0)
p.add_argument("--ema", type=float, default=0.0)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--epoch_pos", type=int, default=100_000_000)
p.add_argument("--batch", type=int, default=16384)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--lr_end", type=float, default=3e-5)
p.add_argument("--qat_last", type=int, default=0, help="fake-quantize L1 in the last N epochs")
p.add_argument("--max_minutes", type=float, default=1e9, help="stop after the epoch that would overrun (resume later)")
p.add_argument("--loaders", type=int, default=8)
args = p.parse_args()

X = "/scratch/ralbe/chess_nnue/exp2"
OUT = f"{X}/nets/{args.name}"
os.makedirs(OUT, exist_ok=True)
SOURCES = N.SOURCES
log = lambda *a: print(*a, flush=True)


def main():
    t_start = time.time()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    spec = [(k, float(w)) for k, w in (x.split(":") for x in args.data.split(","))]
    srcs = [SOURCES[k]() for k, _ in spec]
    val_src = srcs[[k for k, _ in spec].index("old")] if "old" in [k for k, _ in spec] else SOURCES["old"]()
    strata = (args.strata.split(":")[0], float(args.strata.split(":")[1])) if args.strata else None
    data = N.Data(srcs, [w for _, w in spec], args.features, args.kb, strata=strata)
    data.val = data.make(val_src, np.arange(val_src.n_train, val_src.n_train + val_src.val_n))
    net = N.Net4(acc=args.acc, featset=args.features, nkb=args.kb, psqt=args.psqt, h1=args.h1, h2=args.h2,
                 nb=args.nb, ft_act=args.ft_act, hid_act=args.hid_act, cp_scale=args.cp_scale, ft8=args.ft8).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, fused=True)
    ema = copy.deepcopy(net) if args.ema else None
    steps = args.epoch_pos // args.batch
    total = steps * args.epochs
    start_ep, hist = 0, []
    ck = f"{OUT}/last.pt"
    if os.path.exists(ck):
        s = torch.load(ck, map_location="cpu", weights_only=False)
        net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"]); start_ep = s["epoch"]; hist = s["hist"]
        if ema is not None and s.get("ema"):
            ema.load_state_dict(s["ema"])
        log(f"resumed after epoch {start_ep}")
    log(f"{args.name}: {net.cfg}  data {spec}  wdl {args.wdl} pow {args.loss_pow} ema {args.ema}  "
        f"{sum(q.numel() for q in net.parameters())/1e6:.1f}M params  GPU {torch.cuda.get_device_name(0)}")

    @torch.no_grad()
    def evaluate(m):
        m.eval(); m.fq = True
        tot = 0.0
        for i in range(0, len(data.val[0]), 16384):
            us, them, n, cp = (t[i:i + 16384].to(dev) for t in data.val[:4])
            pred = torch.tanh(m(us, them, n) * (m.cfg["cp_scale"] / N.REF_SCALE))
            tot += torch.nn.functional.mse_loss(pred, torch.tanh(cp / N.REF_SCALE), reduction="sum").item()
        m.fq = False; m.train()
        return tot / len(data.val[0])

    restarts = 0
    while True:
      return_to_start = False
      for ep in range(start_ep, args.epochs):
          t0 = time.time()
          net.fq = ep >= args.epochs - args.qat_last
          run, rn = 0.0, 0
          for step, b in enumerate(data.batches(steps, args.batch, seed=ep, loaders=args.loaders)):
              g = ep * steps + step
              for grp in opt.param_groups:
                  grp["lr"] = float(args.lr_end + 0.5 * (args.lr - args.lr_end) * (1 + np.cos(np.pi * g / total)))
              us, them, n, cp, res, has = (t.to(dev, non_blocking=True) for t in b)
              target = torch.tanh(cp / args.cp_scale)
              if args.wdl:
                  lam = args.wdl * has
                  target = (1 - lam) * target + lam * res
              err = (torch.tanh(net(us, them, n)) - target).abs()
              loss = err.pow(args.loss_pow).mean()
              opt.zero_grad(set_to_none=True)
              loss.backward()
              opt.step()
              net.clip()
              if ema is not None:
                  with torch.no_grad():
                      for pe, pn in zip(ema.parameters(), net.parameters()):
                          pe.lerp_(pn, 1 - args.ema)
              if step % 100 == 0:
                  run += loss.item(); rn += 1
          m = ema if ema is not None else net
          v = evaluate(m)
          h = dict(epoch=ep + 1, train_loss=run / max(rn, 1), val_mse=v, minutes=(time.time() - t0) / 60)
          hist.append(h)
          log(f"epoch {ep+1}/{args.epochs}  loss {h['train_loss']:.5f}  val_mse(q) {v:.5f}  ({h['minutes']:.1f} min)")
          if ep == 0 and v > 0.2 and restarts < 3:  # dead init (all hidden units stuck): start over with new weights
              restarts += 1
              log(f"dead net after epoch 1 (val {v:.3f}); restart {restarts} with a fresh initialization")
              torch.manual_seed(1000 + restarts)
              net = N.Net4(**net.cfg).to(dev)
              opt = torch.optim.Adam(net.parameters(), lr=args.lr, fused=True)
              ema = copy.deepcopy(net) if args.ema else None
              hist = []
              return_to_start = True
              break
          torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), ema=ema.state_dict() if ema else None,
                          epoch=ep + 1, hist=hist, cfg=net.cfg, args=vars(args)), ck + ".tmp")
          os.replace(ck + ".tmp", ck)
          N.export_lnn4(m, f"{OUT}/{args.name}.nnue")
          json.dump(dict(args=vars(args), cfg=net.cfg, hist=hist), open(f"{OUT}/history.json", "w"), indent=1)
          el = (time.time() - t_start) / 60
          if ep + 1 < args.epochs and el + h["minutes"] * 1.2 > args.max_minutes:
              log(f"stopping for time after epoch {ep+1} ({el:.0f} min); resume with the same command")
              raise SystemExit(3)
      if not return_to_start:
        break
    log("DONE")


if __name__ == "__main__":
    main()
