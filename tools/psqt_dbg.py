import sys, numpy as np, torch
sys.path.insert(0, "/scratch/ralbe/chess_nnue/exp2/training")
import nnue4 as N
dev = torch.device("cuda")
src = N.SOURCES["old"]()
data = N.Data([src], [1], "halfkp", 32)
for tag, psqt, plr in [("nopsqt", 0, 1.0), ("psqt", 1, 1.0), ("psqt_lr0.1", 1, 0.1)]:
    torch.manual_seed(0)
    net = N.Net4(psqt=psqt, h1=16, h2=32, nb=1 if False else 8, hid_act="crelu", cp_scale=400).to(dev)
    groups = [{"params": [q for k, q in net.named_parameters() if k != "psqt"], "lr": 1e-3}]
    if psqt:
        groups.append({"params": [net.psqt], "lr": 1e-3 * plr})
    opt = torch.optim.Adam(groups)
    losses = []
    for step, b in enumerate(data.batches(600, 16384, seed=1)):
        us, them, n, cp, res, has = (t.to(dev) for t in b)
        loss = (torch.tanh(net(us, them, n)) - torch.tanh(cp / 400)).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); net.clip()
        losses.append(loss.item())
    extra = ""
    if psqt:
        w = net.psqt.detach()
        extra = f" psqt |w| mean {w.abs().mean():.3f} max {w.abs().max():.2f}"
    print(f"{tag:12s} loss first100 {np.mean(losses[:100]):.4f} last100 {np.mean(losses[-100:]):.4f}{extra}", flush=True)
