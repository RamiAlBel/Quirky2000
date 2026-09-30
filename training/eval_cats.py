"""
Validation error of nets per category of position/game (round G: do specialised weights help anywhere?).

usage: python eval_cats.py OUT.json NET [NET ...]      NET = exp2/nets/<NET>/last.pt (factorizer folded, engine-like
       quantized L1, like train_v4's val). Held-out sets, never trained on by any G net:
  sf   last 1M records of data/jul/sf.bin (the sf source's own val part)
  sfo  games with id >= VAL_GID of data/open/sfo.bin (the latest July games; subsets.py keeps them out of training)
Metric: MSE of tanh(cp/400) per category; categories: phase (pieces), stm king bucket (8), stm colour, |cp| band,
and for sfo: opening family (first plies), ECO letter, mean rating, base time.
"""
import json
import sys
import numpy as np
import torch
import nnue4 as N
import subsets as S

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(name):
    s = torch.load(f"{N.X}/nets/{name}/last.pt", map_location="cpu", weights_only=False)
    cfg, sd = N.load_folded(s["ema"] or s["net"], s["cfg"])
    net = N.Net4(**cfg)
    net.load_state_dict(sd)
    net = net.to(dev).eval()
    net.fq = True
    return net


@torch.no_grad()
def sq_err(net, rec):
    c = net.cfg
    out = np.empty(len(rec), np.float32)
    for i in range(0, len(rec), 65536):
        r = rec[i:i + 65536]
        us, them, n = (torch.from_numpy(x).to(dev) for x in N.features(r["pc"], r["stm"], c["featset"], c["nkb"], c.get("sets", "none")))
        pred = torch.tanh(net(us, them, n) * (c["cp_scale"] / N.REF_SCALE))
        tgt = torch.tanh(torch.from_numpy(r["cp"].astype(np.float32)).to(dev) / N.REF_SCALE)
        out[i:i + 65536] = ((pred - tgt) ** 2).cpu().numpy()
    return out


def categories(rec, meta=None):
    pc = rec["pc"].astype(np.int64)
    valid = pc != 0xFFFF
    n = valid.sum(1)
    code, sq = pc >> 6, pc & 63
    stm = rec["stm"]
    ksq = np.where(valid & (code == np.where(stm == 1, 5, 11)[:, None]), sq, 0).max(1)
    ko = np.where(stm == 1, ksq, ksq ^ 56)
    kb = N.kb_table(8)[ko ^ np.where((ko & 7) >= 4, 7, 0)]
    cp = np.abs(rec["cp"].astype(np.int64))
    cats = {
        "phase": (np.digitize(n, [9, 17, 25]), ["2-8", "9-16", "17-24", "25-32"]),
        "king_bucket(stm)": (kb, [f"kb{i}" for i in range(8)]),
        "king_side(stm)": (np.where((ko & 7) >= 4, 1, 0) + 2 * (ko >> 3 >= 2), ["qside,back", "kside,back", "qside,up", "kside,up"]),
        "stm": (stm.astype(np.int64), ["black", "white"]),
        "|cp|": (np.digitize(cp, [50, 150, 400, 1000]), ["<50", "50-150", "150-400", "400-1000", ">=1000"]),
    }
    if meta is not None:
        cats["opening"] = (S.family(np.asarray(meta["m"])), S.FAMS)
        eco = np.asarray(meta["eco"]).astype(np.int64)
        cats["eco"] = (np.where(eco < 500, eco // 100, 5), ["A", "B", "C", "D", "E", "?"])
        cats["elo"] = (np.digitize(meta["elo"], [1400, 1800, 2200]), ["<1400", "1400-1799", "1800-2199", ">=2200"])
        cats["base_time"] = (np.digitize(meta["tc"], [120, 300, 900]), ["bullet<120s", "blitz<300s", "rapid<900s", "classical"])
    return cats


def sfo_val(max_rec=2_000_000):
    src = N.SOURCES["sfo"]()
    meta = S.load_meta(src.path)
    lo = len(src.data) - 3_000_000
    g = np.asarray(meta["game"][lo:])
    keep = np.flatnonzero(g >= S.val_gid(meta)) + lo
    keep = keep[:: max(1, len(keep) // max_rec)]
    return np.array(src.data[keep]), np.array(meta[keep])


def main():
    out, names = sys.argv[1], sys.argv[2:]
    sets = {}
    sf = N.SOURCES["sf"]()
    sets["sf"] = (np.array(sf.data[sf.n_train:]), None)
    import os
    info = f"{N.X}/data/open/sfo_info.json"
    if os.path.exists(info) and json.load(open(info))["done"]:  # extraction finished
        sets["sfo"] = sfo_val()
    res = {}
    for name in names:
        net = load(name)
        res[name] = {}
        for vs, (rec, meta) in sets.items():
            e = sq_err(net, rec)
            r = {"all": [float(e.mean()), len(e)]}
            for cname, (lab, labels) in categories(rec, meta).items():
                r[cname] = {labels[k]: [float(e[lab == k].mean()), int((lab == k).sum())] for k in range(len(labels)) if (lab == k).any()}
            res[name][vs] = r
        print(name, {vs: round(res[name][vs]["all"][0], 5) for vs in res[name]}, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
