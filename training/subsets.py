"""
Categories of training positions, for nets specialised to one kind of position or game.

  open:FAM   opening family of the game, from its first plies (sfo source only; data/open/sfo.meta):
             e4e5, sicil (1.e4 c5), e4oth (other 1.e4), d4d5, d4nf6 (1.d4 Nf6), d4oth, flank (1.c4, 1.Nf3, rest)
  pcs:LO-HI  pieces on the board (incl. kings) in [LO, HI], e.g. pcs:2-16 (endgame-ish), pcs:17-32
  eco:X      ECO letter A..E (sfo only)
  all:       everything (with sfo: still without the val games) -- the control for a fine-tuned expert
restrict(source, spec) limits a Source's training draws to the category (its val part is untouched);
mask(records, meta, spec) gives the boolean mask for evaluation.
"""
import numpy as np
import torch

META = np.dtype([("eco", "<u2"), ("game", "<u4"), ("elo", "<u2"), ("tc", "<u2"), ("m", "<u2", (4,))])
SQ = {f + str(r): (r - 1) * 8 + "abcdefgh".index(f) for f in "abcdefgh" for r in range(1, 9)}
mv = lambda a, b: SQ[a] | SQ[b] << 6
E4, D4, E5, C5, D5, NF6 = mv("e2", "e4"), mv("d2", "d4"), mv("e7", "e5"), mv("c7", "c5"), mv("d7", "d5"), mv("g8", "f6")
FAMS = ["e4e5", "sicil", "e4oth", "d4d5", "d4nf6", "d4oth", "flank"]


def family(m):
    """(N,4) first-ply codes -> family index (see FAMS)"""
    w, b = m[:, 0], m[:, 1]
    f = np.full(len(m), 6)
    e4, d4 = w == E4, w == D4
    f[e4] = 2
    f[e4 & (b == E5)] = 0
    f[e4 & (b == C5)] = 1
    f[d4] = 5
    f[d4 & (b == D5)] = 3
    f[d4 & (b == NF6)] = 4
    return f


def load_meta(path):
    return np.memmap(path.replace(".bin", ".meta"), mode="r", dtype=META)


VAL_GAMES = 10_000  # the latest games of sfo are held out (eval_cats.py); ~400k positions


def val_gid(meta):
    return int(np.asarray(meta["game"][-3_000_000:]).max()) + 1 - VAL_GAMES


def mask(rec, meta, spec):
    kind, arg = spec.split(":")
    if kind == "all":
        return np.ones(len(rec), bool)
    if kind == "open":
        return family(np.asarray(meta["m"])) == FAMS.index(arg)
    if kind == "eco":
        return np.asarray(meta["eco"]) // 100 == "ABCDE".index(arg)
    if kind == "pcs":
        lo, hi = (int(x) for x in arg.split("-"))
        n = (np.asarray(rec["pc"]) != 0xFFFF).sum(1)
        return (n >= lo) & (n <= hi)
    raise ValueError(spec)


def restrict(src, spec, chunk=20_000_000):
    """keep only training records of the category: sets src.idx (int64 indices < n_train)"""
    is_sfo = src.path.endswith("sfo.bin")
    meta = load_meta(src.path) if is_sfo else None
    vg = val_gid(meta) if is_sfo else None
    parts = []
    for i in range(0, src.n_train, chunk):
        j = min(i + chunk, src.n_train)
        m = mask(src.data[i:j], None if meta is None else meta[i:j], spec)
        if is_sfo:
            m &= np.asarray(meta["game"][i:j]) < vg
        parts.append(np.flatnonzero(m).astype(np.int64) + i)
    src.idx = np.concatenate(parts)
    print(f"subset {spec} on {src.path.split('/')[-1]}: {len(src.idx):,} of {src.n_train:,} training records", flush=True)


def init_from(net, name):
    """copy a trained net (exp2/nets/NAME/last.pt, factorizer folded) into `net`: FT into every weight set
    (and every king bucket if the base has fewer), head if the shapes match"""
    import nnue4 as N
    s = torch.load(f"{N.X}/nets/{name}/last.pt", map_location="cpu", weights_only=False)
    cfg, sd = N.load_folded(s["ema"] or s["net"], s["cfg"])
    c = net.cfg
    assert cfg["featset"] == c["featset"] and cfg["acc"] == c["acc"], "base must have the same feature set and width"
    pb = N.per_bucket(c["featset"])
    src = sd["ft"][:-1].view(-1, cfg["nkb"], pb, c["acc"])[0]  # [nkb_base, pb, acc] (base has one set)
    if cfg["nkb"] != c["nkb"]:  # map each target bucket to the base bucket of a representative king square
        kt, kb = N.kb_table(c["nkb"]), N.kb_table(cfg["nkb"])
        src = torch.stack([src[kb[np.flatnonzero(kt == t)[0]]] for t in range(c["nkb"])])
    with torch.no_grad():
        ns = N.NSETS[c.get("sets", "none")]
        net.ft[:-1].copy_(src.unsqueeze(0).expand(ns, -1, -1, -1).reshape(-1, c["acc"]))
        if net.fac is not None:
            net.fac.zero_()
        net.ftb.copy_(sd["ftb"])
        heads = {k: v for k, v in sd.items() if not k.startswith(("ft", "psqt", "fac"))}
        own = net.state_dict()
        if all(k in own and own[k].shape == v.shape for k, v in heads.items()):
            net.load_state_dict({**own, **heads})
            print(f"init from {name}: FT x{ns} sets + head", flush=True)
        else:
            print(f"init from {name}: FT x{ns} sets (head shapes differ, fresh head)", flush=True)
