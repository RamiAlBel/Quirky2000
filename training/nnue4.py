"""
Full NNUE (feature transformer + head) for the exp2 experiments, and the LNN4 export.

Feature sets (per perspective P, the side whose accumulator it is):
  halfkp  HalfKP: king square (flipped for black) x 10 non-king piece planes x 64 -> 40960
  hkb     HalfKA with king buckets and horizontal mirroring (Stockfish HalfKAv2_hm style):
          orient = flip ranks for black, then mirror files if P's king is on files e-h;
          bucket = KB[oriented king square] (nkb = 32, 16 or 8); 11 planes: own P N B R Q,
          opp P N B R Q, both kings -> nkb * 704 features
PSQT (optional): nb extra FT outputs per feature summed like the accumulator; output += (us - them) / 2
Weight sets (LNN5, `sets`): several copies of the whole FT, one chosen per position and perspective:
  phase2  pieces on board (incl. kings) >= 17 -> set 0, else 1       phase4  set = (pieces-1)*4//32 (8-piece bands)
  phase3  >= 25 / 17..24 / <= 16                                     color   set = the perspective's colour
  feature id = (set * nkb + bucket) * per_bucket + plane * 64 + sq; the engine refreshes both accumulators when the
  set changes (a capture crossing a band), through the refresh cache.
Factorizer (`fact`, training only): a shared (plane, square) table added to every (set, bucket) row, so rarely seen
  buckets/sets start from the common piece-square weights (Stockfish's "virtual features"); folded in at export.
Head: see head_lib (clipped-ReLU FT output, optional pairwise, h1 -> [h2] -> 1, output buckets by piece count).
"""
import os
import queue
import struct
import threading
import time
import numpy as np
import torch
from attacks_np import attacked_flags

ROOT = "/scratch/ralbe/chess_nnue"
OLD_DATA = f"{ROOT}/Native NNUE Chess/data/positions.bin"
REC = np.dtype([("pc", "<u2", (32,)), ("cp", "<i2"), ("stm", "u1"), ("ply", "u1")])
REC4 = np.dtype([("pc", "<u2", (32,)), ("cp", "<i2"), ("stm", "u1"), ("ply", "u1"), ("result", "i1"), ("pad", "u1")])
FT_CLIP = 1.0
L1_CLIP = 2.0
REF_SCALE = 400.0
FT_ACTS = ["crelu", "screlu", "pair"]
HID_ACTS = ["crelu", "screlu", "dual"]
FEATSETS = ["halfkp", "hkb", "thr"]  # thr = hkb x (attacked by the opponent or not)
SETS = ["none", "phase2", "phase4", "phase3", "color"]
NSETS = {"none": 1, "phase2": 2, "phase4": 4, "phase3": 3, "color": 2}


def set_of(sets, n, persp):
    """weight set of a position with n pieces (incl. kings), for perspective persp (0 white, 1 black)"""
    if sets == "phase2":
        return np.where(n >= 17, 0, 1)
    if sets == "phase4":
        return (n - 1) * 4 // 32
    if sets == "phase3":
        return np.where(n >= 25, 0, np.where(n >= 17, 1, 2))
    if sets == "color":
        return np.full_like(n, persp)
    return np.zeros_like(n)


def kb_table(nkb):
    """64 oriented king squares (files a-d used) -> bucket"""
    t = np.zeros(64, np.int64)
    for sq in range(64):
        r, f = sq >> 3, sq & 7
        f = min(f, 7 - f)  # only a-d occur after mirroring
        t[sq] = {32: r * 4 + f, 16: min(r, 3) * 4 + f, 8: min(r, 1) * 4 + f}[nkb]
    return t


def per_bucket(featset):
    return {"halfkp": 640, "hkb": 704, "thr": 1408}[featset]


def n_features(featset, nkb, sets="none"):
    return {"halfkp": 40960, "hkb": nkb * 704, "thr": nkb * 1408}[featset] * NSETS[sets]


def features(pc, stm, featset="halfkp", nkb=32, sets="none"):
    """pc (B,32) uint16, stm (B,) -> us, them (B,32) int64 feature ids (pad = n_features), piece count (B,)"""
    pc = pc.astype(np.int64)
    valid = pc != 0xFFFF
    code, sq = pc >> 6, pc & 63
    code = np.where(valid, code, 0)
    wk = np.where(valid & (code == 5), sq, 0).max(1)
    bk = np.where(valid & (code == 11), sq, 0).max(1)
    color, ptype = code // 6, code % 6
    pad = n_features(featset, nkb, sets)
    npc = valid.sum(1)
    if featset == "halfkp":
        assert sets == "none"
        keep = valid & (ptype != 5)
        fw = wk[:, None] * 640 + (color * 5 + ptype) * 64 + sq
        fb = (bk[:, None] ^ 56) * 640 + ((1 - color) * 5 + ptype) * 64 + (sq ^ 56)
    else:
        kb = kb_table(nkb)
        per_bucket, att = (1408, attacked_flags(pc)) if featset == "thr" else (704, None)
        views = []
        for persp, k in ((0, wk), (1, bk ^ 56)):
            mirror = np.where((k & 7) >= 4, 7, 0)
            ko = k ^ mirror
            flip = 56 if persp == 1 else 0
            so = (sq ^ flip) ^ mirror[:, None]
            own = color == persp
            plane = np.where(ptype == 5, 10, np.where(own, 0, 5) + ptype)
            if att is not None:
                plane = plane * 2 + att
            st = set_of(sets, npc, persp)
            views.append((st * nkb + kb[ko])[:, None] * per_bucket + plane * 64 + so)
        fw, fb = views
        keep = valid
    fw = np.where(keep, fw, pad)
    fb = np.where(keep, fb, pad)
    white = (stm == 1)[:, None]
    return np.where(white, fw, fb), np.where(white, fb, fw), npc


def bucket_of(n_pieces, nb):
    return (n_pieces - 1) * nb // 32


def ste(x, q):
    return x + (q - x).detach()


class Bucketed(torch.nn.Module):
    def __init__(self, nb, i, o):
        super().__init__()
        ref = torch.nn.Linear(i, o)
        self.w = torch.nn.Parameter(ref.weight.detach().repeat(nb, 1, 1).clone())
        self.b = torch.nn.Parameter(ref.bias.detach().repeat(nb, 1).clone())
        self.nb, self.o = nb, o

    def forward(self, x, bk, w=None):
        w = self.w if w is None else w
        y = x @ w.reshape(self.nb * self.o, -1).t()
        y = y.view(-1, self.nb, self.o).gather(1, bk.view(-1, 1, 1).expand(-1, 1, self.o)).squeeze(1)
        return y + self.b[bk]


class Net4(torch.nn.Module):
    def __init__(self, acc=512, featset="halfkp", nkb=32, psqt=False, h1=8, h2=0, nb=8, ft_act="crelu",
                 hid_act="dual", cp_scale=350.0, ft8=False, sets="none", fact=False):
        super().__init__()
        self.cfg = dict(acc=acc, featset=featset, nkb=nkb, psqt=bool(psqt), h1=h1, h2=h2, nb=nb, ft_act=ft_act,
                        hid_act=hid_act, cp_scale=float(cp_scale), ft8=bool(ft8), sets=sets, fact=bool(fact))
        nf = n_features(featset, nkb, sets)
        self.pad = nf
        self.pb = per_bucket(featset)
        # with the factorizer the real rows start at 0 so that a fresh net = the shared table (like one bucket)
        self.ft = torch.nn.Parameter(torch.randn(nf + 1, acc) * (0.0 if fact else 0.01))
        self.fac = torch.nn.Parameter(torch.randn(self.pb + 1, acc) * 0.01) if fact else None
        self.ftb = torch.nn.Parameter(torch.full((acc,), 0.5))
        self.psqt = torch.nn.Parameter(torch.zeros(nf + 1, nb)) if psqt else None
        l1_in = acc if ft_act == "pair" else 2 * acc
        mult = 2 if hid_act == "dual" else 1
        self.l1 = Bucketed(nb, l1_in, h1)
        self.l2 = Bucketed(nb, h1 * mult, h2) if h2 else None
        self.out = Bucketed(nb, (h2 if h2 else h1) * mult, 1)
        self.fq = False
        self.ft_clip = FT_CLIP  # stage 2 of a folded factorizer net raises this (row + fac spans 2*FT_CLIP)
        with torch.no_grad():
            self.ft[nf].zero_()

    def forward(self, us, them, n):
        c = self.cfg
        bag = lambda idx, w: torch.nn.functional.embedding_bag(idx, w, mode="sum", padding_idx=self.pad)
        ft = ste(self.ft, torch.round(self.ft * 127) / 127) if c["ft8"] else self.ft  # int8 grid, scale 1/127
        a, b = bag(us, ft) + self.ftb, bag(them, ft) + self.ftb
        if self.fac is not None:
            fidx = lambda i: torch.where(i == self.pad, self.pb, i % self.pb)
            fb = lambda idx, w: torch.nn.functional.embedding_bag(idx, w, mode="sum", padding_idx=self.pb)
            a, b = a + fb(fidx(us), self.fac), b + fb(fidx(them), self.fac)

        def act(x):
            x = torch.clamp(x, 0.0, 1.0)
            if c["ft_act"] == "screlu":
                return x * x
            if c["ft_act"] == "pair":
                h = x.shape[1] // 2
                return x[:, :h] * x[:, h:]
            return x

        def hid(x):
            x = torch.clamp(x, 0.0, 1.0)
            if c["hid_act"] == "screlu":
                return x * x
            if c["hid_act"] == "dual":
                return torch.cat([x, x * x], 1)
            return x
        x = torch.cat([act(a), act(b)], 1)
        bk = bucket_of(n, c["nb"])
        if self.fq:  # engine-like: uint8 L1 input, int8 L1 weights with a per-row scale
            s = 127 / self.l1.w.detach().abs().amax(-1, keepdim=True).clamp_min(1e-12)
            x = hid(self.l1(ste(x, torch.round(x * 127) / 127), bk, ste(self.l1.w, torch.round(self.l1.w * s) / s)))
        else:
            x = hid(self.l1(x, bk))
        if self.l2 is not None:
            x = hid(self.l2(x, bk))
        y = self.out(x, bk).squeeze(1)
        if self.psqt is not None:
            pu, pt = bag(us, self.psqt), bag(them, self.psqt)
            y = y + (pu - pt).gather(1, bk.view(-1, 1)).squeeze(1) / 2
        return y

    @torch.no_grad()
    def clip(self):
        self.ft.clamp_(-self.ft_clip, self.ft_clip)
        self.ft[self.pad].zero_()
        if self.fac is not None:
            self.fac.clamp_(-FT_CLIP, FT_CLIP)
            self.fac[self.pb].zero_()
        self.ftb.clamp_(-FT_CLIP, FT_CLIP)
        self.l1.w.clamp_(-L1_CLIP, L1_CLIP)
        if self.psqt is not None:
            self.psqt[self.pad].zero_()


    @torch.no_grad()
    def folded_ft(self):
        """FT rows with the factorizer added in (what the engine uses)"""
        if self.fac is None:
            return self.ft.detach().clone()
        ft = self.ft.detach().clone()
        nf = self.pad
        ft[:nf] += self.fac[:self.pb].repeat(nf // self.pb, 1)
        return ft


def load_folded(sd, cfg):
    """state dict of a (possibly factorized) net -> (cfg without fact, state dict with ft folded)"""
    cfg = dict(cfg)
    if not cfg.get("fact"):
        return cfg, sd
    net = Net4(**cfg)
    net.load_state_dict(sd)
    sd = dict(sd)
    sd["ft"] = net.folded_ft()
    del sd["fac"]
    cfg["fact"] = False
    return cfg, sd


PSQT_SCALE = 1.0 / 4096  # psqt weights are stored as int32 multiples of this (raw output units)


def export_lnn4(net, path):
    """LNN4: "LNN4" | i32 acc, rows, h1, h2, nb, ft_act, hid_act, featset, nkb, psqt | f32 ftScale, cpScale, psqtScale
    | i16 ftBias[acc] | i16 ft[rows*acc] | (psqt) i32 psqt[rows*nb] | head as LNN3 (f32 per-bucket layers)"""
    c = net.cfg
    ft = net.folded_ft().cpu().float().numpy()
    ftb = net.ftb.detach().cpu().float().numpy()
    scale = 1.0 / 127 if c.get("ft8") else (FT_CLIP * 33) / 32000.0  # ft8: weights land on the int8 grid
    sets = c.get("sets", "none")
    with open(path, "wb") as f:  # LNN5 = LNN4 + i32 weight-set selector (SETS index) after the 10 ints
        f.write((b"LNN4" if sets == "none" else b"LNN5")
                + struct.pack("<10i", c["acc"], ft.shape[0], c["h1"], c["h2"], c["nb"],
                              FT_ACTS.index(c["ft_act"]), HID_ACTS.index(c["hid_act"]),
                              FEATSETS.index(c["featset"]), c["nkb"], int(c["psqt"]))
                + (b"" if sets == "none" else struct.pack("<i", SETS.index(sets)))
                + struct.pack("<3f", scale, c["cp_scale"], PSQT_SCALE))
        f.write(np.clip(np.round(ftb / scale), -32767, 32767).astype("<i2").tobytes())
        f.write(np.clip(np.round(ft / scale), -32767, 32767).astype("<i2").tobytes())
        if net.psqt is not None:
            f.write(np.round(net.psqt.detach().cpu().double().numpy() / PSQT_SCALE).astype("<i4").tobytes())
        for layer in [net.l1, net.l2, net.out]:
            if layer is None:
                continue
            f.write(layer.w.detach().cpu().float().numpy().astype("<f4").tobytes())
            f.write(layer.b.detach().cpu().float().numpy().astype("<f4").tobytes())
    return os.path.getsize(path)


class Source:
    """one memmapped dataset: rec4=False for the old 68-byte positions.bin (no game results)."""
    def __init__(self, path, rec4, val_n=1_000_000, cp_override=None, results=True):
        self.path = path
        rec = REC4 if rec4 else REC
        self.data = np.memmap(path, mode="r", dtype=rec)
        self.has_result = rec is REC4 and results
        n = len(self.data)
        self.val_n = min(val_n, n // 20)
        self.n_train = n - self.val_n  # val = last records (latest games of the file)
        self.cp_override = cp_override  # memmapped int16 teacher labels, same length


class Data:
    """mixture of sources: each batch draws from each source in proportion `weights`.
    strata = ("phase", tau): sample piece-count bands with p ~ share^(1-tau) (tau 1 = equal bands);
             ("hard", alpha): sample error-quantile bins of a reference net with p ~ mean_err^alpha.
    Sources without a strata file (e.g. teacher) are sampled uniformly."""
    def __init__(self, sources, weights, featset, nkb, strata=None, sets="none"):
        self.src, self.w = sources, np.asarray(weights, float) / np.sum(weights)
        self.featset, self.nkb, self.sets = featset, nkb, sets
        self.val = None  # set by the caller: make(source, its val indices)
        self.strat = [None] * len(sources)
        if strata:
            kind, t = strata
            for i, sc in enumerate(sources):
                f = f"{sc.path}.{kind}"
                if not os.path.exists(f):
                    continue
                lab = np.memmap(f, mode="r", dtype="u1")[:sc.n_train]
                K = int(lab.max()) + 1
                idx = [np.flatnonzero(lab == k).astype(np.int64) for k in range(K)]
                share = np.array([len(x) for x in idx], float) / sc.n_train
                if kind == "phase":
                    p = share ** (1 - t)
                else:
                    import json
                    p = share * np.array(json.load(open(f + ".json"))["mean_err"]) ** t
                p = p / p.sum()
                self.strat[i] = (idx, p)
                print(f"strata {kind} {t} on {os.path.basename(sc.path)}: share {np.round(share, 3)} -> p {np.round(p, 3)}", flush=True)

    def make(self, s, idx):
        r = np.array(s.data[idx])
        us, them, n = features(r["pc"], r["stm"], self.featset, self.nkb, self.sets)
        cp = (np.array(s.cp_override[idx]) if s.cp_override is not None else r["cp"]).astype(np.float32)
        res = r["result"].astype(np.float32) if s.has_result else np.zeros(len(r), np.float32)
        has = np.full(len(r), 1.0 if s.has_result else 0.0, np.float32)
        return tuple(torch.from_numpy(x) for x in (us, them, n, cp, res, has))

    def batches(self, steps, batch, seed, loaders=8):
        q = queue.Queue(maxsize=16)

        def loader(k):
            rng = np.random.default_rng(seed * 1000 + k)
            for _ in range(k, steps, loaders):
                counts = rng.multinomial(batch, self.w)
                parts = []
                for s, c, st in zip(self.src, counts, self.strat):
                    if not c:
                        continue
                    if getattr(s, "idx", None) is not None:  # restricted to a category (subsets.restrict)
                        ix = s.idx[rng.integers(0, len(s.idx), c)]
                    elif st is None:
                        ix = rng.integers(0, s.n_train, c)
                    else:
                        idx, p = st
                        ix = np.concatenate([idx[k][rng.integers(0, len(idx[k]), m)]
                                             for k, m in enumerate(rng.multinomial(c, p)) if m])
                    parts.append(self.make(s, np.sort(ix)))
                q.put(tuple(torch.cat(p) for p in zip(*parts)))
            q.put(None)
        for k in range(loaders):
            threading.Thread(target=loader, args=(k,), daemon=True).start()
        done = 0
        while done < loaders:
            b = q.get()
            if b is None:
                done += 1
            else:
                yield b


X = "/scratch/ralbe/chess_nnue/exp2"
JUL = f"{X}/data/jul"
SOURCES = {
    "old": lambda: Source(OLD_DATA, rec4=False),
    "sf": lambda: Source(f"{JUL}/sf.bin", rec4=True),
    "edb": lambda: Source(f"{X}/data/evaldb/edb.bin", rec4=True, results=False),
    "lc0": lambda: Source(f"{X}/data/lc0/lc0s.bin", rec4=True),  # Leela test80 Jun 2024 self-play, cp = lc0 eval x 0.39 (SF cp scale)
    "sfo": lambda: Source(f"{X}/data/open/sfo.bin", rec4=True),  # = sf, re-extracted with sfo.meta (openings, game ids)
    "sp": lambda: Source(f"{X}/data/selfplay/all.bin", rec4=True),  # own self-play (datagen), cp = 5000-node search score  # Lichess eval DB (deep SF, no games)
    "teacher": lambda: Source(f"{JUL}/noeval.bin", rec4=True,
                              cp_override=np.memmap(f"{JUL}/noeval_teacher.i16", mode="r", dtype="<i2")),
}
