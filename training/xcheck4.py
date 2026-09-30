# Engine vs PyTorch for LNN4 nets: python xcheck4.py ENGINE [ckpt.pt net.nnue] ; without a checkpoint it
# builds random nets for every feature-set/psqt combination and also runs acccheck (incremental == refresh).
import subprocess, sys, numpy as np, torch
sys.path.insert(0, "/scratch/ralbe/chess_nnue/exp2/training")
import nnue4 as N
ENG = sys.argv[1]
FENS = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b - - 5 6",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4k3/8/8/3q4/8/8/8/3QK3 b - - 0 1",
        "8/8/3k4/8/8/4K3/8/8 w - - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 b - - 0 10",
        # 17 / 18 / 25 pieces with captures: incremental updates cross the weight-set bands (LNN5)
        "5r2/5kPp/2p1R1nP/p1p5/P2p4/1P1P4/2PB2K1/8 w - - 0 1", "2r5/pb3kb1/1p1Q4/3p2p1/8/2P1PN2/PP3PP1/2K4R w - - 0 1",
        "1r3rk1/5ppp/q1pPp1n1/p1p5/P2p2PP/1P1P2Q1/2PB4/4RRK1 b - - 0 1"]
ACC_FENS = FENS[1:4] + FENS[7:]
def enc(fen):
    board, stm = fen.split()[:2]
    pcs = []
    for r, row in enumerate(board.split("/")):
        f = 0
        for ch in row:
            if ch.isdigit(): f += int(ch); continue
            pcs.append(("PNBRQKpnbrqk".index(ch) << 6) | ((7 - r) * 8 + f)); f += 1
    pc = np.full((1, 32), 0xFFFF, np.uint16); pc[0, :len(pcs)] = pcs
    return pc, np.array([1 if stm == "w" else 0], np.uint8)
def check(net, path, tag):
    net.eval()
    cmds = f"setoption name EvalFile value {path}\n" + "".join(f"position fen {f}\neval\n" for f in FENS)
    for finny in (0, 1):
        cmds += f"setoption name UseFinny value {finny}\n" + "".join(f"position fen {f}\nacccheck 3\n" for f in ACC_FENS)
    out = subprocess.run([ENG], input=cmds + "quit\n", capture_output=True, text=True)
    lines = out.stdout.splitlines()
    evals = [l for l in lines if l.startswith("raw float")]
    acc = [l for l in lines if l.startswith("acccheck")]
    bad_acc = [l for l in acc if not l.endswith(" 0 mismatches")]
    worst = 0
    for fen, l in zip(FENS, evals):
        us, them, n = (torch.from_numpy(t) for t in N.features(*enc(fen), net.cfg["featset"], net.cfg["nkb"], net.cfg.get("sets", "none")))
        with torch.no_grad():
            t = net(us, them, n).item()
        worst = max(worst, abs(t - float(l.split()[2])))
    ok = len(evals) == len(FENS) and worst < 0.01 and len(acc) == 2 * len(ACC_FENS) and not bad_acc
    print(f"{tag:28s} evals {len(evals)}/{len(FENS)}  max|torch-engine float| {worst:.5f}  acccheck {len(acc)} runs, "
          f"{len(bad_acc)} with mismatches  {'OK' if ok else 'FAIL'}  {out.stderr.strip()[:120]}")
    return ok
if len(sys.argv) > 3:
    s = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
    net = N.Net4(**s["cfg"]); net.load_state_dict(s["ema"] or s["net"]); check(net, sys.argv[3], sys.argv[3].split("/")[-1])
else:
    torch.manual_seed(0)
    allok = True
    for fs, nkb, psqt, fa, h1, h2 in [("halfkp", 32, 0, "crelu", 8, 0), ("halfkp", 32, 1, "crelu", 8, 0),
                                      ("hkb", 32, 0, "crelu", 8, 0), ("hkb", 32, 1, "crelu", 8, 0),
                                      ("hkb", 16, 1, "pair", 16, 32), ("hkb", 8, 0, "crelu", 4, 16),
                                      ("thr", 32, 0, "crelu", 8, 0), ("thr", 16, 1, "crelu", 8, 0),
                                      ("hkb", 8, "phase2", "pair", 8, 0), ("hkb", 8, "phase3", "pair", 8, 0),
                                      ("hkb", 16, "phase4", "pair", 8, 0), ("hkb", 8, "color", "pair", 8, 0)]:
        sets = psqt if isinstance(psqt, str) else "none"
        psqt = 0 if isinstance(psqt, str) else psqt
        net = N.Net4(acc=int(__import__("os").environ.get("XACC", 512)), featset=fs, nkb=nkb, psqt=psqt, ft_act=fa, h1=h1, h2=h2,
                     sets=sets, fact=sets != "none")
        with torch.no_grad():
            net.ft.normal_(0, 0.03); net.ft[net.pad].zero_()
            if net.fac is not None: net.fac.normal_(0, 0.03); net.fac[net.pb].zero_()
            if net.psqt is not None: net.psqt.normal_(0, 0.1); net.psqt[net.pad].zero_()
        path = f"/tmp/xc4_{fs}_{nkb}_{psqt}_{fa}_{sets}.nnue"
        N.export_lnn4(net, path)
        allok &= check(net, path, f"{fs} kb{nkb} psqt{psqt} {fa} {h1}/{h2} {sets}")
    print("ALL OK" if allok else "SOME FAILED")
