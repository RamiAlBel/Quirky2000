"""engine `sig` vs PyTorch sigma.py for a few FENs: python xcheck_sigma.py NET SIGNAME ENGINE"""
import subprocess, sys, numpy as np, torch, chess
sys.argv += []
import nnue4 as N
X = "/scratch/ralbe/chess_nnue/exp2"
netname, signame, eng = sys.argv[1:4]
sys.argv = [sys.argv[0], netname, signame]
FENS = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
        "r2q1rk1/pp3ppp/2n1b3/3pP3/3Pn3/2PB1N2/P4PPP/R1BQ1RK1 b - - 0 12",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        "r1b2rk1/2q1bppp/p2ppn2/1p6/3NP3/1BN1B3/PPP1QPPP/2KR3R w - - 0 12"]
s = torch.load(f"{X}/nets/{netname}/last.pt", map_location="cpu", weights_only=False)
cfg = s["cfg"]; net = N.Net4(**cfg); net.load_state_dict(s["ema"] or s["net"]); net.eval(); net.fq = True
import importlib.util
src = open(f"{X}/training/sigma.py").read()
Sigma_src = src[src.index("class Sigma"):src.index("spec = ")]
ns = {"torch": torch, "N": N}
exec(src[src.index("def ft_input"):src.index("class Sigma")], {**ns, "net": net, "cfg": cfg}, ns)
exec(Sigma_src, ns)
l1_in = cfg["acc"] if cfg["ft_act"] == "pair" else 2 * cfg["acc"]
sig = ns["Sigma"](l1_in, 8, 8); sig.load_state_dict(torch.load(f"{X}/nets/{signame}/sigma.pt")["sig"]); sig.eval()
for fen in FENS:
    b = chess.Board(fen)
    pc = np.full((1, 32), 0xFFFF, np.uint16); k = 0
    for color in (chess.WHITE, chess.BLACK):
        for pt in range(1, 7):
            for sq in chess.scan_forward(b.pieces_mask(pt, color)):
                pc[0, k] = ((0 if color else 6) + pt - 1) * 64 + sq; k += 1
    us, them, n = (torch.from_numpy(x) for x in N.features(pc, np.array([1 if b.turn else 0], np.uint8), cfg["featset"], cfg["nkb"]))
    with torch.no_grad():
        ev = net(us, them, n) * cfg["cp_scale"]
        sg = sig(ns["ft_input"](us, them), n, ev) * 100
    cmd = f"setoption name EvalFile value {X}/nets/{netname}/{netname}.nnue\nsetoption name SigmaFile value {X}/nets/{signame}/{signame}.sig\nposition fen {fen}\nsig\nquit\n"
    o = subprocess.run([eng], input=cmd, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    print(f"torch eval {ev.item():7.1f} sigma {sg.item():6.1f}   engine {o}   {fen}")
