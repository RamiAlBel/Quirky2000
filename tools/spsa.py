"""
SPSA tuning of engine UCI parameters with fastchess (Fishtest-style schedule).
usage: python spsa.py CONFIG.json STATE.json ITERATIONS PAIRS_PER_ITER CONCURRENCY TC "fixed opts"
CONFIG: {"Name": [start, min, max, c_end], ...}; fixed opts are applied to both sides (e.g. accepted switches).
Each iteration plays PAIRS game pairs (same opening, colours swapped) of theta+c*delta vs theta-c*delta and moves
theta by a_k/c_k * (wins - losses of theta+) * delta. c_k = c_end*N^0.101/(k+1)^0.101,
a_k = R_end*c_end^2*(A+N)^0.602/(A+k+1)^0.602 with R_end = 0.002, A = 0.1 N. STATE.json is updated every
iteration (resumable) and holds the theta history.
"""
import json, os, random, re, subprocess, sys

X = "/scratch/ralbe/chess_nnue/exp2"
FC = "/scratch/ralbe/chess_nnue/tools/fastchess/fastchess"
BOOK = "/scratch/ralbe/chess_nnue/tools/8moves_v3.pgn"
R_END = 0.002


def opts_args(d):
    d = {"EvalFile": f"{X}/nets/t25p.nnue", **d}  # a fixed EvalFile overrides the t25p default
    a = [f"option.{k}={v}" for k, v in d.items()]
    return a


def main():
    cfg = json.load(open(sys.argv[1]))
    state_path = sys.argv[2]
    N, pairs, conc, tc = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    fixed = dict(kv.split("=", 1) for kv in (sys.argv[7].split() if len(sys.argv) > 7 else []))
    binary = sys.argv[8] if len(sys.argv) > 8 else f"{X}/bin/spsa"
    A = 0.1 * N
    if os.path.exists(state_path):
        st = json.load(open(state_path))
    else:
        st = {"k": 0, "theta": {n: float(v[0]) for n, v in cfg.items()}, "history": []}
    rng = random.Random(st["k"] * 7919 + 1)
    while st["k"] < N:
        k = st["k"]
        plus, minus, delta, ck = {}, {}, {}, {}
        for n, (start, lo, hi, c_end) in cfg.items():
            c = c_end * N ** 0.101 / (k + 1) ** 0.101
            d = rng.choice((-1, 1))
            delta[n], ck[n] = d, c
            plus[n] = int(round(min(hi, max(lo, st["theta"][n] + c * d))))
            minus[n] = int(round(min(hi, max(lo, st["theta"][n] - c * d))))
        cmd = [FC, "-engine", f"cmd={binary}", "name=plus", *opts_args({**fixed, **plus}),
               "-engine", f"cmd={binary}", "name=minus", *opts_args({**fixed, **minus}),
               "-each", f"tc={tc}", "option.Threads=1", "option.Hash=16",
               "-rounds", str(pairs), "-games", "2", "-repeat", "-concurrency", str(conc),
               "-openings", f"file={BOOK}", "format=pgn", "order=random", f"-srand", str(1000 + k), "-recover"]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
        m = re.findall(r"Wins: (\d+), Losses: (\d+), Draws: (\d+)", out)
        if not m:
            print("fastchess output not understood:\n" + out[-2000:], flush=True)
            sys.exit(1)
        w, l, dr = map(int, m[-1])
        res = w - l
        for n, (start, lo, hi, c_end) in cfg.items():
            a = R_END * c_end ** 2 * (A + N) ** 0.602 / (A + k + 1) ** 0.602
            st["theta"][n] = min(hi, max(lo, st["theta"][n] + a / ck[n] * res * delta[n]))
        st["k"] = k + 1
        st["history"].append({"k": k + 1, "wld": [w, l, dr], "theta": dict(st["theta"])})
        json.dump(st, open(state_path + ".tmp", "w"), indent=1)
        os.replace(state_path + ".tmp", state_path)
        print(f"iter {k+1}/{N}  W/L/D {w}/{l}/{dr}  " + "  ".join(f"{n}={v:.1f}" for n, v in st["theta"].items()), flush=True)


if __name__ == "__main__":
    main()
