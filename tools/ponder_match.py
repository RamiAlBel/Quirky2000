"""
Match with real wall clocks where engine A may ponder (fastchess has no ponder support).
usage: python ponder_match.py GAMES CONCURRENCY TC_BASE TC_INC "A opts" "B opts" [ponderA=1]
  opts: space-separated Name=Value UCI options (EvalFile defaults to nets/t25p.nnue)
Openings: random lines from the 8moves book, each played with both colours. Clocks are measured on the
wall clock by this harness (a flag = loss). Adjudication: 3-fold/50-move/insufficient material (python-chess),
|score| >= 1000 cp for 4 consecutive moves of both engines, or 300 plies = draw.
"""
import math, multiprocessing as mp, random, sys, time
import chess, chess.engine, chess.pgn

X = "/scratch/ralbe/chess_nnue/exp2"
BOOK = "/scratch/ralbe/chess_nnue/tools/8moves_v3.pgn"


def parse_opts(s):
    o = {"EvalFile": f"{X}/nets/t25p.nnue", "Hash": 32, "Threads": 1}
    for kv in s.split():
        k, v = kv.split("=", 1)
        o[k] = int(v) if v.lstrip("-").isdigit() else v
    return o


def load_openings(n):
    games = []
    with open(BOOK) as f:
        while len(games) < 20000:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            games.append(list(g.mainline_moves()))
    random.Random(7).shuffle(games)
    return games[:n]


def play_game(args):
    opening, a_white, base, inc, optsA, optsB, ponderA, binary = args
    engines = {}
    try:
        for name, opts in (("A", optsA), ("B", optsB)):
            e = chess.engine.SimpleEngine.popen_uci(binary)
            e.configure({k: v for k, v in opts.items()})
            engines[name] = e
        board = chess.Board()
        for m in opening:
            board.push(m)
        side = {chess.WHITE: "A" if a_white else "B", chess.BLACK: "B" if a_white else "A"}
        clock = {"A": base, "B": base}
        streak = 0
        result = None
        while result is None:
            if board.is_game_over(claim_draw=True) or board.ply() >= 300:
                o = board.outcome(claim_draw=True)
                result = o.result() if o else "1/2-1/2"
                break
            who = side[board.turn]
            wc, bc = clock[side[chess.WHITE]], clock[side[chess.BLACK]]
            limit = chess.engine.Limit(white_clock=wc, black_clock=bc, white_inc=inc, black_inc=inc)
            t0 = time.time()
            r = engines[who].play(board, limit, ponder=(who == "A" and ponderA), info=chess.engine.INFO_SCORE)
            clock[who] -= time.time() - t0
            if clock[who] < 0:
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                break
            clock[who] += inc
            sc = r.info.get("score")
            if sc is not None and abs(sc.white().score(mate_score=100000)) >= 1000:
                streak += 1
                if streak >= 8:
                    result = "1-0" if sc.white().score(mate_score=100000) > 0 else "0-1"
                    break
            else:
                streak = 0
            board.push(r.move)
        a_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
        return a_score if a_white else 1 - a_score
    finally:
        for e in engines.values():
            try:
                e.quit()
            except Exception:
                pass


def main():
    games, conc, base, inc = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    optsA, optsB = parse_opts(sys.argv[5]), parse_opts(sys.argv[6])
    ponderA = len(sys.argv) <= 7 or sys.argv[7] == "1"
    binary = sys.argv[8] if len(sys.argv) > 8 else f"{X}/bin/dev"
    ops = load_openings(games // 2)
    tasks = [(o, w, base, inc, optsA, optsB, ponderA, binary) for o in ops for w in (True, False)]
    scores = []
    with mp.Pool(conc) as pool:
        for s in pool.imap_unordered(play_game, tasks):
            scores.append(s)
            if len(scores) % 50 == 0 or len(scores) == len(tasks):
                n = len(scores); mu = sum(scores) / n
                var = sum((x - mu) ** 2 for x in scores) / max(1, n - 1)
                elo = lambda p: -400 * math.log10(1 / min(max(p, 1e-6), 1 - 1e-6) - 1)
                se = math.sqrt(var / n)
                los = 0.5 * (1 + math.erf((mu - 0.5) / (se * math.sqrt(2)))) if se > 0 else 0.5
                print(f"{n} games  A score {mu:.3f}  Elo {elo(mu):+.1f} [{elo(mu - 1.96*se):+.1f}, {elo(mu + 1.96*se):+.1f}]  "
                      f"LOS {100*los:.1f}%  W/D/L {scores.count(1.0)}/{scores.count(0.5)}/{scores.count(0.0)}", flush=True)


if __name__ == "__main__":
    main()
