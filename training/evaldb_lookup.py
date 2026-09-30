"""
Look up Lichess eval-DB (deep Stockfish, multi-PV) evaluations for a set of positions given by Polyglot keys.

zstd -dc lichess_db_eval.jsonl.zst | python evaldb_lookup.py KEYS.npy OUT.pkl [workers]
KEYS.npy: uint64 Polyglot keys. OUT.pkl: {key: (depth, {first move uci: cp from the side to move's view})} using the
deepest analysis of each position (mate = +-1500). Only FENs with >= MIN_PIECES pieces are parsed (opening sets).
"""
import json
import multiprocessing as mp
import pickle
import sys
import numpy as np

MIN_PIECES, CLAMP = 16, 1500
KEYS = None


def init(path):
    global KEYS
    KEYS = np.load(path)


def work(lines):
    import chess, chess.polyglot as pg
    out = {}
    for ln in lines:
        board = ln[8:ln.index(b" ", 8)]  # after {"fen":"
        if sum(c in b"pnbrqkPNBRQK" for c in board) < MIN_PIECES:
            continue
        d = json.loads(ln)
        b = chess.Board(d["fen"] + " 0 1")
        k = np.uint64(pg.zobrist_hash(b))
        i = np.searchsorted(KEYS, k)
        if i >= len(KEYS) or KEYS[i] != k:
            continue
        ev = max(d["evals"], key=lambda e: e["depth"])
        pv = {}
        for p in ev["pvs"]:
            mv = p["line"].split()[0] if p.get("line") else None
            if mv is None:
                continue
            cp = (CLAMP if p["mate"] > 0 else -CLAMP) if "mate" in p else max(-CLAMP, min(CLAMP, p["cp"]))
            if b.turn == chess.BLACK:  # eval-DB scores are from white's view
                cp = -cp
            pv[mv] = cp
        if pv:
            out[int(k)] = (ev["depth"], pv)
    return out


def batches():
    batch = []
    for ln in sys.stdin.buffer:
        batch.append(ln)
        if len(batch) >= 20000:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    res = {}
    n = 0
    with mp.Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 8, initializer=init, initargs=(sys.argv[1],)) as pool:
        for d in pool.imap_unordered(work, batches(), chunksize=4):
            for k, v in d.items():
                if k not in res or v[0] > res[k][0]:
                    res[k] = v
            n += 1
            if n % 2000 == 0:
                print(f"{n*20000/1e6:.0f}M lines, {len(res):,} positions found", flush=True)
    pickle.dump(res, open(sys.argv[2], "wb"))
    print(f"DONE {len(res):,} of {len(np.load(sys.argv[1])):,} positions found", flush=True)


if __name__ == "__main__":
    main()
