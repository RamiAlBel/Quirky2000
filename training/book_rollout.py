"""
Simulate how an opening book serves our bot against Lichess opponents.

python book_rollout.py OPP.npz OUT.pkl NAME=PATH[:best|:weighted] ...   (PATH "none" = no book)
For each book and each colour of ours, N rollouts from the start position: on our turn the book move (highest
weight, or weighted random), on the opponent's turn a move sampled from the opponent model (opp_tree.py counts).
A rollout ends when our side has no book move (exit), the opponent model has no data for the position (censored),
or at MAX_PLY. Saved per book: list of rollouts (colour, our book moves as (key, uci, child key),
exit key, exit ply, end reason) -> book_score.py adds Stockfish evals and prints the comparison.
"""
import pickle
import sys
import numpy as np
import chess
import chess.polyglot as pg

N, MAX_PLY = int(__import__("os").environ.get("ROLL_N", 20000)), 80
BOOKREC = np.dtype([("key", ">u8"), ("move", ">u2"), ("weight", ">u2"), ("learn", ">u4")])


def decode(board, pm):
    to, frm, pr = pm & 63, (pm >> 6) & 63, (pm >> 12) & 7
    p = board.piece_at(frm)
    if p is not None and p.piece_type == chess.KING:  # polyglot castling = king takes own rook
        r = board.piece_at(to)
        if r is not None and r.piece_type == chess.ROOK and r.color == p.color:
            to = chess.square(6 if chess.square_file(to) > chess.square_file(frm) else 2, chess.square_rank(frm))
    m = chess.Move(frm, to, promotion=pr + 1 if pr else None)
    return m if m in board.legal_moves else None


class Table:
    def __init__(self, key, move, weight):
        o = np.argsort(key, kind="stable")
        self.key, self.move, self.weight = key[o], move[o], weight[o].astype(np.float64)

    def moves(self, k):
        lo, hi = np.searchsorted(self.key, k, "left"), np.searchsorted(self.key, k, "right")
        return self.move[lo:hi], self.weight[lo:hi]


def load_book(path):
    d = np.fromfile(path, dtype=BOOKREC)
    return Table(d["key"].astype(np.uint64), d["move"].astype(np.uint16), d["weight"])


def rollout(book, mode, opp, ours, rng):
    b = chess.Board()
    played = []
    while b.ply() < MAX_PLY:
        k = np.uint64(pg.zobrist_hash(b))
        if b.turn == ours:
            if book is None:
                return played, int(k), b.ply(), "exit"
            mv, w = book.moves(k)
            ok = w > 0
            mv, w = mv[ok], w[ok]
            if len(mv) == 0:
                return played, int(k), b.ply(), "exit"
            i = int(np.argmax(w)) if mode == "best" else int(rng.choice(len(mv), p=w / w.sum()))
            m = decode(b, int(mv[i]))
            if m is None:
                return played, int(k), b.ply(), "exit"
            b.push(m)
            played.append((int(k), m.uci(), int(pg.zobrist_hash(b))))
        else:
            mv, w = opp.moves(k)
            if len(mv) == 0:
                return played, int(k), b.ply(), "censored"
            m = decode(b, int(mv[rng.choice(len(mv), p=w / w.sum())]))
            if m is None:
                return played, int(k), b.ply(), "censored"
            b.push(m)
    return played, int(pg.zobrist_hash(b)), b.ply(), "maxply"


def main():
    o = np.load(sys.argv[1])
    opp = Table(o["key"], o["move"], o["count"])
    out = {}
    for spec in sys.argv[3:]:
        name, rest = spec.split("=", 1)
        path, mode = (rest.rsplit(":", 1) + ["best"])[:2] if rest.endswith((":best", ":weighted")) else (rest, "best")
        book = None if path == "none" else load_book(path)
        rng = np.random.default_rng(1)
        res = []
        for ours in (chess.WHITE, chess.BLACK):
            for _ in range(N):
                played, ek, ep, why = rollout(book, mode, opp, ours, rng)
                res.append((ours, played, ek, ep, why))
        out[name] = dict(path=path, mode=mode, rollouts=res)
        nb = np.array([len(r[1]) for r in res])
        print(f"{name:14s} {mode:8s} our book moves: mean {nb.mean():.1f}  median {np.median(nb):.0f}  "
              f"0 moves {np.mean(nb == 0):.1%}  >=10 {np.mean(nb >= 10):.1%}", flush=True)
    pickle.dump(out, open(sys.argv[2], "wb"))


if __name__ == "__main__":
    main()
