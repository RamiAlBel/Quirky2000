"""
Opponent model for opening-book evaluation: how often each move is played in each position by Lichess players.
zstd -dc lichess.pgn.zst | python opp_tree.py OUT.npz [workers]
Games: both players >= MIN_ELO, base time >= MIN_BASE s; first MAX_PLY plies; (position, move) pairs seen >= MIN_GAMES
times are kept. OUT.npz: key u64, move u16 (polyglot encoding), count u32, sorted by key.
"""
import collections, multiprocessing as mp, sys
import numpy as np
import chess, chess.polyglot as pg
sys.path.insert(0, __import__("os").path.dirname(__file__))
from build_book import poly_move, games as _games, ELO_RE, TC_RE, SAN_RE
import build_book
build_book.MIN_ELO, build_book.MIN_BASE = 2000, 180
MAX_PLY, MIN_GAMES = 40, 3


def work(batch):
    stats = collections.Counter()
    for raw in batch:
        text = raw.decode("utf-8", "replace")
        i = text.find("\n1. ")
        if i < 0:
            continue
        b = chess.Board()
        try:
            for m in SAN_RE.finditer(text[i + 1:]):
                san = m.group(1)
                if not san:
                    continue
                if b.ply() >= MAX_PLY:
                    break
                mv = b.parse_san(san)
                stats[(pg.zobrist_hash(b), poly_move(b, mv))] += 1
                b.push(mv)
        except Exception:
            pass
    return stats


def main():
    total = collections.Counter()
    n = 0
    with mp.Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as pool:
        for d in pool.imap_unordered(work, _games()):
            total.update(d)
            n += 1
            if n % 200 == 0:
                print(f"{n*2000/1e6:.1f}M games, {len(total)/1e6:.1f}M pairs", flush=True)
    items = [(k, m, c) for (k, m), c in total.items() if c >= MIN_GAMES]
    items.sort()
    a = np.array(items, dtype=np.uint64) if items else np.zeros((0, 3), np.uint64)
    np.savez(sys.argv[1], key=a[:, 0], move=a[:, 1].astype(np.uint16), count=a[:, 2].astype(np.uint32))
    print(f"DONE {n*2000} games (batches), {len(items)} (pos, move) pairs", flush=True)


if __name__ == "__main__":
    main()
