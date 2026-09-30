"""
Build a Polyglot opening book from a Lichess PGN dump on stdin (zstd -dc x.pgn.zst | python build_book.py out.bin).
Games: both players >= MIN_ELO, base time >= MIN_BASE seconds. First MAX_PLY plies. A move is kept if it was
played >= MIN_GAMES times in that position; weight = 2*wins + draws of the side that played it (scaled to u16).
"""
import collections, multiprocessing as mp, re, struct, sys
import chess, chess.polyglot as pg

MIN_ELO, MIN_BASE, MAX_PLY, MIN_GAMES = 2200, 180, 20, 25
ELO_RE = re.compile(rb'\[(White|Black)Elo "(\d+)"\]')
TC_RE = re.compile(rb'\[TimeControl "(\d+)\+(\d+)"\]')
RES_RE = re.compile(rb'\[Result "(1-0|0-1|1/2-1/2)"\]')
SAN_RE = re.compile(r"\{[^}]*\}|\d+\.+|([NBRQKa-hO][A-Za-z0-9+#=\-]*)")
SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def poly_move(board, move):
    to = move.to_square
    if board.is_castling(move):  # polyglot: king "takes" its own rook
        to = chess.square(7 if chess.square_file(move.to_square) == 6 else 0, chess.square_rank(move.from_square))
    promo = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[move.promotion]
    return to | (move.from_square << 6) | (promo << 12)


def work(games):
    stats = collections.defaultdict(lambda: [0, 0.0])
    for raw in games:
        res = RES_RE.search(raw)
        if not res:
            continue
        white_score = SCORE[res.group(1).decode()]
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
                s = stats[(pg.zobrist_hash(b), poly_move(b, mv))]
                s[0] += 1
                s[1] += white_score if b.turn == chess.WHITE else 1 - white_score
                b.push(mv)
        except Exception:
            pass
    return dict(stats)


def games():
    buf, batch, sep = b"", [], b"\n\n[Event "
    while True:
        chunk = sys.stdin.buffer.read(1 << 24)
        if not chunk:
            break
        buf += chunk
        parts = buf.split(sep)
        buf = parts.pop()
        for p in parts:
            tc = TC_RE.search(p)
            elos = ELO_RE.findall(p)
            if not tc or int(tc.group(1)) < MIN_BASE or len(elos) < 2 or min(int(e) for _, e in elos) < MIN_ELO:
                continue
            batch.append(p)
            if len(batch) >= 2000:
                yield batch
                batch = []
    if batch:
        yield batch


def main():
    total = collections.defaultdict(lambda: [0, 0.0])
    n = 0
    with mp.Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 8) as pool:
        for d in pool.imap_unordered(work, games()):
            n += 1
            for k, (c, s) in d.items():
                t = total[k]; t[0] += c; t[1] += s
            if n % 200 == 0:
                print(f"{n*2000/1e6:.1f}M games, {len(total)/1e6:.2f}M (pos,move) pairs", flush=True)
    by_key = collections.defaultdict(list)
    for (key, mv), (c, s) in total.items():
        if c >= MIN_GAMES:
            by_key[key].append((mv, 2 * s))  # 2*wins + draws
    out = []
    for key, mvs in by_key.items():
        top = max(w for _, w in mvs)
        if top <= 0:  # every candidate lost in every game: leave the position out of the book
            continue
        for mv, w in mvs:
            ww = int(round(65535 * w / top))
            if ww > 0:
                out.append((key, mv, ww))
    out.sort(key=lambda e: (e[0], -e[2]))
    with open(sys.argv[1], "wb") as f:
        for key, mv, w in out:
            f.write(struct.pack(">QHHI", key, mv, w, 0))
    print(f"DONE: {len(by_key)} positions, {len(out)} entries -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
