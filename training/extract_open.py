"""
extract_open.py: like extract_v4.py, but only games with %eval, and a sidecar per record with game-level categories
(for specialised nets: opening family, rating, time control). Outputs OUT_DIR/sfo.bin (REC4) + sfo.meta (META):
  eco u16 (letter*100 + number, A00 = 0 .. E99 = 499; 65535 unknown), game u32 (running game id), elo u16 (mean
  rating), tc u16 (base seconds, clamped), m u16[4] first four plies (from | to << 6 | promo << 12)

Extract training positions + game results from a Lichess PGN dump (streamed on stdin, e.g. `zstd -dc x.pgn.zst |`).

Two outputs in OUT_DIR, both fixed 70-byte records (REC4):
  pc     uint16[32]  piece codes (color*6 + type)*64 + square, kings included, pad 0xFFFF
  cp     int16       Stockfish eval, side to move, clamped +-1500 (mate = +-1500); 0 in noeval.bin
  stm    uint8       1 = white to move
  ply    uint8       game ply (clamped 255)
  result int8        game result from the side to move's view: +1 win, 0 draw, -1 loss
  pad    uint8
- sf.bin:     every quiet position of games that carry %eval comments (like extract_v3, plus result)
- noeval.bin: a sample of quiet positions from games without evals (both players >= NOEVAL_MIN_ELO),
              for labelling with a teacher net
Filters: ply >= 8, side to move not in check, next move quiet (no capture/promotion), not the final position.

usage: zstd -dc lichess.pgn.zst | python extract_v4.py OUT_DIR [workers]
"""
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
import numpy as np

REC4 = np.dtype([("pc", "<u2", (32,)), ("cp", "<i2"), ("stm", "u1"), ("ply", "u1"), ("result", "i1"), ("pad", "u1")])
assert REC4.itemsize == 70
META = np.dtype([("eco", "<u2"), ("game", "<u4"), ("elo", "<u2"), ("tc", "<u2"), ("m", "<u2", (4,))])
ECO_RE = re.compile(rb'\[ECO "([A-E])(\d\d)"\]')
TC_RE = re.compile(rb'\[TimeControl "(\d+)\+\d+"\]')
MIN_PLY = 8
CP_CLAMP = 1500
GAMES_PER_TASK = 400
NOEVAL_GAME_P = 0.30   # fraction of eval-less games replayed
NOEVAL_POS_P = 0.15    # fraction of their quiet positions kept
NOEVAL_MIN_ELO = 1600

TOKEN_RE = re.compile(r"\{([^}]*)\}|([NBRQKa-hO][A-Za-z0-9+#=\-]*)")
EVAL_RE = re.compile(r"\[%eval\s+(#?-?\d+(?:\.\d+)?)\]")
RESULT_RE = re.compile(rb'\[Result "([^"]+)"\]')
ELO_RE = re.compile(rb'\[(White|Black)Elo "(\d+)"\]')
RESULTS = {b"1-0": 1, b"0-1": -1, b"1/2-1/2": 0}


def pack(board, chess, ply, cp_stm, res_white, out):
    rec = np.full(32, 0xFFFF, dtype=np.uint16)
    n = 0
    for color in (chess.WHITE, chess.BLACK):
        cbase = 0 if color == chess.WHITE else 6
        for pt in range(1, 7):
            code = (cbase + pt - 1) * 64
            for sq in chess.scan_forward(board.pieces_mask(pt, color)):
                if n < 32:
                    rec[n] = code + sq
                n += 1
    if n > 32:
        return
    out.append((rec, cp_stm, 1 if board.turn else 0, min(ply, 255), res_white if board.turn else -res_white))


def parse_games(task):
    import chess
    raw_games, has_eval, seed, gid0 = task
    rng = random.Random(seed)
    out, meta = [], []
    for gi, raw in enumerate(raw_games):
        n0 = len(out)
        m = RESULT_RE.search(raw)
        if not m or m.group(1) not in RESULTS:
            continue
        res_white = RESULTS[m.group(1)]
        text = raw.decode("utf-8", "replace")
        if "[Variant " in text or "[SetUp " in text:
            continue
        i = text.find("\n1. ")
        if i < 0:
            continue
        board = chess.Board()
        ply = 0
        first4 = [0, 0, 0, 0]
        pending = None  # (ply, cp_stm) for the current board, waiting to see the next move
        try:
            for t in TOKEN_RE.finditer(text[i + 1:]):
                comment, san = t.group(1), t.group(2)
                if comment is not None:
                    if not has_eval:
                        continue
                    e = EVAL_RE.search(comment)
                    if e and ply >= MIN_PLY and not board.is_check():
                        raw_e = e.group(1)
                        if raw_e.startswith("#"):
                            cp_white = CP_CLAMP if int(raw_e[1:]) > 0 else -CP_CLAMP
                        else:
                            cp_white = max(-CP_CLAMP, min(CP_CLAMP, int(round(float(raw_e) * 100))))
                        pending = (ply, cp_white if board.turn else -cp_white)
                    continue
                move = board.parse_san(san)
                if not has_eval and ply >= MIN_PLY and rng.random() < NOEVAL_POS_P and not board.is_check():
                    pending = (ply, 0)
                if pending is not None:
                    if not board.is_capture(move) and move.promotion is None:
                        pack(board, chess, pending[0], pending[1], res_white, out)
                    pending = None
                if ply < 4:
                    first4[ply] = move.from_square | move.to_square << 6 | (move.promotion or 0) << 12
                board.push(move)
                ply += 1
        except Exception:
            pass
        if len(out) > n0:
            e = ECO_RE.search(raw)
            eco = (e.group(1)[0] - 65) * 100 + int(e.group(2)) if e else 65535
            elos = [int(x) for _, x in ELO_RE.findall(raw)]
            t = TC_RE.search(raw)
            meta.append((len(out) - n0, eco, gid0 + gi, sum(elos) // max(len(elos), 1), min(int(t.group(1)), 65535) if t else 0,
                         first4))
    arr = np.zeros(len(out), dtype=REC4)
    for k, (rec, cp, stm, pl, res) in enumerate(out):
        arr[k]["pc"], arr[k]["cp"], arr[k]["stm"], arr[k]["ply"], arr[k]["result"] = rec, cp, stm, pl, res
    md = np.zeros(len(out), dtype=META)
    k = 0
    for cnt, eco, gid, elo, tc, f4 in meta:
        md[k:k + cnt]["eco"], md[k:k + cnt]["game"], md[k:k + cnt]["elo"], md[k:k + cnt]["tc"] = eco, gid, elo, tc
        md[k:k + cnt]["m"] = f4
        k += cnt
    return arr.tobytes(), md.tobytes(), len(raw_games)


def game_batches(stats):
    rng = random.Random(12345)
    sep = b"\n\n[Event "
    src = sys.stdin.buffer
    buf = b""
    batches = {True: [], False: []}
    seed = 0
    while True:
        chunk = src.read(1 << 24)
        if not chunk:
            break
        buf += chunk
        parts = buf.split(sep)
        buf = parts.pop()
        for p in parts:
            stats["games"] += 1
            if b"%eval" not in p:
                continue
            batches[True].append(p)
            if len(batches[True]) >= GAMES_PER_TASK:
                seed += 1
                yield batches[True], True, seed, stats["gid"]
                stats["gid"] += len(batches[True])
                batches[True] = []
    if batches[True]:
        yield batches[True], True, seed + 1, stats["gid"]


def main():
    out_dir = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    os.makedirs(out_dir, exist_ok=True)
    files = {True: open(os.path.join(out_dir, "sfo.bin"), "wb")}
    mfile = open(os.path.join(out_dir, "sfo.meta"), "wb")
    counts = {True: 0, False: 0}
    games = {True: 0, False: 0}
    stats = {"games": 0, "gid": 0}
    t0 = last = time.time()

    def save_meta(done):
        json.dump(dict(sf_positions=counts[True], noeval_positions=counts[False], eval_games=games[True],
                       noeval_games=games[False], games_scanned=stats["games"], record_bytes=REC4.itemsize,
                       min_ply=MIN_PLY, cp_clamp=CP_CLAMP, noeval_game_p=NOEVAL_GAME_P, noeval_pos_p=NOEVAL_POS_P,
                       noeval_min_elo=NOEVAL_MIN_ELO, done=done,
                       filters="ply>=8, not in check, next move quiet (no capture/promotion)"),
                  open(os.path.join(out_dir, "sfo_info.json"), "w"), indent=1)

    with mp.Pool(workers) as pool:
        for data, md, ng in pool.imap_unordered(parse_games, game_batches(stats), chunksize=1):
            ev = True
            files[ev].write(data)
            mfile.write(md)
            counts[ev] += len(data) // REC4.itemsize
            games[ev] += ng
            now = time.time()
            if now - last > 60:
                el = now - t0
                print(f"[{el/60:6.1f} min] sf {counts[True]/1e6:7.2f}M pos ({games[True]/1e6:.2f}M games)  "
                      f"noeval {counts[False]/1e6:7.2f}M pos ({games[False]/1e6:.2f}M games)  scanned {stats['games']/1e6:.1f}M",
                      flush=True)
                last = now
                save_meta(False)
    for f in files.values():
        f.close()
    mfile.close()
    save_meta(True)
    print(f"DONE in {(time.time()-t0)/60:.1f} min: sf {counts[True]:,}  noeval {counts[False]:,}", flush=True)


if __name__ == "__main__":
    main()
