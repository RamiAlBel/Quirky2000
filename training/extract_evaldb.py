"""
Convert the Lichess evaluation database (lichess_db_eval.jsonl.zst, streamed on stdin) into REC4 training records.

  zstd -dc lichess_db_eval.jsonl.zst | python extract_evaldb.py OUT_DIR [workers]

Per position the deepest eval is used. Outputs in OUT_DIR:
  edb.bin    REC4 records (see extract_v4.py); cp = eval of the deepest analysis, side to move, clamped +-1500
             (mate = +-1500); result = 0 and ply = 0 (unknown: the source has no games)
  edb.move   uint16 per record: best move (first move of PV 1) = from | to << 6 | promo << 12 (promo 1..4 = n,b,r,q)
  edb.gap    int16 per record: cp(PV1) - cp(PV2) from the side to move's view (1500 if only one PV)
  edb.depth  uint8 per record: depth of the analysis used
Same filters as extract_v4.py: side to move not in check, best move quiet (no capture/promotion/castling).
"""
import json
import multiprocessing as mp
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from extract_v4 import REC4, CP_CLAMP

LINES_PER_TASK = 20000


def cp_of(pv):
    if "mate" in pv:
        return CP_CLAMP if pv["mate"] > 0 else -CP_CLAMP
    return max(-CP_CLAMP, min(CP_CLAMP, pv["cp"]))


def work(lines):
    import chess
    recs, moves, gaps, depths = [], [], [], []
    for ln in lines:
        try:
            d = json.loads(ln)
            ev = max(d["evals"], key=lambda e: e["depth"])
            pvs = ev["pvs"]
            first = pvs[0]["line"].split()[0]
            board = chess.Board(d["fen"] + " 0 1", chess960=True)
            if board.is_check():
                continue
            mv = chess.Move.from_uci(first)
            tgt = board.piece_at(mv.to_square)
            if tgt is not None or mv.promotion or board.is_en_passant(mv) or board.is_castling(mv):
                continue
            sgn = 1 if board.turn else -1
            cp1 = sgn * cp_of(pvs[0])
            gap = cp1 - sgn * cp_of(pvs[1]) if len(pvs) > 1 else CP_CLAMP
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
                continue
            recs.append((rec, cp1, 1 if board.turn else 0))
            moves.append(mv.from_square | mv.to_square << 6)
            gaps.append(max(-32000, min(32000, gap)))
            depths.append(min(255, ev["depth"]))
        except Exception:
            continue
    arr = np.zeros(len(recs), dtype=REC4)
    for k, (rec, cp, stm) in enumerate(recs):
        arr[k]["pc"], arr[k]["cp"], arr[k]["stm"] = rec, cp, stm
    return (arr.tobytes(), np.array(moves, "<u2").tobytes(), np.array(gaps, "<i2").tobytes(),
            np.array(depths, "u1").tobytes(), len(lines))


def chunks():
    buf = []
    for ln in sys.stdin.buffer:
        buf.append(ln)
        if len(buf) == LINES_PER_TASK:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    out = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    os.makedirs(out, exist_ok=True)
    fs = [open(f"{out}/edb.{x}", "wb") for x in ("bin", "move", "gap", "depth")]
    t0, seen, kept = time.time(), 0, 0
    with mp.Pool(workers) as pool:
        for i, (b, m, g, dp, n) in enumerate(pool.imap(work, chunks(), chunksize=1)):
            for f, x in zip(fs, (b, m, g, dp)):
                f.write(x)
            seen += n
            kept += len(b) // REC4.itemsize
            if i % 200 == 0:
                print(f"{seen / 1e6:.1f}M lines  kept {kept / 1e6:.1f}M  {seen / (time.time() - t0) / 1e3:.0f}k lines/s", flush=True)
    for f in fs:
        f.close()
    json.dump({"lines": seen, "kept": kept}, open(f"{out}/edb_stats.json", "w"))
    print(f"done: {seen} lines, kept {kept}", flush=True)


if __name__ == "__main__":
    main()
