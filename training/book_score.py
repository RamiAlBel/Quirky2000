"""
Score opening books from book_rollout.py output with Lichess eval-DB (Stockfish) evaluations.

python book_score.py ROLLOUTS.pkl keys OUT_KEYS.npy      -> positions to look up (then evaldb_lookup.py)
python book_score.py ROLLOUTS.pkl score EVALS.pkl [OUT.md]
Per book: our book moves per game (mean, share of games with >= 10), censored share (opponent model ran out while
we were still in book), per book move: cp loss vs the best eval-DB move (multi-PV, else via the child position) and
share of moves within 10 cp of best; eval at the end of the book from our side (eval-DB, cp).
"""
import pickle
import sys
import numpy as np


def castle_alt(uci):  # the eval DB may write castling as king-takes-rook
    alt = {"e1g1": "e1h1", "e1c1": "e1a1", "e8g8": "e8h8", "e8c8": "e8a8"}
    return alt.get(uci)


def main():
    ro = pickle.load(open(sys.argv[1], "rb"))
    if sys.argv[2] == "keys":
        ks = set()
        for b in ro.values():
            for ours, played, ek, ep, why in b["rollouts"]:
                ks.add(ek)
                for k, u, c in played:
                    ks.add(k); ks.add(c)
        a = np.array(sorted(ks), dtype=np.uint64)
        np.save(sys.argv[3], a)
        print(f"{len(a):,} keys")
        return
    ev = pickle.load(open(sys.argv[3], "rb"))
    best = lambda k: max(ev[k][1].values()) if k in ev else None
    rows = []
    for name, b in ro.items():
        R = b["rollouts"]
        nb = np.array([len(r[1]) for r in R])
        cens = np.mean([r[4] == "censored" for r in R])
        losses, known = [], 0
        seen = 0
        for ours, played, ek, ep, why in R:
            for k, u, c in played:
                seen += 1
                bk = best(k)
                if bk is None:
                    continue
                pv = ev[k][1]
                if u in pv or castle_alt(u) in pv:
                    losses.append(bk - pv.get(u, pv.get(castle_alt(u))))
                elif c in ev:
                    losses.append(bk + best(c))  # child eval is from the opponent's view
                else:
                    continue
                known += 1
        losses = np.clip(np.array(losses, float), 0, None)
        exit_ev = []
        for ours, played, ek, ep, why in R:
            e = best(ek)
            if e is None:
                continue
            our_turn = (ep % 2 == 0) == ours  # ours: True = white; even ply = white to move
            exit_ev.append(e if our_turn else -e)
        exit_ev = np.array(exit_ev, float)
        rows.append((name, b["mode"], nb.mean(), np.mean(nb >= 10), cens, known / max(seen, 1),
                     losses.mean() if len(losses) else np.nan, np.mean(losses <= 10) if len(losses) else np.nan,
                     np.mean(losses >= 50) if len(losses) else np.nan,
                     exit_ev.mean() if len(exit_ev) else np.nan, len(exit_ev) / len(R)))
    hdr = ("| book | pick | our book moves / game | games >= 10 book moves | censored | moves with SF eval | "
           "mean cp loss / move | moves <= 10 cp loss | moves >= 50 cp loss | eval at book exit (ours, cp) | exits with SF eval |")
    lines = [hdr, "|" + "---|" * 11]
    for r in sorted(rows, key=lambda r: -r[2]):
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.0%} | {r[4]:.0%} | {r[5]:.0%} | {r[6]:.1f} | {r[7]:.0%} | "
                     f"{r[8]:.1%} | {r[9]:+.0f} | {r[10]:.0%} |")
    txt = "\n".join(lines)
    print(txt)
    if len(sys.argv) > 4:
        open(sys.argv[4], "w").write(txt + "\n")


if __name__ == "__main__":
    main()
