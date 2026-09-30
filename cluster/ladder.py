#!/usr/bin/env python3
"""Winner-stays-on ladder over the sweep nets. A net is 'finished' when nets/N_s2/N_s2.nnue exists and no job
named N is left in the queue. Rules: with no champion, the two earliest finished nets play; afterwards the champion
plays the LATEST finished unplayed net. Loser is out for good; ties (equal score) go to the champion.
State: runs/ladder.json (restart-safe). Log: logs/ladder.out."""
import json, os, re, subprocess, time
X = "/scratch/ralbe/chess_nnue/exp2"; ST = f"{X}/runs/ladder.json"
ALL = [f"W{w}_{s}" for s in ("sf", "edb", "lc0") for w in (512, 1024, 4096)]
NETSF = f"{X}/runs/ladder_nets.txt"  # nets allowed to play now (edit to add the next phase)
nets = lambda: open(NETSF).read().split() if os.path.exists(NETSF) else ALL
st = json.load(open(ST)) if os.path.exists(ST) else {"champ": None, "out": [], "played": [], "matches": []}
save = lambda: json.dump(st, open(ST, "w"), indent=1)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()


def finished():
    r = []
    for n in nets():
        f = f"{X}/nets/{n}_s2/{n}_s2.nnue"
        if os.path.exists(f) and not sh(f"squeue -h -u ralbe -n {n}"):
            r.append((os.path.getmtime(f), n))
    return [n for _, n in sorted(r)]


def score(a, b):
    pgn = open(f"{X}/runs/m_{a}_vs_{b}/games.pgn").read()
    w = {a: 0, b: 0}; d = 0
    for white, black, res in re.findall(r'\[White "(.*?)"\]\s*\[Black "(.*?)"\]\s*\[Result "(.*?)"\]', pgn):
        if res == "1-0": w[white] += 1
        elif res == "0-1": w[black] += 1
        elif res == "1/2-1/2": d += 1
    return w[a], w[b], d


def play(a, b):
    log(f"match {a} (champ) vs {b}")
    for attempt in range(3):  # ~380 games per match (same sample size as the 512 round)
        sh(f"cd {X} && ROUNDS=190 sbatch --export=ALL --wait -J m_{a}_vs_{b} -o logs/match_{a}_vs_{b}.out match.sh {a} {b}")
        try: wa, wb, d = score(a, b); break
        except Exception as e: log("match failed, retry", e)
    else: raise SystemExit("match failed 3 times")
    win, lose = (b, a) if wb > wa else (a, b)
    st["matches"].append({"a": a, "b": b, "a_wins": wa, "b_wins": wb, "draws": d, "winner": win})
    st["champ"] = win; st["out"].append(lose); save()
    log(f"  {a} {wa} - {wb} {b}  draws {d}  -> winner {win}, {lose} eliminated")
    sh(f"cd {X} && python3 report.py")


while True:
    new = [n for n in finished() if n not in st["played"] and n not in st["out"] and n != st["champ"]]
    if st["champ"] is None and len(new) >= 2:
        a, b = new[0], new[1]; st["played"] += [a, b]; save(); play(a, b)
    elif st["champ"] and new:
        b = new[-1]; st["played"].append(b); save(); play(st["champ"], b)
    elif st["champ"] and len(st["out"]) == len(ALL) - 1:
        break
    else:
        time.sleep(120)
log("LADDER DONE, best net:", st["champ"])
