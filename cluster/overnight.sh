#!/bin/bash
# waits for the direct sf-vs-edb match (cancelled at 371 games by its watcher), seeds the ladder with its result,
# opens the ladder to all 9 nets and runs it to the end; runs/RESULTS.md is rewritten after every match.
cd /scratch/ralbe/chess_nnue/exp2
while squeue -u ralbe -h -o "%j" | grep -q '^d_W512_edb'; do sleep 30; done
python3 - <<'PY'
import re, json
a, b = "W512_sf", "W512_edb"
p = open("runs/m_W512_edb_vs_W512_sf_direct/games.pgn").read(); w = {a: 0, b: 0}; d = 0
for wh, bl, r in re.findall(r'\[White "(.*?)"\]\s*\[Black "(.*?)"\]\s*\[Result "(.*?)"\]', p):
    if r == "1-0": w[wh] += 1
    elif r == "0-1": w[bl] += 1
    elif r == "1/2-1/2": d += 1
win, lose = (b, a) if w[b] > w[a] else (a, b)
st = json.load(open("runs/ladder.json"))
st["matches"].append({"a": a, "b": b, "a_wins": w[a], "b_wins": w[b], "draws": d, "winner": win})
st["champ"] = win; st["out"].append(lose); st["played"].append(a)
json.dump(st, open("runs/ladder.json", "w"), indent=1)
print("512 round done:", w, d, "winner", win)
PY
python3 report.py
rm -f runs/ladder_nets.txt   # ladder now sees all 9 nets
python3 ladder.py
python3 report.py
echo OVERNIGHT_DONE
