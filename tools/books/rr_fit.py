"""
Joint Elo ratings from round-robin PGNs (Bradley-Terry / Ordo-style maximum likelihood, draws = half points).

python rr_fit.py ANCHOR PGN [PGN ...]     ratings relative to ANCHOR (= 0), 95% intervals from 300 bootstrap
resamples of the games, plus P(best) = share of resamples in which
the engine has the top rating.
"""
import re
import sys
import numpy as np

GAME = re.compile(r'\[White "([^"]+)"\]\s*\[Black "([^"]+)"\]\s*\[Result "(1-0|0-1|1/2-1/2)"\]')


def fit(names, w, b, s, iters=3000):
    n = len(names)
    r = np.zeros(n)
    for _ in range(iters):  # gradient ascent on the log-likelihood of scores s under p = 1/(1+10^(-(r_w-r_b)/400))
        p = 1 / (1 + 10 ** (-(r[w] - r[b]) / 400))
        g = np.zeros(n)
        np.add.at(g, w, s - p)
        np.add.at(g, b, p - s)
        cnt = np.bincount(np.concatenate([w, b]), minlength=n)
        step = 400 * g / np.maximum(cnt, 1)
        r += step
        r -= r.mean()
        if np.abs(step).max() < 1e-4:
            break
    return r


def main():
    anchor = sys.argv[1]
    games = []
    for f in sys.argv[2:]:
        games += GAME.findall(open(f, errors="replace").read())
    names = sorted({x for g in games for x in g[:2]})
    idx = {n: i for i, n in enumerate(names)}
    w = np.array([idx[g[0]] for g in games])
    b = np.array([idx[g[1]] for g in games])
    s = np.array([{"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[g[2]] for g in games])
    a = idx[anchor]
    r = fit(names, w, b, s)
    r -= r[a]
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(300):
        sel = rng.integers(0, len(games), len(games))
        rb = fit(names, w[sel], b[sel], s[sel], iters=800)
        boot.append(rb - rb[a])
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    pbest = np.bincount(boot.argmax(1), minlength=len(names)) / len(boot)
    ngames = np.bincount(np.concatenate([w, b]), minlength=len(names))
    score = (np.bincount(w, s, len(names)) + np.bincount(b, 1 - s, len(names))) / np.maximum(ngames, 1)
    print(f"{len(games)} games, anchor {anchor} = 0\n")
    print("| rank | book | Elo | 95% interval | games | score | P(best) |\n|---|---|---|---|---|---|---|")
    for k, i in enumerate(np.argsort(-r)):
        print(f"| {k+1} | {names[i]} | {r[i]:+.0f} | [{lo[i]:+.0f}, {hi[i]:+.0f}] | {ngames[i]} | {score[i]:.1%} | {pbest[i]:.0%} |")


if __name__ == "__main__":
    main()
