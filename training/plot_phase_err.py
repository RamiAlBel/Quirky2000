"""Plot runs/phase_err/tables.npy from net_phase_err.py: cp-MAE vs game phase, 95% CI bands."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1]
T = np.load(f"{d}/tables.npy", allow_pickle=True).item()
NETS = [("W512_sf (yours)", "#2a78d6"), ("lc0 no-threats", "#eb6834"), ("lc0 threats", "#1baf7a")]
PANELS = [("lc0|ply", "Leela reference (held-out Jul 2024)", "game ply"),
          ("lc0|pieces", "Leela reference (held-out Jul 2024)", "pieces on board"),
          ("sf|pieces", "Stockfish reference (Lichess eval DB, depth ≥ 30)", "pieces on board")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, (key, title, xl) in zip(axs, PANELS):
    rows = T[key]
    x = np.arange(len(rows))
    for lab, col in NETS:
        m = np.array([r[lab][0] for r in rows]); lo = np.array([r[lab][1] for r in rows]); hi = np.array([r[lab][2] for r in rows])
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(x, m, color=col, lw=2, marker="o", ms=8, mec="white", mew=1.5, label=lab)
    ax.set_xticks(x, [r["bin"] for r in rows], rotation=0, fontsize=8.5)
    ax.set_xlabel(xl + ("  (opening → endgame)" if "pieces" in key else ""))
    ax.set_title(title, fontsize=10.5, color="#222")
    ax.grid(axis="y", color="#ddd", lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors="#555")
axs[0].set_ylabel("mean |eval error| (cp, SF units, after per-net scale fit)")
axs[0].legend(frameon=False, loc="upper left")
fig.suptitle("Static-eval error by game phase — lower is better; bands = 95% CI (bootstrap over games)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{d}/phase_err.png", dpi=130)
print(f"{d}/phase_err.png")
