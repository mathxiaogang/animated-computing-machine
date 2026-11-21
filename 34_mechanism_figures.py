"""Step 34 - figures for the handover mechanisms (directions 1-4), greyscale, 6.5-in width, PNG 300 dpi + PDF.

Figure 5  tracking (D1) and stroke coupling (D2): binned curves with the slopes of both designs
Figure 6  what gets cut (D3): raw cut rates and within-chord effects
Figure 7  two-process model (D4): distributions of the per-performance parameters
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import FIG, RES

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
K, G1, G2 = "0.1", "0.45", "0.75"
S = json.loads((RES / "mechanisms_summary.json").read_text())


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ Figure 5
b1 = pd.read_csv(RES / "d1_binned.csv"); b2 = pd.read_csv(RES / "d2_binned.csv")
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.8))
a = ax[0]
lim = [-200, 380]
a.plot(lim, lim, color=G2, lw=0.8, ls=":"); a.text(225, 330, "slope 1", fontsize=7.5, color=G1, ha="right", va="center")
a.plot(b1.xc, b1.yc, "o-", color=K, ms=3.5, lw=1)
a.axhline(0, color=G2, lw=0.6); a.axvline(0, color=G2, lw=0.6)
a.set_xlabel("Re-depression later than usual (ms)")
a.text(0.97, 0.05, f"slope within performance {S['D1']['within']['beta']:.2f}\nslope at same score position {S['D1']['same_position']['b']:.2f}", transform=a.transAxes, ha="right", va="bottom", fontsize=7.5); a.set_ylabel("Key release later than usual (ms)")
a.set_title("A", loc="left", fontweight="bold")
b = ax[1]
xs = np.array([20, 220])
b.plot(xs, xs + (b2.y_ms.iloc[0] - b2.D_ms.iloc[0]), color=G2, lw=0.8, ls=":")
b.plot(xs, np.full(2, b2.y_ms.iloc[0]), color=G2, lw=0.8, ls="--")
b.text(215, xs[1] + (b2.y_ms.iloc[0] - b2.D_ms.iloc[0]) - 8, "locked to end", fontsize=7.5, color=G1, ha="right", va="top")
b.text(22, b2.y_ms.iloc[0] - 6, "locked to start", fontsize=7.5, color=G1, ha="left", va="top")
b.plot(b2.D_ms, b2.y_ms, "o-", color=K, ms=3.5, lw=1)
b.set_xlabel("Pedal down-stroke duration, 32 to 96 (ms)")
b.text(0.97, 0.14, f"slope within performance {S['D2']['within']['beta']:.2f}\nslope at same score position {S['D2']['same_position']['b']:.2f}", transform=b.transAxes, ha="right", va="bottom", fontsize=7.5); b.set_ylabel("Key release after stroke start (ms)")
b.set_title("B", loc="left", fontweight="bold")
fig.tight_layout(); save(fig, "Figure5")

# ------------------------------------------------------------------ Figure 6
rates = pd.DataFrame(S["D3"]["rates"], columns=["need", "doubled", "rate", "n"])
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw={"width_ratios": [1, 1.25]})
a = ax[0]
for j, dbl in enumerate([0, 1]):
    r = rates[rates.doubled == dbl].sort_values("need")
    a.bar(np.arange(2) + (j - 0.5) * 0.36, r.rate * 100, width=0.34, color=[G1, G2][j], edgecolor=K, lw=0.6,
          label=["pitch class not doubled", "pitch class doubled"][j])
a.set_xticks([0, 1]); a.set_xticklabels(["finger free", "finger needed\nfor a nearby note"])
a.set_ylabel("Held notes cut before re-depression (%)"); a.set_ylim(0, 64); a.legend(frameon=False, loc="upper left"); a.set_title("A", loc="left", fontweight="bold")
b = ax[1]
lp = S["D3"]["lpm"]
items = [("Finger needed", lp["need"]), ("Pitch class doubled", lp["doubled"]), ("Velocity +10", lp["vel"]),
         ("Inner voice (vs top)", lp["inner"]), ("Bottom voice (vs top)", lp["bottom"]), ("Register +1 octave", lp["register"])]
for i, (lab, v) in enumerate(items):
    b.plot([100 * (v["b"] - 1.96 * v["se"]), 100 * (v["b"] + 1.96 * v["se"])], [i, i], color=K, lw=1.2); b.plot(100 * v["b"], i, "o", color=K, ms=4)
b.axvline(0, color=G1, lw=0.8); b.set_yticks(range(len(items))); b.set_yticklabels([x[0] for x in items]); b.invert_yaxis()
b.set_xlabel("Change in cut probability, same chord (percentage points)"); b.set_title("B", loc="left", fontweight="bold")
fig.tight_layout(); save(fig, "Figure6")

# ------------------------------------------------------------------ Figure 7
R = pd.read_csv(RES / "d4_mixture_parameters.csv")
fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.2))
for a, (c, lab, rng_) in zip(ax, [("w", "Share locked to the foot, w", (0, 1)), ("mu", "Latency after re-depression, mu (ms)", (-100, 200)),
                                  ("sigma", "Spread, sigma (ms)", (0, 150))]):
    a.hist(R[c], bins=30, range=rng_, color=G1, edgecolor="white", lw=0.4); a.axvline(R[c].median(), color=K, lw=1)
    a.set_xlabel(lab); a.set_yticks([])
    a.text(0.97, 0.95, f"median {R[c].median():.2f}" if c == "w" else f"median {R[c].median():.0f}", transform=a.transAxes, ha="right", va="top", fontsize=7.5)
ax[0].set_ylabel("Performances")
fig.tight_layout(); save(fig, "Figure7")
print("figures 5-7 written")
