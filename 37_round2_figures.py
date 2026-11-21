"""Step 37 - figures for the second round (greyscale, 6.5-in width, PNG 300 dpi + PDF).

Figure 8  two-way relay (A: avoidance and handover indices for the foot at the start and the hand at the end of a change,
          with the placebo) and anticipatory coupling (B: slopes of release quantiles on the unpredicted re-press shift)
Figure 9  reference point (A: release dispersion relative to the pedal stroke, the hand's own onset and a surrogate) and
          release prediction (B: error of four predictors)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import FIG, RES

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
K, G1, G2 = "0.1", "0.45", "0.75"
S = json.loads((RES / "round2_summary.json").read_text()); H1 = json.loads((RES / "h1_summary.json").read_text())


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ Figure 8
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
a = ax[0]
items = [("Foot waits for hand\n(start of change)", S["relay"]["A_foot"], S["relay"]["H_foot"]),
         ("Hand waits for foot\n(end of change)", H1["H1a_A"], H1["H1b_H"]),
         ("Placebo: chords\nshifted by 0.5 s", S["relay"]["A_placebo"], S["relay"]["H_placebo"])]
for i, (lab, A, Hh) in enumerate(items):
    for dx, v, mk, fc in ((-0.12, A, "s", K), (0.12, Hh, "o", "white")):
        a.plot([v["ci_lo"], v["ci_hi"]], [i + dx, i + dx], color=K, lw=1.2); a.plot(v["beta"], i + dx, mk, ms=4.5, color=fc, mec=K)
a.axvline(0, color=G1, lw=0.8); a.set_yticks(range(3)); a.set_yticklabels([x[0] for x in items], fontsize=7.5); a.invert_yaxis()
a.plot([], [], "s", color=K, ms=4.5, label="avoidance (log ratio)"); a.plot([], [], "o", color="white", mec=K, ms=4.5, label="handover (log ratio)")
a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=7.5); a.set_xlabel("Observed vs surrogate (log ratio)"); a.set_title("A  A two-way relay", loc="left", fontweight="bold")
b = ax[1]
taus = ["0.1", "0.25", "0.5", "0.75"]; x = np.array([10, 25, 50, 75])
qh = S["ctrl"]["quantiles_habit"]; qs = S["ctrl"]["quantiles_same_position"]
b.errorbar(x - 1.5, [qh[t]["beta"] for t in taus], yerr=[[qh[t]["beta"] - qh[t]["ci_lo"] for t in taus], [qh[t]["ci_hi"] - qh[t]["beta"] for t in taus]],
           fmt="s-", color=K, ms=4, lw=1, capsize=0, label="unpredicted from the pianist's habit")
b.errorbar(x + 1.5, [qs[t]["slope"] for t in taus], yerr=[[qs[t]["slope"] - qs[t]["ci"][0] for t in taus], [qs[t]["ci"][1] - qs[t]["slope"] for t in taus]],
           fmt="o--", color=G1, mfc="white", ms=4, lw=1, capsize=0, label="unpredicted from the score position")
b.axhline(0, color=G2, lw=0.7); b.set_xticks(x); b.set_xticklabels(["10th", "25th", "50th", "75th"]); b.set_ylim(0, 0.85)
b.set_xlabel("Percentile of key-release time"); b.set_ylabel("Release shift per ms of re-press shift")
b.text(22, 0.83, f"10th percentile falls {abs(S['ctrl']['p10_release_minus_press_ms']):.0f} ms\nbefore the re-press", fontsize=7.5, va="top")
b.legend(frameon=False, loc="lower right", fontsize=7.5); b.set_title("B  Even early releases follow the foot", loc="left", fontweight="bold")
fig.tight_layout(); save(fig, "Figure8")

# ------------------------------------------------------------------ Figure 9
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={"width_ratios": [1.2, 1]})
a = ax[0]
th = [int(t) for t in S["ref"]["median_mad_ms"] if t != "onset"]
a.plot(th, [S["ref"]["median_mad_ms"][str(t)] for t in th], "o-", color=K, ms=4, lw=1.2, label="relative to the pedal stroke")
a.plot(th, [S["ref"]["median_sur_ms"][str(t)] for t in th], "s--", color=G1, mfc="white", ms=4, lw=1, label="relative to another change's stroke")
a.axhline(S["ref"]["median_mad_ms"]["onset"], color=K, lw=0.9, ls=":"); a.text(16, S["ref"]["median_mad_ms"]["onset"] + 2, "relative to the hand's own onset", fontsize=7.5, va="bottom")
a.set_xlabel("Controller value passed on the down-stroke"); a.set_ylabel("Dispersion of key release (MAD, ms)")
a.set_ylim(70, max(S["ref"]["median_sur_ms"].values()) * 1.22); a.legend(frameon=False, loc="upper left", fontsize=7.5)
a.set_title("A  Releases are timed to the foot", loc="left", fontweight="bold")
b = ax[1]
labs = [("sound_offset", "Sound\noffset"), ("onset", "Onset\nonly"), ("pedal", "Re-press\nonly"), ("blend", "Onset +\nre-press")]
vals = [S["pred"]["median_mean_abs_error_ms"][k] for k, _ in labs]; med = [S["pred"]["median_mae_ms"][k] for k, _ in labs]
b.bar(range(4), vals, color=[G2, G1, G1, K], edgecolor=K, lw=0.6, width=0.6)
b.plot(range(4), med, "D", color="white", mec=K, ms=4.5, label="median absolute error")
for i, v in enumerate(vals):
    b.text(i, v + 8, f"{v:.0f}", ha="center", fontsize=7.5)
b.set_xticks(range(4)); b.set_xticklabels([x[1] for x in labs], fontsize=7.5); b.set_ylabel("Mean absolute error (ms)")
b.legend(frameon=False, loc="upper right", fontsize=7.5); b.set_title("B  Predicting key releases", loc="left", fontweight="bold")
fig.tight_layout(); save(fig, "Figure9")
print("figures 8-9 written")
