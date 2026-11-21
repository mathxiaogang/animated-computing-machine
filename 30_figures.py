"""Step 30 - figures (greyscale, sans serif 8-14 pt at a 6.5-in width; PNG 300 dpi + PDF).

Usage:
    python 30_figures.py

Figure 1  hand-foot handover: release-minus-re-press density (real vs surrogate) and per-performance indices (H1)
Figure 2  where the "later release under the pedal" goes: the design ladder and the study 2 variants (H2a)
Figure 3  specificity: release and onset effects with equivalence bounds (H2b, H3a, H3b)
Figure 4  individual differences in the handover (E2)
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
from release_common import FIG, PLAN, RES

FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
K, G1, G2 = "0.1", "0.45", "0.75"
J = lambda n: json.loads((RES / n).read_text())


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ Figure 1
cur = pd.read_csv(RES / "h1_density_curves.csv"); R = pd.read_csv(RES / "h1_performance_indices.csv"); h1 = J("h1_summary.json")
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
a = ax[0]
for w in (PLAN["avoid_window_ms"], PLAN["handover_window_ms"]):
    a.axvspan(*w, color=G2, alpha=0.45, lw=0)
a.plot(cur.lag_ms, cur.surrogate * 1000, color=G1, ls="--", lw=1.3, label="Surrogate (onset-locked)")
a.plot(cur.lag_ms, cur.real * 1000, color=K, lw=1.6, label="Observed")
a.axvline(0, color=K, lw=0.8)
pk = h1["H1c"]["peak_ms"]; ymax = cur.real.max() * 1000
a.plot([pk, pk], [ymax * 1.02, ymax * 1.08], color=K, lw=0.8); a.text(pk - 6, ymax * 1.10, f"peak {pk:.0f} ms", ha="right", fontsize=7.5)
a.text(np.mean(PLAN["avoid_window_ms"]), ymax * 0.12, "avoid", ha="center", fontsize=7.5); a.text(np.mean(PLAN["handover_window_ms"]), ymax * 0.12, "hand-\nover", ha="center", fontsize=7.5)
a.set_ylim(0, ymax * 1.2)
a.set_xlabel("Key release minus pedal re-press (ms)"); a.set_ylabel("Density (per 1,000 ms)"); a.set_xlim(-300, 400)
a.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=2, handlelength=1.8); a.set_title("A", loc="left", fontweight="bold")
b = ax[1]
b.scatter(R.A, R.H, s=4, color=K, alpha=0.35, lw=0)
b.axhline(0, color=G1, lw=0.7); b.axvline(0, color=G1, lw=0.7)
b.set_xlabel("Avoidance index A (log ratio)"); b.set_ylabel("Handover index H (log ratio)")
b.text(0.97, 0.05, f"{(R.A < 0).mean():.0%} of performances A < 0\n{(R.H > 0).mean():.0%} H > 0", transform=b.transAxes, ha="right", fontsize=7.5)
b.set_title("B", loc="left", fontweight="bold")
fig.tight_layout(); save(fig, "Figure1")

# ------------------------------------------------------------------ Figure 2
lad = J("study2_design_ladder.json"); s1 = J("study1_summary.json")
rows = [("Study 1: same performance,\nmatched strata (304 perf.)", s1["H2a_loc_ms"]["beta"], s1["H2a_loc_ms"]["ci_lo"], s1["H2a_loc_ms"]["ci_hi"]),
        ("Aligned transitions,\nsame performance only", lad["performance_FE_only_all_aligned"]["b"], None, lad["performance_FE_only_all_aligned"]["se"]),
        ("Same score position only", lad["position_FE_only_mixed_sets"]["b"], None, lad["position_FE_only_mixed_sets"]["se"]),
        ("Same position + same\nperformance, all sets", lad["two_way_FE_all_sets"]["b"], None, lad["two_way_FE_all_sets"]["se"]),
        ("Study 2: same position +\nsame performance (main)", J("study2_summary.json")["H2a_loc_ms"]["b"], *J("study2_summary.json")["H2a_loc_ms"]["ci"])]
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
a = ax[0]
for i, (lab, est, lo, hi) in enumerate(rows):
    if lo is None:
        lo, hi = est - 1.96 * hi, est + 1.96 * hi
    a.plot([lo, hi], [i, i], color=K, lw=1.2); a.plot(est, i, "o", ms=4, color=K if i in (0, 4) else "white", mec=K)
a.axvline(0, color=G1, lw=0.8); a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows], fontsize=7.5); a.invert_yaxis()
a.set_xlabel("Pedal down minus up: key overlap (ms)"); a.set_title("A", loc="left", fontweight="bold")
b = ax[1]
names = [("", "main"), ("_thr32", "cut-off 32"), ("_thr96", "cut-off 96"), ("_win025", "window 0.25 s"), ("_win060", "window 0.6 s"),
         ("_separated", "separated voice"), ("_interval12", "intervals <= 12"), ("_kot1000", "|KOT| <= 1 s")]
for i, (tag, lab) in enumerate(names):
    s = J(f"study2_summary{tag}.json")["H2a_loc_ms"]
    b.plot(s["ci"], [i, i], color=K, lw=1.2); b.plot(s["b"], i, "o", ms=4, color=K if i == 0 else "white", mec=K)
b.axvline(0, color=G1, lw=0.8); b.set_yticks(range(len(names))); b.set_yticklabels([n[1] for n in names], fontsize=7.5); b.invert_yaxis()
b.set_xlabel("Study 2 effect (ms), piece bootstrap CI"); b.set_title("B", loc="left", fontweight="bold")
fig.tight_layout(); save(fig, "Figure2")

# ------------------------------------------------------------------ Figure 3
sp = J("specificity_summary.json"); s2 = J("study2_summary.json")
items = [("Release variability, sustain\n(H2b, log scale ratio)", s1["H2b_logscale"]["beta"], s1["H2b_logscale"]["ci_lo"], s1["H2b_logscale"]["ci_hi"],
          s2["H2b_logscale"]["b"], s2["H2b_logscale"]["b"] - 1.96 * s2["H2b_logscale"]["se"], s2["H2b_logscale"]["b"] + 1.96 * s2["H2b_logscale"]["se"]),
         ("Chord onset spread, sustain\n(H3a, log ratio)", sp["H3a_study1"]["beta"], sp["H3a_study1"]["ci_lo"], sp["H3a_study1"]["ci_hi"],
          sp["H3a_study2"]["b"], sp["H3a_study2"]["b"] - 1.96 * sp["H3a_study2"]["se"], sp["H3a_study2"]["b"] + 1.96 * sp["H3a_study2"]["se"]),
         ("Release variability, una corda\n(H3b, log scale ratio)", sp["H3b_study1"]["logscale"]["beta"], sp["H3b_study1"]["logscale"]["ci_lo"], sp["H3b_study1"]["logscale"]["ci_hi"],
          sp["H3b_study2"]["logscale"]["b"], sp["H3b_study2"]["logscale"]["b"] - 1.96 * sp["H3b_study2"]["logscale"]["se"], sp["H3b_study2"]["logscale"]["b"] + 1.96 * sp["H3b_study2"]["logscale"]["se"])]
fig, ax = plt.subplots(figsize=(6.5, 2.5))
bound = PLAN["sesoi_log_scale"]
ax.axvspan(-bound, bound, color=G2, alpha=0.5, lw=0); ax.axvline(0, color=G1, lw=0.8)
for i, (lab, e1, l1, h1_, e2, l2, h2) in enumerate(items):
    ax.plot([l1, h1_], [i - 0.15, i - 0.15], color=K, lw=1.2); ax.plot(e1, i - 0.15, "s", ms=4, color=K)
    ax.plot([l2, h2], [i + 0.15, i + 0.15], color=K, lw=1.2); ax.plot(e2, i + 0.15, "o", ms=4, color="white", mec=K)
ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items], fontsize=7.5); ax.invert_yaxis()
ax.plot([], [], "s", color=K, ms=4, label="Study 1: same performance"); ax.plot([], [], "o", color="white", mec=K, ms=4, label="Study 2: same score position")
ax.legend(frameon=False, loc="lower right"); ax.set_xlabel("Pedal down minus up (log ratio); grey band = equivalence bounds (+-10%)")
fig.tight_layout(); save(fig, "Figure3")

# ------------------------------------------------------------------ Figure 4
rng = np.random.default_rng(1)
g = R.groupby("pianist").H.apply(list); g = g[g.apply(len) >= 2]
a_, b_ = [], []
for v in g:
    v = rng.permutation(v); h = len(v) // 2; a_.append(np.mean(v[:h])); b_.append(np.mean(v[h:]))
ind = J("individuality_summary.json")
pm = R.groupby("pianist")[["H", "sus_frac"]].mean()
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.7))
ax[0].scatter(a_, b_, s=8, color=K, alpha=0.6, lw=0); ax[0].set_xlabel("Handover H, one half of performances"); ax[0].set_ylabel("Other half")
ax[0].text(0.03, 0.95, f"Spearman-Brown reliability {ind['E2']['handover_H']['split_half']['spearman_brown']:.2f}\npianist share of variance {ind['E2']['handover_H']['variance_shares']['pianist']:.0%}",
           transform=ax[0].transAxes, va="top", fontsize=7.5); ax[0].set_title("A", loc="left", fontweight="bold")
ax[1].scatter(pm.sus_frac * 100, pm.H, s=8, color=K, alpha=0.6, lw=0); ax[1].set_xlabel("Sustain pedal down (% of time), pianist mean"); ax[1].set_ylabel("Handover H, pianist mean")
ax[1].text(0.03, 0.95, f"Spearman r = {ind['E2']['pianist_H_vs_sustain_share']['r']:.2f}", transform=ax[1].transAxes, va="top", fontsize=7.5)
ax[1].set_title("B", loc="left", fontweight="bold")
fig.tight_layout(); save(fig, "Figure4")
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
