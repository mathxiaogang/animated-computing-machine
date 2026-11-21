"""Step 40 - figures for the paper "Waiting for Each Other" (greyscale, 6.5-in width, PNG 300 dpi + PDF) -> figures/paper/.

Fig1  schematic of one sustain-pedal change: the two waits (typical timings from the event tables)
Fig2  the two-way relay: (A) pedal lifts around chord onsets, (B) key releases around the re-press; observed vs surrogate
Fig3  edition-by-edition replication: avoidance and handover indices for both directions, and the early-release slope
Fig4  = Figure5 (tracking and stroke coupling)      Fig6 = Figure6 (which held notes are cut)
Fig5  timing reference and planning: (A) release dispersion by reference point, (B) quantile slopes with/without tempo control
Fig7  = Figure2 (release under the pedal at the same score position: design ladder and variants)
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import DER, FIG, PLAN, RES, clusters, crossings, kde_curve, load_cache

OUT = FIG / "paper"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False})
K, G1, G2 = "0.1", "0.45", "0.75"
J = lambda n: json.loads((RES / n).read_text())
R2, SC, H1 = J("round2_summary.json"), J("supplementary_checks.json"), J("h1_summary.json")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ Fig 1: schematic
EV = pd.read_pickle(DER / "handover_events.pkl"); NT = pd.read_pickle(DER / "handover_notes.pkl")
t_r = float((EV.r - EV.anchor).median() * 1000); t_p = float((EV.p - EV.anchor).median() * 1000)
t_rel = t_p + float(NT.lag.median() * 1000)
fig, ax = plt.subplots(figsize=(6.5, 2.9))
ax.set_xlim(-330, 420); ax.set_ylim(-0.2, 3.35); ax.axis("off")
yH, yF = 2.35, 0.55
ax.text(-325, yH + 0.35, "Hand", fontweight="bold", va="center"); ax.text(-325, yF + 0.62, "Foot (sustain pedal)", fontweight="bold", va="center")
# hand: previous chord released around the new onset, new chord held through the change
ax.add_patch(plt.Rectangle((-300, yH + 0.05), 290, 0.22, color=G2, lw=0)); ax.text(-160, yH + 0.16, "previous chord", ha="center", va="center", fontsize=7.5)
ax.add_patch(plt.Rectangle((0, yH - 0.25), t_rel, 0.22, color=G1, lw=0)); ax.text(t_rel / 2, yH - 0.14, "new chord: keys held", ha="center", va="center", fontsize=7.5, color="white")
ax.plot([0, 0], [yF - 0.1, yH + 0.45], color=K, lw=0.8, ls=":"); ax.text(0, yH + 0.52, "new chord onset", ha="center", fontsize=7.5)
ax.plot([t_rel, t_rel], [yH - 0.3, yH + 0.02], color=K, lw=1.4); ax.text(t_rel + 6, yH + 0.06, "key release", fontsize=7.5)
# foot: controller trace
tt = np.linspace(-300, 420, 600); down = 110
lift = 1 / (1 + np.exp((tt - t_r) / 6)); press = 1 / (1 + np.exp(-(tt - t_p) / 14))
v = down * np.maximum(lift, press)
ax.plot(tt, yF + v / 127 * 0.9, color=K, lw=1.6)
ax.plot([t_r, t_r], [yF - 0.12, yF + 0.95], color=K, lw=0.8, ls="--"); ax.text(t_r, yF - 0.22, "lift", ha="center", fontsize=7.5)
ax.plot([t_p, t_p], [yF - 0.12, yH - 0.3], color=K, lw=0.8, ls="--"); ax.text(t_p, yF - 0.22, "re-depression", ha="center", fontsize=7.5)
ax.annotate("", xy=(t_r, yF + 1.12), xytext=(0, yF + 1.12), arrowprops=dict(arrowstyle="->", color=K, lw=1.1))
ax.text(-8, yF + 1.12, "1  foot waits for hand", ha="right", va="center", fontsize=8)
ax.annotate("", xy=(t_rel, yH - 0.42), xytext=(t_p, yH - 0.42), arrowprops=dict(arrowstyle="->", color=K, lw=1.1))
ax.text(t_rel + 6, yH - 0.52, "2  hand waits for foot", ha="left", fontsize=8)
ax.text(-300, yF + 0.84, "pedal down: dampers off the strings", fontsize=7, color=G1); ax.text(t_r + 14, yF + 0.22, "dampers on", fontsize=7, color=G1)
for x in (-200, -100, 0, 100, 200, 300, 400):
    ax.plot([x, x], [-0.02, 0.03], color=K, lw=0.6); ax.text(x, -0.15, f"{x}", ha="center", fontsize=7)
ax.plot([-300, 420], [0.0, 0.0], color=K, lw=0.6); ax.text(420, -0.15, "ms", ha="left", fontsize=7)
fig.tight_layout(); save(fig, "Fig1")
SCHEM = {"lift_ms": t_r, "repress_ms": t_p, "release_ms": t_rel}

# ------------------------------------------------------------------ Fig 2: the two-way relay
CACHE = DER / "paper_fig2_curves.npz"
C = {} if CACHE.exists() else load_cache(); rng = np.random.default_rng(PLAN["seed"] + 40)
obs, sur, nlift = [], [], 0
for f, d in C.items():
    pr, rl = crossings(d["cc64"], PLAN["pedal_threshold"])
    if len(rl) < PLAN["min_events"] or len(pr) < 2:
        continue
    j = np.searchsorted(pr, rl); ok = j < len(pr); r = rl[ok]; r = r[(pr[j[ok]] - r) < PLAN["change_max_gap_s"]]
    i = np.searchsorted(pr, r) - 1; has = i >= 0; r = r[has]; pprev = pr[i[has]]
    if len(r) < PLAN["min_events"]:
        continue
    on = d["on"]; cid = clusters(on, PLAN["cluster_gap_s"]); cnt = np.bincount(cid); first = np.r_[True, np.diff(cid) > 0]; ch = on[first][cnt >= 2]

    def pairs(x):
        lo = np.searchsorted(ch, x - 0.45); hi = np.searchsorted(ch, x + 0.35)
        return np.concatenate([x[k] - ch[lo[k]:hi[k]] for k in range(len(x))]) if len(x) else np.array([])
    obs.append(pairs(r)); nlift += len(r)
    hold = r - pprev
    for _ in range(5):
        sur.append(pairs(np.sort(pprev + rng.permutation(hold))))
if CACHE.exists():
    z = np.load(CACHE); c_o, s_o, c_s, s_s, nlift = z["c_o"], z["s_o"], z["c_s"], z["s_s"], int(z["nlift"])
else:
    c_o, s_o = kde_curve(np.concatenate(obs), grid_ms=[-300, 400]); c_s, s_s = kde_curve(np.concatenate(sur), grid_ms=[-300, 400])
    s_o = s_o / nlift * 1000; s_s = s_s / (5 * nlift) * 1000
    np.savez(CACHE, c_o=c_o, s_o=s_o, c_s=c_s, s_s=s_s, nlift=nlift)
cur = pd.read_csv(RES / "h1_density_curves.csv")
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.75), sharey=False)
for a, (x, yo, ys, xlab, pk, ttl) in zip(ax, [(c_o, s_o, s_s, "Pedal lift minus chord onset (ms)", None, "A  The foot waits for the hand"),
                                          (cur.lag_ms.values, cur.real.values * 1000, cur.surrogate.values * 1000, "Key release minus pedal re-depression (ms)", H1["H1c"]["peak_ms"], "B  The hand waits for the foot")]):
    for w in (PLAN["avoid_window_ms"], PLAN["handover_window_ms"]):
        a.axvspan(*w, color=G2, alpha=0.45, lw=0)
    a.plot(x, ys, color=G1, ls="--", lw=1.3, label="surrogate"); a.plot(x, yo, color=K, lw=1.6, label="observed"); a.axvline(0, color=K, lw=0.8)
    ym = max(yo.max(), ys.max()); a.set_ylim(0, ym * 1.28); a.set_xlim(-300, 400)
    pk = float(x[np.argmax(yo)]) if pk is None else pk
    a.plot([pk, pk], [ym * 1.03, ym * 1.1], color=K, lw=0.8); a.text(pk + 8, ym * 1.12, f"peak {pk:.0f} ms", fontsize=7.5)
    a.text(np.mean(PLAN["avoid_window_ms"]), ym * 0.06, "avoid", ha="center", fontsize=7.5); a.text(np.mean(PLAN["handover_window_ms"]) + 2, ym * 0.06, "hand-\nover", ha="center", fontsize=7)
    a.set_xlabel(xlab); a.set_title(ttl.split()[0], loc="left", fontweight="bold", fontsize=10)
ax[0].set_ylabel("Chord onsets per pedal lift\n(rate per s)"); ax[1].set_ylabel("Key releases\n(density per s)")
ax[0].legend(frameon=False, loc="upper left", fontsize=7.5)
fig.tight_layout(); save(fig, "Fig2")
FOOT_CURVE_PEAK = float(c_o[np.argmax(s_o)])

# ------------------------------------------------------------------ Fig 3: edition replication
ED = pd.read_csv(RES / "edition_replication.csv"); yrs = sorted(ED.year.unique()); x = np.arange(len(yrs))
fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.6))
spec = [(ax[0], [("A_foot", "foot waits for hand", "s", K, -0.12), ("A", "hand waits for foot", "o", "white", 0.12)], "A  Avoidance", "log ratio vs surrogate"),
        (ax[1], [("H_foot", "foot waits for hand", "s", K, -0.12), ("H", "hand waits for foot", "o", "white", 0.12)], "B  Handover", "log ratio vs surrogate"),
        (ax[2], [("p10_slope_habit_tempo", "10th percentile", "D", K, 0.0)], "C  Early releases", "slope on unpredicted re-depression")]
for a, series, ttl, ylab in spec:
    for col, lab, mk, fc, dx in series:
        g = ED[ED["index"] == col].set_index("year").loc[yrs]
        a.vlines(x + dx, g.ci_lo, g.ci_hi, color=K, lw=0.9); a.plot(x + dx, g.beta, mk, ms=3.8, color=fc, mec=K, mew=0.8, label=lab, ls="none")
    a.axhline(0, color=G1, lw=0.7); a.set_xticks(x); a.set_xticklabels([str(y)[2:] for y in yrs], fontsize=7); a.set_xlabel("Edition (20xx)")
    a.set_title(ttl.split()[0], loc="left", fontweight="bold", fontsize=10); a.set_ylabel(ylab, fontsize=8)
ax[0].set_ylim(-0.85, 0.05); ax[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.95)); ax[2].set_ylim(0, 0.5)
fig.tight_layout(); save(fig, "Fig3")

# ------------------------------------------------------------------ Fig 5: timing reference and planning
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.85), gridspec_kw={"width_ratios": [1.05, 1]})
a = ax[0]; ref = R2["ref"]
th = [int(t) for t in ref["median_mad_ms"] if t != "onset"]
a.plot(th, [ref["median_mad_ms"][str(t)] for t in th], "o-", color=K, ms=3.8, lw=1.2, label="pedal stroke of the same change")
a.plot(th, [ref["median_sur_ms"][str(t)] for t in th], "s--", color=G1, mfc="white", ms=3.8, lw=1, label="pedal stroke of another change")
a.axhline(ref["median_mad_ms"]["onset"], color=K, lw=0.9, ls=":"); a.text(16, ref["median_mad_ms"]["onset"] + 1.5, "the hand's own onset", fontsize=7.5, va="bottom")
a.set_xlabel("Reference: controller value passed on the down-stroke"); a.set_ylabel("Release dispersion (median MAD, ms)")
a.set_ylim(70, 162); a.legend(frameon=False, loc="upper left", fontsize=7, title="release timed relative to ...", title_fontsize=7)
a.set_title("A", loc="left", fontweight="bold", fontsize=10)
b = ax[1]; taus = ["0.1", "0.25", "0.5", "0.75"]; xq = np.array([10, 25, 50, 75])
hq = SC["habit_quantile_slopes"]; sq = SC["same_position_quantile_slopes"]
sets = [(hq["lioi_only"], "beta", "ci_lo", "ci_hi", "s", "white", G1, -2.4, "habit, raw"),
        (hq["tempo"], "beta", "ci_lo", "ci_hi", "s", K, K, -0.8, "habit, tempo-controlled"),
        (sq["none"], "slope", None, None, "o", "white", G1, 0.8, "position, raw"),
        (sq["tempo"], "slope", None, None, "o", K, K, 2.4, "position, tempo-controlled")]
for d, kb, klo, khi, mk, fc, lc, dx, lab in sets:
    bb = np.array([d[t][kb] for t in taus])
    lo = np.array([d[t][klo] if klo else d[t]["ci"][0] for t in taus]); hi = np.array([d[t][khi] if khi else d[t]["ci"][1] for t in taus])
    b.vlines(xq + dx, lo, hi, color=lc, lw=0.9); b.plot(xq + dx, bb, marker=mk, mfc=fc, mec=K, mew=0.8, ms=3.8, label=lab, ls="-" if fc == K else ":", color=lc, lw=0.8)
b.set_xticks(xq); b.set_xticklabels(["10th", "25th", "50th", "75th"]); b.set_ylim(0, 1.12); b.axhline(0, color=G2, lw=0.7)
b.set_xlabel("Percentile of key-release time"); b.set_ylabel("Release shift per ms of\nunpredicted re-depression shift")
b.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6)
b.set_title("B", loc="left", fontweight="bold", fontsize=10)
fig.tight_layout(); save(fig, "Fig5")

for src, dst in (("Figure5", "Fig4"), ("Figure6", "Fig6"), ("Figure2", "Fig7")):
    for ext in ("png", "pdf"):
        shutil.copy(FIG / f"{src}.{ext}", OUT / f"{dst}.{ext}")
(RES / "paper_figure_values.json").write_text(json.dumps({"schematic": SCHEM, "foot_correlogram_peak_ms": FOOT_CURVE_PEAK, "n_lifts": nlift}, indent=2))
print("paper figures written", SCHEM, "foot correlogram peak", FOOT_CURVE_PEAK, "lifts", nlift)
