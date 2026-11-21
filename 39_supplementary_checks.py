"""Step 39 - two checks on the second-round results, run before writing the paper (exploratory; requested after step 36).

1. Edition-by-edition replication. The two-way relay (foot waits for hand: A_foot, H_foot; hand waits for foot: A, H) is
   pooled separately within each of the ten competition editions. The anticipatory ("shared plan") slope of the 10th
   release percentile is re-estimated leaving out one edition at a time (same-score-position version) and pooled within
   each edition (habit version).
2. Local-tempo (rubato) control. Local tempo moves both the re-press and the key release. Three tempo covariates are
   computed for every pedal change from the onset clusters around its anchor note: log inter-onset interval to the previous
   cluster, to the next cluster, and log span from two clusters before to two clusters after. Both the unpredicted re-press
   shift and the key release are residualised on them (together with log IOI to the next onset, as in step 36) before the
   quantile slopes are computed (Frisch-Waugh-Lovell). Habit version: within each performance. Same-position version:
   after removing score-position and performance fixed effects from every variable.

Output: results/supplementary_checks.json, results/edition_replication.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import DER, PLAN, RES, clusters, demean2, load_cache, load_meta, pool, save_json

rng = np.random.default_rng(PLAN["seed"] + 39)
M = load_meta().set_index("midi_filename")
OUT = {}
TAUS = (0.10, 0.25, 0.50, 0.75)

# ------------------------------------------------------------------ 1a. relay, edition by edition
R = pd.read_csv(RES / "relay_foot_indices.csv")
rows = []
for yr, g in R.groupby("year"):
    for col in ("A_foot", "H_foot", "A", "H"):
        r = pool(g, col, groups=("pianist", "piece"), B=300)
        rows.append({"year": yr, "index": col, "n": r["n"], "pianists": r["pianists"], "beta": r["beta"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
ED = pd.DataFrame(rows)
OUT["relay_by_edition"] = {c: {"all_editions_expected_sign": bool((np.sign(ED[ED["index"] == c].beta) == (-1 if c.startswith("A") else 1)).all()),
                               "all_cis_exclude_0": bool(((ED[ED["index"] == c].ci_hi < 0) | (ED[ED["index"] == c].ci_lo > 0)).all()),
                               "min": float(ED[ED["index"] == c].beta.min()), "max": float(ED[ED["index"] == c].beta.max())}
                           for c in ("A_foot", "H_foot", "A", "H")}
print("1a. relay by edition:", {k: (v["all_editions_expected_sign"], v["all_cis_exclude_0"], round(v["min"], 2), round(v["max"], 2)) for k, v in OUT["relay_by_edition"].items()})

# ------------------------------------------------------------------ local tempo covariates for every pedal change
C = load_cache()
EV = pd.read_pickle(DER / "handover_events.pkl").sort_values(["midi_filename", "anchor"]).copy()
EV = EV[EV.ioi_next > 0].copy()
parts = []
for f, g in EV.groupby("midi_filename", sort=False):
    on = np.sort(C[f]["on"]); cid = clusters(on, PLAN["cluster_gap_s"]); starts = on[np.r_[True, np.diff(cid) > 0]]
    k = cid[np.clip(np.searchsorted(on, g.anchor.values - 1e-6), 0, len(on) - 1)]
    def at(i):
        return np.where((i >= 0) & (i < len(starts)), starts[np.clip(i, 0, len(starts) - 1)], np.nan)
    parts.append(pd.DataFrame({"midi_filename": f, "event": g.event.values,
                               "l_prev": np.log(at(k) - at(k - 1)), "l_next": np.log(at(k + 1) - at(k)), "l_span": np.log(at(k + 2) - at(k - 2))}))
TC = pd.concat(parts)
for c in ("l_prev", "l_next", "l_span"):
    TC.loc[~np.isfinite(TC[c]), c] = np.nan
EV = EV.merge(TC, on=["midi_filename", "event"], how="left")
TEMPO = ["lioi", "l_prev", "l_next", "l_span"]

# ------------------------------------------------------------------ 2a. habit version (within performance)
EV["x"] = EV.p - EV.anchor; EV["y"] = EV.rel_med - EV.anchor; EV["lioi"] = np.log(EV.ioi_next)
EV["pred"] = EV.groupby("midi_filename").x.transform(lambda s: s.shift(1).rolling(5).mean()); EV["res"] = EV.x - EV.pred
mrows = []
for f, g in EV.dropna(subset=["pred", "y", *TEMPO]).groupby("midi_filename"):
    g = g[g.y < 2.0]
    if len(g) < 30:
        continue
    X = np.column_stack([np.ones(len(g)), g.pred, g.res, *[g[c] for c in TEMPO]]); b = np.linalg.lstsq(X, g.y, rcond=None)[0]
    mrows.append({"midi_filename": f, "slope_res_tempo": b[2]})
MR = pd.DataFrame(mrows).join(M[["pianist", "piece", "year"]], on="midi_filename")
OUT["habit_mean_slope_tempo_controlled"] = pool(MR, "slope_res_tempo", B=300)

NT = pd.read_pickle(DER / "handover_notes.pkl").merge(EV[["midi_filename", "event", "res", *TEMPO]], on=["midi_filename", "event"])
NT["yv"] = NT.off - NT.anchor; NT = NT[NT.yv < 2.0].dropna(subset=["res", *TEMPO]).copy()


def resid_within(g, cols, covs):
    Z = np.column_stack([np.ones(len(g)), *[g[c].values for c in covs]])
    return {c: g[c].values - Z @ np.linalg.lstsq(Z, g[c].values, rcond=None)[0] for c in cols}


qrows = []
for f, g in NT.groupby("midi_filename"):
    if g.event.nunique() < 30:
        continue
    for label, covs in (("lioi_only", ["lioi"]), ("tempo", TEMPO)):
        r = resid_within(g, ["yv", "res"], covs)
        xr = r["res"] if label == "tempo" else g.res.values  # step 36 residualised only the release on lioi
        q = pd.qcut(xr, 5, labels=False, duplicates="drop")
        if len(np.unique(q)) < 4:
            continue
        t = pd.DataFrame({"q": q, "x": xr, "y": r["yv"]})
        for tau in TAUS:
            tt = t.groupby("q").agg(yq=("y", lambda s: s.quantile(tau)), xm=("x", "mean"))
            qrows.append({"midi_filename": f, "control": label, "tau": tau, "slope": float(np.polyfit(tt.xm, tt.yq, 1)[0])})
Q = pd.DataFrame(qrows).join(M[["pianist", "piece", "year"]], on="midi_filename")
OUT["habit_quantile_slopes"] = {lab: {str(t): pool(Q[(Q.control == lab) & (Q.tau == t)], "slope", B=300) for t in TAUS} for lab in ("lioi_only", "tempo")}
print("2a. habit quantile slopes (lioi only -> tempo controlled):",
      {t: (round(OUT["habit_quantile_slopes"]["lioi_only"][str(t)]["beta"], 3), round(OUT["habit_quantile_slopes"]["tempo"][str(t)]["beta"], 3)) for t in TAUS},
      "| mean slope, tempo controlled", round(OUT["habit_mean_slope_tempo_controlled"]["beta"], 3))
q10 = Q[(Q.control == "tempo") & (Q.tau == 0.10)]
erows = []
for yr, g in q10.groupby("year"):
    r = pool(g, "slope", groups=("pianist", "piece"), B=300)
    erows.append({"year": yr, "index": "p10_slope_habit_tempo", "n": r["n"], "pianists": r["pianists"], "beta": r["beta"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
ED = pd.concat([ED, pd.DataFrame(erows)])

# ------------------------------------------------------------------ 2b. same-score-position version
MT = pd.read_pickle(DER / "handover_events_matched.pkl")
MT = MT[MT.groupby("set").midi_filename.transform("nunique") >= 2].copy()
MT = MT.merge(TC, on=["midi_filename", "event"], how="left"); MT["lioi"] = np.log(MT.ioi_next.where(MT.ioi_next > 0))
NS = pd.read_pickle(DER / "handover_notes.pkl").merge(MT[["midi_filename", "event", "set", *TEMPO]], on=["midi_filename", "event"])
NS["x"] = NS.p - NS.anchor; NS["yv"] = NS.off - NS.anchor
NS = NS[NS.yv < 2.0].dropna(subset=TEMPO).join(M[["pianist", "year"]], on="midi_filename").reset_index(drop=True)


def same_position(d, covs):
    """note-level two-way demeaning (set x performance) of x, y and covariates; FWL residualisation on covs; decile slopes."""
    d = d[d.groupby("set").midi_filename.transform("nunique") >= 2]
    D = demean2({c: d[c].values for c in ["x", "yv", *covs]}, d.set.values, d.midi_filename.values)
    x, y = D["x"], D["yv"]
    if covs:
        Z = np.column_stack([D[c] for c in covs]); x = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]; y = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    t = pd.DataFrame({"x": x, "y": y, "pianist": d.pianist.values, "dec": pd.qcut(x, 10, labels=False)})
    return t


def dec_slopes(t):
    out = {}
    for tau in TAUS:
        a = t.groupby("dec").agg(yq=("y", lambda s: s.quantile(tau)), xm=("x", "mean"))
        out[tau] = float(np.polyfit(a.xm, a.yq, 1)[0])
    return out


sp = {}
for label, covs in (("none", []), ("tempo", TEMPO)):
    t = same_position(NS, covs); est = dec_slopes(t)
    pis = t.pianist.unique(); gi = {k: np.where(t.pianist.values == k)[0] for k in pis}; bs = []
    for _ in range(200):
        bs.append(dec_slopes(t.iloc[np.concatenate([gi[k] for k in rng.choice(pis, len(pis))])]))
    sp[label] = {str(tau): {"slope": est[tau], "ci": np.percentile([b[tau] for b in bs], [2.5, 97.5]).tolist()} for tau in TAUS}
OUT["same_position_quantile_slopes"] = sp | {"n_notes": int(len(NS))}
print("2b. same-position decile slopes (none -> tempo):", {t: (round(sp["none"][str(t)]["slope"], 3), round(sp["tempo"][str(t)]["slope"], 3)) for t in TAUS})

loeo = {}
for yr in sorted(NS.year.unique()):
    t = same_position(NS[NS.year != yr], TEMPO); loeo[str(yr)] = dec_slopes(t)[0.10]
OUT["same_position_p10_tempo_leave_one_edition_out"] = {"by_left_out_edition": loeo, "min": min(loeo.values()), "max": max(loeo.values())}
print("2c. leave-one-edition-out p10 slope (tempo controlled):", {k: round(v, 3) for k, v in loeo.items()})

# early releases: share of held notes released before the re-press, and how far before
OUT["early_release"] = {"p10_release_minus_press_ms_median": float(NT.groupby("midi_filename").apply(lambda g: np.percentile(g.off - g.p, 10), include_groups=False).median() * 1000)
                        if "p" in NT else None}
ED.to_csv(RES / "edition_replication.csv", index=False)
OUT["p10_habit_tempo_by_edition"] = {"all_positive": bool((ED[ED["index"] == "p10_slope_habit_tempo"].beta > 0).all()),
                                     "all_cis_exclude_0": bool((ED[ED["index"] == "p10_slope_habit_tempo"].ci_lo > 0).all()),
                                     "min": float(ED[ED["index"] == "p10_slope_habit_tempo"].beta.min()), "max": float(ED[ED["index"] == "p10_slope_habit_tempo"].beta.max())}
print("2d. habit p10 slope (tempo) by edition:", OUT["p10_habit_tempo_by_edition"])
save_json(OUT, RES / "supplementary_checks.json")
