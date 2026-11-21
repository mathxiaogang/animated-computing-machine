"""Step 27 - H3 specificity and E1 dose-response.

H3a  onset precision: chord spread (onset clusters of >= 3 notes within 60 ms) with the sustain pedal down vs up. Onsets are
     audible in every pedal state, so "control follows audibility" predicts no difference (equivalence within +-10% of spread).
H3b  una corda control: release timing with the una corda down vs up, sustain state held fixed. The una corda moves the action,
     not the dampers, so release timing should not change (equivalence within +-10 ms and +-10% scale).
E1   dose-response: full sustain (controller >= 100 throughout) vs partial (64-99) against pedal up.

Usage:
    python 27_specificity.py part=study1|study2
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import (DER, PLAN, RES, add_strata, calibrate_scale, fe_location_scale, fe_regress, load_meta, matched_effect,
                            pool, save_json, tost)

OPTS = dict(a.split("=", 1) for a in sys.argv[1:]); PART = OPTS.get("part", "study1")
M = load_meta().set_index("midi_filename")
LOC_B, SC_B = PLAN["sesoi_location_ms"], PLAN["sesoi_log_scale"]
out_path = RES / "specificity_summary.json"
S = __import__("json").loads(out_path.read_text()) if out_path.exists() else {}


def chord_effect(g, min_n=PLAN["min_per_cell"]):
    """Weighted mean over strata (notes x velocity x pitch span) of the median log-spread difference, pedal down minus up."""
    g = g[g.ped.isin([0, 1])]
    vals, w = [], []
    for _, c in g.groupby(["s_n", "s_vel", "s_span"]):
        a = c[c.ped == 1].lspread.values; b = c[c.ped == 0].lspread.values
        if len(a) >= min_n and len(b) >= min_n:
            vals.append(np.median(a) - np.median(b)); w.append(min(len(a), len(b)))
    return (float(np.average(vals, weights=w)), len(w)) if w else (None, 0)


if PART == "study1":
    # ---------------- H3a, within performance
    CH = pd.read_pickle(DER / "chords.pkl")
    CH["lspread"] = np.log(CH.spread * 1000 + 1)  # log of spread in ms (+1 ms)
    CH["s_n"] = np.clip(CH.n, 3, 5); CH["s_vel"] = pd.cut(CH.vel, PLAN["strata"]["vel"], labels=False)
    CH["s_span"] = pd.cut(CH.top - CH.bot, [-1, 11, 24, 128], labels=False)
    rows = []
    for f, g in CH.groupby("midi_filename"):
        e, k = chord_effect(g)
        if e is not None:
            rows.append({"midi_filename": f, "lspread_diff": e, "n_strata": k, "median_spread_ms": float(g.spread.median() * 1000)})
    R = pd.DataFrame(rows).join(M[["pianist", "piece", "year"]], on="midi_filename")
    R.to_csv(RES / "h3a_performance_effects.csv", index=False)
    p = pool(R, "lspread_diff")
    S["H3a_study1"] = p | {"tost_p": tost(p["beta"], p["se"], SC_B), "median_spread_ms": float(R.median_spread_ms.median())}
    print("H3a study1: perf", len(R), "| log spread diff", round(p["beta"], 4), "ci", [round(p["ci_lo"], 4), round(p["ci_hi"], 4)], "| TOST p", round(S["H3a_study1"]["tost_p"], 4))

    # ---------------- H3b, within performance, sustain state as an extra stratum
    T = add_strata(pd.read_pickle(DER / "transitions_main.pkl"))
    T = T[T.ped.isin([0, 1]) & T.soft.isin([0, 1])]
    rows = []
    for f, g in T.groupby("midi_filename"):
        e = matched_effect(g, cond_col="soft", extra_strata=("ped",))
        if e:
            rows.append({"midi_filename": f, **e})
    R = pd.DataFrame(rows).join(M[["pianist", "piece", "year"]], on="midi_filename"); R["loc_ms"] = R["loc"] * 1000
    R.to_csv(RES / "h3b_performance_effects.csv", index=False)
    pl, ps = pool(R, "loc_ms"), pool(R, "logscale")
    S["H3b_study1"] = {"loc_ms": pl | {"tost_p": tost(pl["beta"], pl["se"], LOC_B)}, "logscale": ps | {"tost_p": tost(ps["beta"], ps["se"], SC_B)}}
    print("H3b study1: perf", len(R), "| loc ms", round(pl["beta"], 2), [round(pl["ci_lo"], 2), round(pl["ci_hi"], 2)], "TOST p", round(S["H3b_study1"]["loc_ms"]["tost_p"], 4),
          "| logscale", round(ps["beta"], 3), [round(ps["ci_lo"], 3), round(ps["ci_hi"], 3)], "TOST p", round(S["H3b_study1"]["logscale"]["tost_p"], 4))

    # ---------------- E1 dose-response (within performance)
    T = add_strata(pd.read_pickle(DER / "transitions_main.pkl"))
    up = T.ped_max < PLAN["pedal_threshold"]
    full = T.ped_min >= 100; partial = (T.ped_min >= PLAN["pedal_threshold"]) & (T.ped_max < 100)
    res = {}
    for lab, m in [("full", full), ("partial", partial)]:
        U = T[up | m].copy(); U["dose"] = m[up | m].astype(int)
        rows = []
        for f, g in U.groupby("midi_filename"):
            e = matched_effect(g, cond_col="dose")
            if e:
                rows.append({"midi_filename": f, **e})
        R = pd.DataFrame(rows).join(M[["pianist", "piece", "year"]], on="midi_filename"); R["loc_ms"] = R["loc"] * 1000
        res[lab] = {"n_transitions_down": int(m.sum()), "loc_ms": pool(R, "loc_ms", B=300), "logscale": pool(R, "logscale", B=300)}
        print("E1", lab, "transitions", int(m.sum()), "| perf", len(R), "| loc ms", round(res[lab]["loc_ms"]["beta"], 2), "| logscale", round(res[lab]["logscale"]["beta"], 3))
    S["E1_dose"] = res

elif PART == "study2":
    # ---------------- H3a at the same score position
    DC = pd.read_pickle(DER / "matched_chords.pkl"); DC = DC[DC.ped.isin([0, 1])].copy()
    DC = DC[DC.groupby("set").ped.transform("nunique") == 2].join(M[["pianist"]], on="midi_filename")
    DC["lspread"] = np.log(DC.spread * 1000 + 1); DC["vel_c"] = (DC.vel - DC.vel.mean()) / 10
    f = fe_regress(DC.lspread.values, {"ped": DC.ped.values.astype(float), "vel_c": DC.vel_c.values}, DC.set.values, DC.midi_filename.values,
                   DC.pianist.values, DC.piece.values)
    S["H3a_study2"] = f["ped"] | {"tost_p": tost(f["ped"]["b"], f["ped"]["se"], SC_B), "n": len(DC), "n_sets": int(DC.set.nunique())}
    print("H3a study2: chords", len(DC), "sets", DC.set.nunique(), "| log spread diff", round(f["ped"]["b"], 4), "se", round(f["ped"]["se"], 4), "TOST p", round(S["H3a_study2"]["tost_p"], 4))
    # ---------------- H3b at the same score position: una corda and sustain in one model
    D = pd.read_pickle(DER / "matched_transitions.pkl")
    D = D[D.ped.isin([0, 1]) & D.soft.isin([0, 1])].copy()
    D = D[D.groupby("set").soft.transform("nunique") == 2].join(M[["pianist"]], on="midi_filename")
    D["kot_ms"] = D.kot * 1000; D["lioi"] = np.log(D.ioi) - np.log(D.ioi).mean(); D["vel_c"] = (D.vel - D.vel.mean()) / 10
    X = {"soft": D.soft.values.astype(float), "ped": D.ped.values.astype(float), "lioi": D.lioi.values, "vel_c": D.vel_c.values}
    loc, sc = fe_location_scale(D.kot_ms.values, X, D.set.values, D.midi_filename.values, D.pianist.values, D.piece.values)
    cal = calibrate_scale(X, D.set.values, D.midi_filename.values, D.pianist.values, focus="soft", sigma=float(np.std(loc["_resid"])), reps=6, seed=3)
    b = (sc["soft"]["b"] - cal["intercept"]) / cal["slope"]; se = sc["soft"]["se"] / cal["slope"]
    S["H3b_study2"] = {"n": len(D), "n_sets": int(D.set.nunique()), "loc_ms": loc["soft"] | {"tost_p": tost(loc["soft"]["b"], loc["soft"]["se"], LOC_B)},
                       "logscale": {"b": b, "se": se, "raw": sc["soft"]["b"], "tost_p": tost(b, se, SC_B), "calibration": cal},
                       "sustain_in_same_model_loc_ms": loc["ped"]}
    print("H3b study2: transitions", len(D), "sets", D.set.nunique(), "| una corda loc ms", round(loc["soft"]["b"], 2), "se", round(loc["soft"]["se"], 2),
          "TOST p", round(S["H3b_study2"]["loc_ms"]["tost_p"], 4), "| scale", round(b, 3), "se", round(se, 3), "TOST p", round(S["H3b_study2"]["logscale"]["tost_p"], 4),
          "| sustain loc ms in same model", round(loc["ped"]["b"], 2))
save_json(S, out_path)
