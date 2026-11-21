"""Step 26 - H2 models.

part=study1: within-performance matched strata (location = median KOT difference, scale = log MAD ratio), pooled across
             performances with crossed random intercepts for pianist, piece and edition (REML) + pianist-cluster bootstrap.
part=study2: transitions at the same score position across performances (step 25): two-way fixed effects (transition set x
             performance) location-scale model with calibrated scale, two-way clustered SEs and a piece-cluster bootstrap.

Usage:
    python 26_models.py part=study1 [variant=main]
    python 26_models.py part=study2 [variant=main]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import (DER, PLAN, RES, add_strata, calibrate_scale, cluster_bootstrap_fe, fe_location_scale, load_meta,
                            matched_effect, pool, save_json)

OPTS = dict(a.split("=", 1) for a in sys.argv[1:])
PART = OPTS.get("part", "study1"); VAR = OPTS.get("variant", "main")
SFX = "" if VAR == "main" else f"_{VAR}"
B = int(OPTS.get("B", PLAN["bootstrap_B"]))
M = load_meta().set_index("midi_filename")

if PART == "study1":
    plan = {**PLAN, "strata": dict(PLAN["strata"])}
    if VAR == "interval12":  # the interval strata must reach the wider interval range
        plan["strata"]["interval"] = [0, 2, 12]
    if VAR == "bottom":      # bass voice: register strata below C4
        plan["strata"]["pitch"] = [0, 47, 59]
    T = add_strata(pd.read_pickle(DER / f"transitions_{VAR}.pkl"), plan)
    if OPTS.get("subset") == "no_baroque":
        T = T[T.midi_filename.map(M.era) != "Baroque"]; SFX += "_nobaroque"
    if OPTS.get("subset") == "no_transcription_short":
        keep = (~M.is_transcription.astype(bool)) & (M.duration >= 120)
        T = T[T.midi_filename.map(keep)]; SFX += "_notrans_short"
    rows = []
    for f, g in T.groupby("midi_filename"):
        e = matched_effect(g)
        if e is None:
            continue
        r = {"midi_filename": f, **e}
        es = matched_effect(g[g.s_ioi <= 1]); el = matched_effect(g[g.s_ioi >= 2])
        r["loc_short"] = es["loc"] if es else np.nan; r["loc_long"] = el["loc"] if el else np.nan
        r["sc_short"] = es["logscale"] if es else np.nan; r["sc_long"] = el["logscale"] if el else np.nan
        rows.append(r)
    R = pd.DataFrame(rows).join(M[["pianist", "piece", "year", "group", "sus_frac", "era"]], on="midi_filename")
    R["loc_ms"] = R["loc"] * 1000
    R["loc_long_minus_short_ms"] = (R.loc_long - R.loc_short) * 1000; R["sc_long_minus_short"] = R.sc_long - R.sc_short
    R.to_csv(RES / f"study1_performance_effects{SFX}.csv", index=False)
    S = {"variant": VAR, "n_perf": len(R), "n_pianists": int(R.pianist.nunique()),
         "H2a_loc_ms": pool(R, "loc_ms", B=B), "H2b_logscale": pool(R, "logscale", B=B),
         "share_loc_gt0": float((R.loc_ms > 0).mean()), "share_scale_gt0": float((R.logscale > 0).mean())}
    if VAR == "main" and "subset" not in OPTS:
        both = R.dropna(subset=["loc_long_minus_short_ms"])
        S["H2c_loc_long_minus_short_ms"] = pool(both, "loc_long_minus_short_ms", B=B)
        S["H2c_scale_long_minus_short"] = pool(both.dropna(subset=["sc_long_minus_short"]), "sc_long_minus_short", B=B)
    save_json(S, RES / f"study1_summary{SFX}.json")
    print(VAR, OPTS.get("subset", ""), "| perf", S["n_perf"], "| H2a loc ms", round(S["H2a_loc_ms"]["beta"], 2), [round(x, 2) for x in (S["H2a_loc_ms"]["ci_lo"], S["H2a_loc_ms"]["ci_hi"])],
          "p", f'{S["H2a_loc_ms"]["p"]:.2g}', "| H2b logscale", round(S["H2b_logscale"]["beta"], 3), [round(x, 3) for x in (S["H2b_logscale"]["ci_lo"], S["H2b_logscale"]["ci_hi"])],
          "p", f'{S["H2b_logscale"]["p"]:.2g}')
    if "H2c_loc_long_minus_short_ms" in S:
        print("H2c (study 1) loc long-short ms", round(S["H2c_loc_long_minus_short_ms"]["beta"], 2), "p", f'{S["H2c_loc_long_minus_short_ms"]["p"]:.2g}',
              "| scale long-short", round(S["H2c_scale_long_minus_short"]["beta"], 3), "p", f'{S["H2c_scale_long_minus_short"]["p"]:.2g}')

elif PART == "study2":
    D = pd.read_pickle(DER / f"matched_transitions{SFX}.pkl")
    D = D[D.ped.isin([0, 1])].copy()
    D["ped"] = D.ped.astype(float)
    # transition sets that contain both pedal states (plan); the others still fix performance effects in a robustness run
    mixed = D.groupby("set").ped.transform("nunique") == 2
    if OPTS.get("sets", "mixed") == "mixed":
        D = D[mixed].copy()
    D = D.join(M[["pianist", "piece", "year"]], on="midi_filename", rsuffix="_m")
    D["kot_ms"] = D.kot * 1000
    D["lioi"] = np.log(D.ioi) - np.log(D.ioi).mean()
    D["vel_c"] = (D.vel - D.vel.mean()) / 10
    D["ped_x_lioi"] = D.ped * D.lioi
    xc = ["ped", "lioi", "vel_c"]
    X = {c: D[c].values for c in xc}
    loc, sc = fe_location_scale(D.kot_ms.values, X, D.set.values, D.midi_filename.values, D.pianist.values, D.piece.values)
    cal = calibrate_scale(X, D.set.values, D.midi_filename.values, D.pianist.values, sigma=float(np.std(loc["_resid"])), reps=6, seed=1)
    ped_sc = (sc["ped"]["b"] - cal["intercept"]) / cal["slope"]; ped_sc_se = sc["ped"]["se"] / cal["slope"]
    X2 = {c: D[c].values for c in xc + ["ped_x_lioi"]}
    loc2, sc2 = fe_location_scale(D.kot_ms.values, X2, D.set.values, D.midi_filename.values, D.pianist.values, D.piece.values)
    boot = cluster_bootstrap_fe(D, "kot_ms", xc, "set", "midi_filename", cluster="piece", B=int(OPTS.get("Bfe", 200)), seed=2, calib=cal)
    S = {"variant": VAR, "n_transitions": len(D), "n_sets": int(D.set.nunique()), "n_perf": int(D.midi_filename.nunique()),
         "n_pianists": int(D.pianist.nunique()), "n_pieces": int(D.piece.nunique()), "share_down": float(D.ped.mean()),
         "H2a_loc_ms": loc["ped"] | {"ci": boot["loc_ci"]},
         "H2b_logscale": {"raw": sc["ped"]["b"], "b": ped_sc, "se": ped_sc_se, "z": ped_sc / ped_sc_se,
                          "p": float(2 * __import__("scipy").stats.norm.sf(abs(ped_sc / ped_sc_se))), "ci": boot["scale_ci"], "calibration": cal},
         "H2c_loc_ped_x_lioi_ms": loc2["ped_x_lioi"], "H2c_scale_ped_x_lioi_raw": sc2["ped_x_lioi"],
         "covariates_loc": {k: loc[k] for k in ["lioi", "vel_c"]}}
    save_json(S, RES / f"study2_summary{SFX}.json")
    print(VAR, "| transitions", S["n_transitions"], "sets", S["n_sets"], "perf", S["n_perf"], "pianists", S["n_pianists"], "pieces", S["n_pieces"])
    print("H2a loc ms", round(loc["ped"]["b"], 2), "se", round(loc["ped"]["se"], 2), "ci", [round(x, 2) for x in boot["loc_ci"]])
    print("H2b logscale raw", round(sc["ped"]["b"], 3), "calibrated", round(ped_sc, 3), "se", round(ped_sc_se, 3), "ci", [round(x, 3) for x in boot["scale_ci"]],
          "| calibration", round(cal["intercept"], 3), round(cal["slope"], 3))
    print("H2c ped x log IOI (loc, ms per log unit)", round(loc2["ped_x_lioi"]["b"], 2), "se", round(loc2["ped_x_lioi"]["se"], 2),
          "| scale raw", round(sc2["ped_x_lioi"]["b"], 3), "se", round(sc2["ped_x_lioi"]["se"], 3))
