"""Step 20 - freeze the analysis plan, then check on synthetic data that every estimator recovers known effects,
and estimate power for the smallest effects of interest.

Usage:
    python 20_synthetic_validation.py

Writes results/analysis_plan_frozen.json (all parameters + SHA-256 of the plan and of release_common.py, with a UTC time stamp),
results/synthetic_validation.json and results/power.json. Run this before steps 22-29.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import (PLAN, RES, add_strata, calibrate_scale, fe_location_scale, fe_regress, handover_events, handover_indices, kde_peak, lmm,
                            matched_effect, save_json)

RES.mkdir(parents=True, exist_ok=True)
# ------------------------------------------------------------------ 1. freeze
plan_txt = json.dumps(PLAN, sort_keys=True, ensure_ascii=False)
common = (Path(__file__).resolve().parent / "release_common.py").read_bytes()
frozen = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "plan": PLAN,
          "sha256_plan": hashlib.sha256(plan_txt.encode()).hexdigest(), "sha256_release_common": hashlib.sha256(common).hexdigest(),
          "note": "Frozen before steps 22-29 were run. Not a public preregistration: the authors chose to run all analyses at once."}
frozen_path = RES / "analysis_plan_frozen.json"
if frozen_path.exists() and json.loads(frozen_path.read_text())["sha256_release_common"] != frozen["sha256_release_common"]:
    frozen_path.rename(RES / f"analysis_plan_frozen_superseded_{json.loads(frozen_path.read_text())['sha256_release_common'][:8]}.json")
if not frozen_path.exists():  # never overwrite an earlier freeze
    save_json(frozen, frozen_path)
print("plan sha256:", frozen["sha256_plan"][:16], "| common sha256:", frozen["sha256_release_common"][:16])

rng = np.random.default_rng(PLAN["seed"])


# ------------------------------------------------------------------ 2. Study 1 estimator (within-performance matched strata)
def synth_perf(rng, delta, gamma, n=700, style=0.0):
    ioi = np.clip(np.exp(rng.normal(np.log(0.22), 0.45, n)), 0.125, 1.15)
    # pedal blocks whose probability rises with IOI (a confound the strata must absorb)
    ped = np.zeros(n, int); state = rng.random() < 0.5
    for i in range(n):
        if rng.random() < 0.06:
            state = rng.random() < (0.3 + 0.5 * (ioi[i] > 0.3))
        ped[i] = state
    kot = 0.0 + style + 0.06 * (ioi - 0.25) + delta * ped + 0.04 * np.exp(gamma * ped) * rng.standard_normal(n)
    T = pd.DataFrame({"ioi": ioi, "kot": kot, "pa": rng.integers(60, 96, n), "interval": rng.integers(1, 8, n),
                      "vel": np.clip(rng.normal(65, 12, n), 1, 127), "ped": ped})
    T = T[np.abs(T.kot) < PLAN["kot_abs_max_s"]]
    return add_strata(T)


def study1(rng, delta, gamma, n_perf=220):
    rows = []
    for j in range(n_perf):
        e = matched_effect(synth_perf(rng, delta, gamma, style=rng.normal(0, 0.01)))
        if e:
            rows.append({**e, "pianist": f"p{j // 2}", "piece": f"w{rng.integers(0, 150)}", "year": f"y{j % 10}"})
    R = pd.DataFrame(rows)
    out = {}
    for c in ["loc", "logscale"]:
        f = lmm(R[c].values, np.ones((len(R), 1)), [R[g].values for g in ["pianist", "piece", "year"]])
        out[c] = (float(f["beta"][0]), float(f["se"][0]), float(f["p"][0]))
    return out, len(R)


val = {}
for name, d_, g_ in [("null", 0.0, 0.0), ("effect_20ms_scale0.3", 0.02, 0.3)]:
    est, n = study1(rng, d_, g_)
    val[f"study1_{name}"] = {"true_loc_ms": d_ * 1000, "true_logscale": g_, "n_perf": n,
                             "est_loc_ms": est["loc"][0] * 1000, "se_loc_ms": est["loc"][1] * 1000, "p_loc": est["loc"][2],
                             "est_logscale": est["logscale"][0], "se_logscale": est["logscale"][1], "p_logscale": est["logscale"][2]}
    print(name, json.dumps(val[f"study1_{name}"], indent=None)[:300])

# false-positive rate under the null (Wald, 40 replications)
fp = [study1(rng, 0.0, 0.0, n_perf=150)[0] for _ in range(40)]
val["study1_null_rejection_rate"] = {"loc": float(np.mean([f["loc"][2] < .05 for f in fp])), "logscale": float(np.mean([f["logscale"][2] < .05 for f in fp])), "reps": 40}
print("null rejection:", val["study1_null_rejection_rate"])


# ------------------------------------------------------------------ 3. Study 2 estimator (two-way fixed effects, pianist confound)
def study2(rng, delta, gamma, n_piece=60, n_perf=5, n_set=150, calib=True):
    rows = []
    for w in range(n_piece):
        style = rng.normal(0, 0.015, n_perf)
        prop = 0.6 * style / 0.015 + rng.normal(0, 0.8, n_perf)  # pianists who pedal more also hold longer: a confound
        for s in range(n_set):
            music = rng.normal(0, 0.03); tend = rng.normal(0, 1.2)
            for j in range(n_perf):
                if rng.random() < 0.2:
                    continue
                ped = int(1 / (1 + np.exp(-(tend + prop[j]))) > rng.random())
                kot = music + style[j] + delta * ped + 0.04 * np.exp(gamma * ped) * rng.standard_normal()
                rows.append((f"{w}-{s}", f"{w}-{j}", f"pn{w}-{j}", ped, kot))
    D = pd.DataFrame(rows, columns=["set", "perf", "pianist", "ped", "kot"])
    D["piece"] = D.set.str.split("-").str[0]  # inference clusters by pianist AND piece (two-way), as in step 26
    mixed = D.groupby("set").ped.transform(lambda x: x.nunique() == 2)
    D = D[mixed].reset_index(drop=True)
    f, g = fe_location_scale(D.kot.values, {"ped": D.ped.values.astype(float)}, D.set.values, D.perf.values, D.pianist.values, D.piece.values)
    naive = D.groupby("set").apply(lambda x: x[x.ped == 1].kot.mean() - x[x.ped == 0].kot.mean(), include_groups=False).mean()
    cal = calibrate_scale({"ped": D.ped.values.astype(float)}, D.set.values, D.perf.values, D.pianist.values, reps=4) if calib else {"intercept": 0.0, "slope": 1.0}
    return {"n": len(D), "est_loc_ms": f["ped"]["b"] * 1000, "se_loc_ms": f["ped"]["se"] * 1000, "raw_logscale": g["ped"]["b"],
            "est_logscale": (g["ped"]["b"] - cal["intercept"]) / cal["slope"], "se_logscale": g["ped"]["se"] / cal["slope"],
            "calibration": [cal["intercept"], cal["slope"]], "naive_set_only_loc_ms": float(naive * 1000)}


for name, d_, g_ in [("null", 0.0, 0.0), ("effect_20ms_scale0.3", 0.02, 0.3)]:
    val[f"study2_{name}"] = {"true_loc_ms": d_ * 1000, "true_logscale": g_, **study2(rng, d_, g_)}
    print("study2", name, val[f"study2_{name}"])

fp2 = [study2(rng, 0.0, 0.0, n_piece=25) for _ in range(20)]
val["study2_null_rejection_rate"] = {"loc": float(np.mean([abs(f["est_loc_ms"] / f["se_loc_ms"]) > 1.96 for f in fp2])), "logscale": float(np.mean([abs(f["est_logscale"] / f["se_logscale"]) > 1.96 for f in fp2])), "reps": 20}
print("study2 null rejection:", val["study2_null_rejection_rate"])


# ------------------------------------------------------------------ 4. handover surrogate test (H1)
def synth_handover(rng, share_locked, n_ev=400):
    """Chords, pedal changes after each chord, held notes released either independently of the foot or locked to it."""
    on, off, pitch, cc = [], [], [], [[0.0, 0]]
    t = 1.0
    for _ in range(n_ev):
        chord_t = t
        r = chord_t + max(0.01, rng.normal(0.07, 0.03)); p = r + max(0.03, rng.normal(0.12, 0.05))
        cc += [[r - 0.6, 127], [r, 0], [p, 127]]
        for k in range(3):
            o = chord_t + rng.uniform(0, 0.02)
            rel = o + np.exp(rng.normal(np.log(0.3), 0.5)) if rng.random() > share_locked else p + rng.normal(0.02, 0.025)
            on.append(o); off.append(max(rel, o + 0.03)); pitch.append(60 + 4 * k)
        t = p + rng.uniform(0.3, 0.9)
    o = np.argsort(on)
    cc = np.array(cc, float); cc = cc[np.argsort(cc[:, 0], kind="stable")]
    return {"on": np.array(on)[o], "off": np.array(off)[o], "pitch": np.array(pitch)[o], "vel": np.full(len(on), 64), "cc64": cc}


for name, share in [("null", 0.0), ("locked_30pct", 0.3)]:
    res = []
    for _ in range(30):
        E = handover_events(synth_handover(rng, share))
        res.append(handover_indices(E, rng, n_sur=50) | {"peak_ms": kde_peak(E.lag.values)})
    R = pd.DataFrame(res)
    val[f"handover_{name}"] = {"mean_A": R.A.mean(), "sd_A": R.A.std(), "mean_H": R.H.mean(), "sd_H": R.H.std(),
                               "share_H_gt0": (R.H > 0).mean(), "median_peak_ms": R.peak_ms.median()}
    print("handover", name, val[f"handover_{name}"])
save_json(val, RES / "synthetic_validation.json")

# ------------------------------------------------------------------ 5. power for the smallest effects of interest (Study 1 design)
# between-performance SD of the per-performance effects, taken from the pilot (exploratory) run
pilot = pd.read_csv(Path(__file__).resolve().parent.parent / "results" / "derived" / "pilot_release_strata.csv") if (Path(__file__).resolve().parent.parent / "results" / "derived" / "pilot_release_strata.csv").exists() else None
sd_loc = 0.045 if pilot is None else float(pilot["loc"].std())
sd_sc = 0.35 if pilot is None else float(pilot["logscale"].std())
pw = {}
for n in [100, 150, 220, 300]:
    zc = 1.96
    for lab, eff, sd in [("loc_10ms", 0.010, sd_loc), ("logscale_0.095", PLAN["sesoi_log_scale"], sd_sc)]:
        se = sd / np.sqrt(n) * 1.3  # x1.3 for clustering by pianist and piece
        pw[f"{lab}_n{n}"] = float(1 - norm.cdf(zc - eff / se) + norm.cdf(-zc - eff / se))
pw["assumed_between_performance_sd"] = {"loc_s": sd_loc, "logscale": sd_sc, "source": "pilot" if pilot is not None else "default"}
save_json(pw, RES / "power.json")
print("power:", {k: round(v, 3) for k, v in pw.items() if isinstance(v, float)})
