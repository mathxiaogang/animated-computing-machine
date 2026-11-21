"""Step 29 - robustness checks and the Holm-corrected table of the eight confirmatory tests.

Usage:
    python 29_robustness.py [stage=all|h1|study1|study2|compile]

h1:     step 23 with controller cut-offs 32/96, maximum change gap 0.4/0.8 s, and a shifted-press surrogate
study1: step 26 (part=study1) on every transition variant of step 22, without Baroque pieces, without transcriptions and short pieces
study2: steps 25-26 (part=study2) on the transition variants; the "design ladder" that shows where the H2a effect goes
compile: leave-one-edition-out, the H1c bootstrap p, and results/confirmatory_tests.csv + results/robustness_table.csv
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import DER, PLAN, RES, fe_regress, kde_peak, load_meta, pool, save_json

HERE = Path(__file__).resolve().parent
STAGE = dict(a.split("=", 1) for a in sys.argv[1:]).get("stage", "all")


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=HERE.parent, capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    print(" ".join(map(str, args[1:])), "->", lines[-1] if lines else r.stderr[-500:])


if STAGE in ("all", "h1"):
    for a in [("thr=32", "tag=thr32"), ("thr=96", "tag=thr96"), ("gap=0.4", "tag=gap04"), ("gap=0.8", "tag=gap08"), ("surrogate=shift", "tag=shift")]:
        run(HERE / "23_handover.py", *a)

S1_VARIANTS = ["thr32", "thr96", "win025", "win060", "bottom", "separated", "interval12", "ioi015_080", "kot1000"]
if STAGE in ("all", "study1"):
    for v in S1_VARIANTS:
        run(HERE / "26_models.py", "part=study1", f"variant={v}", "B=300")
    for sub in ["no_baroque", "no_transcription_short"]:
        run(HERE / "26_models.py", "part=study1", f"subset={sub}", "B=300")

S2_VARIANTS = ["thr32", "thr96", "win025", "win060", "separated", "interval12", "kot1000"]
if STAGE in ("all", "study2"):
    for v in S2_VARIANTS:
        run(HERE / "25_matched_transitions.py", f"variant={v}")
        run(HERE / "26_models.py", "part=study2", f"variant={v}", "Bfe=100")
    # design ladder on the aligned transitions: where does the within-performance "later release" go?
    M = load_meta().set_index("midi_filename")
    D = pd.read_pickle(DER / "matched_transitions.pkl"); D = D[D.ped.isin([0, 1])].join(M[["pianist"]], on="midi_filename")
    D["kot_ms"] = D.kot * 1000; D["lioi"] = np.log(D.ioi) - np.log(D.ioi).mean(); D["vel_c"] = (D.vel - D.vel.mean()) / 10
    D["ped"] = D.ped.astype(float)
    mixed = D.groupby("set").ped.transform("nunique") == 2
    lad = {}

    def one_way(d, g):
        dm = d[["kot_ms", "ped", "lioi", "vel_c"]] - d.groupby(g)[["kot_ms", "ped", "lioi", "vel_c"]].transform("mean")
        X = dm[["ped", "lioi", "vel_c"]].values; b = np.linalg.lstsq(X, dm.kot_ms.values, rcond=None)[0]
        e = dm.kot_ms.values - X @ b; XtXi = np.linalg.inv(X.T @ X)
        codes = pd.factorize(d.pianist)[0]; Sg = np.zeros((codes.max() + 1, 3)); np.add.at(Sg, codes, X * e[:, None])
        se = np.sqrt(np.diag(XtXi @ (Sg.T @ Sg) @ XtXi))
        return {"b": float(b[0]), "se": float(se[0]), "n": len(d)}
    lad["performance_FE_only_all_aligned"] = one_way(D, "midi_filename")
    lad["position_FE_only_mixed_sets"] = one_way(D[mixed], "set")
    for lab, d in [("two_way_FE_all_sets", D), ("two_way_FE_mixed_sets", D[mixed])]:
        f = fe_regress(d.kot_ms.values, {"ped": d.ped.values, "lioi": d.lioi.values, "vel_c": d.vel_c.values}, d.set.values,
                       d.midi_filename.values, d.pianist.values, d.piece.values)
        lad[lab] = {"b": f["ped"]["b"], "se": f["ped"]["se"], "n": len(d)}
    save_json(lad, RES / "study2_design_ladder.json")
    print("design ladder:", {k: (round(v["b"], 2), round(v["se"], 2)) for k, v in lad.items()})

if STAGE in ("all", "compile"):
    M = load_meta().set_index("midi_filename")
    rng = np.random.default_rng(PLAN["seed"] + 1)
    rows = []
    # ---- H1 variants
    for tag, lab in [("", "main"), ("_thr32", "cut-off 32"), ("_thr96", "cut-off 96"), ("_gap04", "change gap <= 0.4 s"), ("_gap08", "change gap <= 0.8 s"),
                     ("_shift", "surrogate: press shifted +-0.5 s")]:
        p = RES / f"h1_summary{tag}.json"
        if p.exists():
            s = json.loads(p.read_text())
            rows.append({"analysis": "H1a avoidance A", "variant": lab, "estimate": s["H1a_A"]["beta"], "ci_lo": s["H1a_A"]["ci_lo"], "ci_hi": s["H1a_A"]["ci_hi"], "p": s["H1a_A"]["p"], "n": s["n_perf"]})
            rows.append({"analysis": "H1b handover H", "variant": lab, "estimate": s["H1b_H"]["beta"], "ci_lo": s["H1b_H"]["ci_lo"], "ci_hi": s["H1b_H"]["ci_hi"], "p": s["H1b_H"]["p"], "n": s["n_perf"]})
            lp = DER / f"h1_lags{tag}.pkl"
            if lp.exists() and tag != "_shift":  # H1c peak under each definition of the re-press (pianist bootstrap, 300 draws)
                Lx = pd.read_pickle(lp); pis = Lx.pianist.unique(); gx = {k: np.where(Lx.pianist.values == k)[0] for k in pis}
                bs = [kde_peak(Lx.lag.values[np.concatenate([gx[k] for k in rng.choice(pis, len(pis))])]) for _ in range(300)]
                rows.append({"analysis": "H1c peak (ms after re-press)", "variant": lab, "estimate": kde_peak(Lx.lag.values),
                             "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)), "p": np.nan, "n": s["n_perf"]})
    # ---- study 1 variants
    for tag, lab in [("", "main")] + [(f"_{v}", v) for v in S1_VARIANTS] + [("_nobaroque", "no Baroque"), ("_notrans_short", "no transcriptions, >= 2 min")]:
        p = RES / f"study1_summary{tag}.json"
        if p.exists():
            s = json.loads(p.read_text())
            rows.append({"analysis": "H2a study 1 location (ms)", "variant": lab, "estimate": s["H2a_loc_ms"]["beta"], "ci_lo": s["H2a_loc_ms"]["ci_lo"], "ci_hi": s["H2a_loc_ms"]["ci_hi"], "p": s["H2a_loc_ms"]["p"], "n": s["n_perf"]})
            rows.append({"analysis": "H2b study 1 log scale", "variant": lab, "estimate": s["H2b_logscale"]["beta"], "ci_lo": s["H2b_logscale"]["ci_lo"], "ci_hi": s["H2b_logscale"]["ci_hi"], "p": s["H2b_logscale"]["p"], "n": s["n_perf"]})
    # ---- study 2 variants
    for tag, lab in [("", "main")] + [(f"_{v}", v) for v in S2_VARIANTS]:
        p = RES / f"study2_summary{tag}.json"
        if p.exists():
            s = json.loads(p.read_text())
            rows.append({"analysis": "H2a study 2 location (ms)", "variant": lab, "estimate": s["H2a_loc_ms"]["b"], "ci_lo": s["H2a_loc_ms"]["ci"][0], "ci_hi": s["H2a_loc_ms"]["ci"][1], "p": s["H2a_loc_ms"]["p"], "n": s["n_transitions"]})
            rows.append({"analysis": "H2b study 2 log scale (calibrated)", "variant": lab, "estimate": s["H2b_logscale"]["b"], "ci_lo": s["H2b_logscale"]["ci"][0], "ci_hi": s["H2b_logscale"]["ci"][1], "p": s["H2b_logscale"]["p"], "n": s["n_transitions"]})
    # ---- leave one edition out (H1 and study 1, Wald only)
    H = pd.read_csv(RES / "h1_performance_indices.csv"); S1 = pd.read_csv(RES / "study1_performance_effects.csv")
    for yr in sorted(H.year.unique()):
        for col, d, lab in [("A", H, "H1a avoidance A"), ("H", H, "H1b handover H"), ("loc_ms", S1, "H2a study 1 location (ms)"), ("logscale", S1, "H2b study 1 log scale")]:
            p = pool(d[d.year != yr], col, B=0)
            rows.append({"analysis": lab, "variant": f"without {yr}", "estimate": p["beta"], "ci_lo": p["beta"] - 1.96 * p["se"], "ci_hi": p["beta"] + 1.96 * p["se"], "p": p["p"], "n": p["n"]})
    R = pd.DataFrame(rows); R.to_csv(RES / "robustness_table.csv", index=False)

    # ---- H1c bootstrap p: share of pianist-bootstrap peaks at or beyond the 100 ms bound
    L = pd.read_pickle(DER / "h1_lags.pkl"); pianists = L.pianist.unique(); gi = {k: np.where(L.pianist.values == k)[0] for k in pianists}
    peaks = [kde_peak(L.lag.values[np.concatenate([gi[k] for k in rng.choice(pianists, len(pianists))])]) for _ in range(PLAN["bootstrap_B"])]
    p_h1c = (np.sum(np.array(peaks) >= PLAN["feedforward_bound_ms"]) + 1) / (len(peaks) + 1)
    # ---- the eight confirmatory tests (studies 2-3 are the confirmatory designs for H2-H3)
    h1 = json.loads((RES / "h1_summary.json").read_text()); s2 = json.loads((RES / "study2_summary.json").read_text())
    sp = json.loads((RES / "specificity_summary.json").read_text())
    tests = [
        ("H1a", "fewer releases 30-210 ms before the re-press than surrogate", h1["H1a_A"]["beta"], h1["H1a_A"]["p"], h1["H1a_A"]["beta"] < 0),
        ("H1b", "more releases 0-60 ms after the re-press than surrogate", h1["H1b_H"]["beta"], h1["H1b_H"]["p"], h1["H1b_H"]["beta"] > 0),
        ("H1c", "release peak earlier than 100 ms after the re-press", h1["H1c"]["peak_ms"], float(p_h1c), True),
        ("H2a", "later release with the pedal down (same score position)", s2["H2a_loc_ms"]["b"], s2["H2a_loc_ms"]["p"], s2["H2a_loc_ms"]["b"] > 0),
        ("H2b", "more variable release with the pedal down (same score position)", s2["H2b_logscale"]["b"], s2["H2b_logscale"]["p"], s2["H2b_logscale"]["b"] > 0),
        ("H2c", "pedal effect grows with IOI (same score position)", s2["H2c_loc_ped_x_lioi_ms"]["b"], s2["H2c_loc_ped_x_lioi_ms"]["p"], s2["H2c_loc_ped_x_lioi_ms"]["b"] > 0),
        ("H3a", "chord onset spread equivalent with and without pedal (TOST)", sp["H3a_study2"]["b"], sp["H3a_study2"]["tost_p"], True),
        ("H3b", "release timing equivalent with and without una corda (TOST)", sp["H3b_study2"]["loc_ms"]["b"], sp["H3b_study2"]["loc_ms"]["tost_p"], True),
    ]
    T = pd.DataFrame(tests, columns=["hypothesis", "prediction", "estimate", "p", "direction_as_predicted"])
    order = np.argsort(T.p.values); m = len(T); adj = np.empty(m); run_max = 0
    for rank, i in enumerate(order):
        run_max = max(run_max, (m - rank) * T.p.values[i]); adj[i] = min(1.0, run_max)
    T["p_holm"] = adj
    T["supported"] = (T.p_holm < .05) & T.direction_as_predicted
    T.to_csv(RES / "confirmatory_tests.csv", index=False)
    print(T.to_string(index=False))
    print("robustness rows:", len(R))
