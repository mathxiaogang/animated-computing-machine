"""Step 33 - how the hand-foot handover is organised (directions 1-4; exploratory extensions designed after a pilot on the same
data, so p values are descriptive, not confirmatory).

D1 tracking: does a later re-press go with a later key release, event by event? Slope of (median release - anchor) on
   (re-press - anchor), controlling log local IOI: within each performance (pooled, crossed random intercepts), and at the same
   score position across performances (two-way fixed effects: event set x performance; clusters pianist and piece).
D2 stroke coupling: slope of (release - t32) on the down-stroke duration D = t96 - t32 (0 = locked to the stroke's start,
   1 = locked to its end); same two designs.
D3 what gets cut: within the same chord (event fixed effects), is a held note released before the re-press more often when its
   finger is needed for a nearby new note (motor need) or when its pitch class is doubled by another held note (acoustic
   redundancy)? Linear probability model, clusters pianist and piece.
D4 two-process model: per performance, lag density on [-300, 400] ms = (1 - w) x onset-locked surrogate density
   + w x Normal(mu, sigma) locked to the re-press; maximum likelihood; synthetic recovery check; individual differences.

Usage:
    python 33_handover_mechanisms.py [part=all|d1|d2|d3|d4]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import (DER, PLAN, RES, fe_regress, handover_events, handover_indices, lmm, load_cache, load_meta, pool,
                            save_json)

PART = dict(a.split("=", 1) for a in sys.argv[1:]).get("part", "all")
M = load_meta().set_index("midi_filename")
EV = pd.read_pickle(DER / "handover_events.pkl")
out_path = RES / "mechanisms_summary.json"
S = json.loads(out_path.read_text()) if out_path.exists() else {}


def within_slopes(df, y, x, covs, min_n=20, by="midi_filename"):
    rows = []
    for f, g in df.groupby(by):
        g = g.dropna(subset=[y, x, *covs])
        if len(g) < min_n or g[x].std() < 1e-3:
            continue
        X = np.column_stack([np.ones(len(g)), g[x].values] + [g[c].values for c in covs])
        b = np.linalg.lstsq(X, g[y].values, rcond=None)[0]
        rows.append({by: f, "slope": float(b[1]), "n": len(g)})
    return pd.DataFrame(rows).join(M[["pianist", "piece", "year", "sus_frac"]], on=by)


def same_position(df, y, x, covs):
    d = df.dropna(subset=[y, x, *covs]).copy()
    d = d[d.groupby("set").midi_filename.transform("nunique") >= 2].join(M[["pianist"]], on="midi_filename")
    f = fe_regress(d[y].values, {x: d[x].values, **{c: d[c].values for c in covs}}, d.set.values, d.midi_filename.values,
                   d.pianist.values, d.piece.values)
    return f[x] | {"n_events": len(d), "n_sets": int(d.set.nunique()), "n_perf": int(d.midi_filename.nunique())}


EV["y1"] = EV.rel_med - EV.anchor; EV["x1"] = EV.p - EV.anchor; EV["lioi"] = np.log(EV.ioi_next)
EV.loc[(EV.y1 > 2.0) | (EV.ioi_next <= 0), ["y1"]] = np.nan
EV["y2"] = EV.rel_win_med - EV.t32
ok2 = (EV.D > 0.005) & (EV.D <= 0.3)

if PART in ("all", "d1"):
    W = within_slopes(EV, "y1", "x1", ["lioi"])
    MT = pd.read_pickle(DER / "handover_events_matched.pkl")
    MT["y1"] = MT.rel_med - MT.anchor; MT["x1"] = MT.p - MT.anchor; MT["lioi"] = np.log(MT.ioi_next)
    MT.loc[(MT.y1 > 2.0) | (MT.ioi_next <= 0), "y1"] = np.nan
    S["D1"] = {"within": pool(W, "slope") | {"share_gt0": float((W.slope > 0).mean())}, "same_position": same_position(MT, "y1", "x1", ["lioi"])}
    W.to_csv(RES / "d1_tracking_slopes.csv", index=False)
    print("D1 within: slope", round(S["D1"]["within"]["beta"], 3), [round(S["D1"]["within"]["ci_lo"], 3), round(S["D1"]["within"]["ci_hi"], 3)],
          "perf", S["D1"]["within"]["n"], "| same position:", round(S["D1"]["same_position"]["b"], 3), "se", round(S["D1"]["same_position"]["se"], 3),
          "events", S["D1"]["same_position"]["n_events"])
    # binned curve for the figure (within-performance centred)
    E1 = EV.dropna(subset=["y1", "x1"]).copy()
    E1["xc"] = E1.x1 - E1.groupby("midi_filename").x1.transform("mean"); E1["yc"] = E1.y1 - E1.groupby("midi_filename").y1.transform("mean")
    E1["bin"] = pd.qcut(E1.xc, 20, labels=False)
    E1.groupby("bin")[["xc", "yc"]].mean().mul(1000).to_csv(RES / "d1_binned.csv")

if PART in ("all", "d2"):
    NT = pd.read_pickle(DER / "handover_notes.pkl").merge(EV[["midi_filename", "event", "t32", "D", "lioi"]], on=["midi_filename", "event"])
    NT = NT[(NT.D > 0.005) & (NT.D <= 0.3) & (NT.off > NT.t32 - 0.1) & (NT.off < NT.t32 + 0.4)].copy()
    NT["y"] = NT.off - NT.t32
    W = within_slopes(NT, "y", "D", ["lioi"], min_n=40)
    MT = pd.read_pickle(DER / "handover_events_matched.pkl")
    MT["y2"] = MT.rel_win_med - MT.t32; MT["lioi"] = np.log(MT.ioi_next)
    MT = MT[(MT.D > 0.005) & (MT.D <= 0.3) & (MT.ioi_next > 0)]
    S["D2"] = {"within": pool(W, "slope") | {"share_gt0": float((W.slope > 0).mean()), "share_lt1": float((W.slope < 1).mean())},
               "same_position": same_position(MT, "y2", "D", ["lioi"]), "median_D_ms": float(EV.D[ok2].median() * 1000),
               "median_phase": float(np.median((NT.y / NT.D).clip(-5, 10)))}
    W.to_csv(RES / "d2_stroke_slopes.csv", index=False)
    print("D2 within: slope", round(S["D2"]["within"]["beta"], 3), [round(S["D2"]["within"]["ci_lo"], 3), round(S["D2"]["within"]["ci_hi"], 3)],
          "perf", S["D2"]["within"]["n"], "| same position:", round(S["D2"]["same_position"]["b"], 3), "se", round(S["D2"]["same_position"]["se"], 3),
          "events", S["D2"]["same_position"]["n_events"])
    NT["Dbin"] = pd.qcut(NT.D, 10, labels=False)
    NT.groupby("Dbin").agg(D_ms=("D", lambda x: x.mean() * 1000), y_ms=("y", lambda x: x.median() * 1000)).to_csv(RES / "d2_binned.csv")

if PART in ("all", "d3"):
    NT = pd.read_pickle(DER / "handover_notes.pkl")
    NT["uid"] = NT.midi_filename + "|" + NT.event.astype(str)
    NT = NT[NT.groupby("uid").note.transform("size") >= 2].join(M[["pianist", "piece"]], on="midi_filename")
    X = {"need": NT.need.values.astype(float), "doubled": NT.doubled.values.astype(float),
         "inner": (NT["rank"] == "inner").values.astype(float), "bottom": (NT["rank"] == "bottom").values.astype(float),
         "register": (NT.pitch.values - 60) / 12, "vel": NT.vel.values / 10}
    f = fe_regress(NT.cut.values.astype(float), X, NT.uid.values, NT.midi_filename.values, NT.pianist.values, NT.piece.values)
    X2 = dict(X); X2["need_x_doubled"] = X["need"] * X["doubled"]
    f2 = fe_regress(NT.cut.values.astype(float), X2, NT.uid.values, NT.midi_filename.values, NT.pianist.values, NT.piece.values)
    rates = NT.groupby(["need", "doubled"]).cut.agg(["mean", "size"]).reset_index()
    S["D3"] = {"n_notes": len(NT), "n_events": int(NT.uid.nunique()), "share_need": float(NT.need.mean()), "share_doubled": float(NT.doubled.mean()),
               "lpm": {k: v for k, v in f.items() if not k.startswith("_")}, "interaction": f2["need_x_doubled"], "rates": rates.values.tolist()}
    print("D3 notes", len(NT), "| need", {k: round(v, 4) for k, v in f["need"].items()}, "| doubled", {k: round(v, 4) for k, v in f["doubled"].items()})
    print("   inner", round(f["inner"]["b"], 4), "bottom", round(f["bottom"]["b"], 4), "register", round(f["register"]["b"], 4), "| need x doubled", round(f2["need_x_doubled"]["b"], 4))
    print(rates)

if PART in ("all", "d4"):
    C = load_cache(); rng = np.random.default_rng(PLAN["seed"] + 4)
    LO, HI = -300.0, 400.0

    def fit(L_ms, S_ms):
        L = L_ms[(L_ms > LO) & (L_ms < HI)]; Sx = S_ms[(S_ms > LO) & (S_ms < HI)]
        if len(L) < 80 or len(Sx) < 200:
            return None
        kde = stats.gaussian_kde(Sx, bw_method=0.15); sd = kde(L) / kde.integrate_box_1d(LO, HI)

        def nll(t):
            w, mu, sg = t
            gn = stats.norm.pdf(L, mu, sg) / (stats.norm.cdf(HI, mu, sg) - stats.norm.cdf(LO, mu, sg))
            return -np.sum(np.log((1 - w) * sd + w * gn + 1e-300))
        best = None
        for start in ([0.2, 30, 40], [0.4, 60, 80], [0.1, 0, 30]):
            o = optimize.minimize(nll, start, bounds=[(0, 0.99), (-100, 200), (5, 150)], method="L-BFGS-B")
            best = o if best is None or o.fun < best.fun else best
        l0 = -np.sum(np.log(sd + 1e-300))
        return {"w": float(best.x[0]), "mu": float(best.x[1]), "sigma": float(best.x[2]), "lr": float(2 * (l0 - best.fun)), "n_lags": len(L)}

    # synthetic recovery (same generator as step 20)
    src = (Path(__file__).resolve().parent / "20_synthetic_validation.py").read_text()
    ns = {"np": np, "pd": pd}
    exec(src[src.index("def synth_handover"):src.index("for name, share in")], ns)
    rec = []
    for share in (0.0, 0.3, 0.6):
        for _ in range(10):
            E = handover_events(ns["synth_handover"](rng, share, n_ev=300))
            r = handover_indices(E, rng, n_sur=50, return_sur=True)
            ft = fit(E.lag.values * 1000, r["sur_lags"] * 1000)
            if ft:
                rec.append({"true_w": share, **ft})
    R0 = pd.DataFrame(rec)
    S["D4_synthetic"] = R0.groupby("true_w")[["w", "mu", "sigma", "lr"]].median().reset_index().to_dict("records")
    print("D4 synthetic recovery (true locked latency 20 ms, sd 25 ms):\n", R0.groupby("true_w")[["w", "mu", "sigma", "lr"]].median().round(2))
    rows = []
    for f, d in C.items():
        E = handover_events(d)
        if len(E) == 0 or E.event.nunique() < 40:
            continue
        r = handover_indices(E, rng, return_sur=True)
        ft = fit(E.lag.values * 1000, r["sur_lags"] * 1000)
        if ft:
            rows.append({"midi_filename": f, **ft})
    R = pd.DataFrame(rows).join(M[["pianist", "piece", "year", "group", "sus_frac"]], on="midi_filename")
    H = pd.read_csv(RES / "h1_performance_indices.csv")[["midi_filename", "H", "A"]]
    R = R.merge(H, on="midi_filename", how="left")
    R.to_csv(RES / "d4_mixture_parameters.csv", index=False)

    def shares(d, col):
        d = d.dropna(subset=[col]); y = (d[col] - d[col].mean()) / d[col].std()
        v = lmm(y.values, np.ones((len(d), 1)), [d[g].astype(str).values for g in ["pianist", "piece", "year"]])["vc"]
        v = v / v.sum(); return {"resid": float(v[0]), "pianist": float(v[1]), "piece": float(v[2]), "edition": float(v[3])}

    def split_half(d, col, reps=200):
        g = d.dropna(subset=[col]).groupby("pianist")[col].apply(list); g = g[g.apply(len) >= 2]; rs = []
        for _ in range(reps):
            a, b = [], []
            for v in g:
                v = rng.permutation(v); h = len(v) // 2; a.append(np.mean(v[:h])); b.append(np.mean(v[h:]))
            rs.append(stats.pearsonr(a, b)[0])
        r_ = float(np.mean(rs)); return {"n_pianists": int(len(g)), "r_half": r_, "spearman_brown": 2 * r_ / (1 + r_)}
    S["D4"] = {"n_perf": len(R), "n_pianists": int(R.pianist.nunique()), "share_lr_gt_3.84": float((R.lr > 3.84).mean()),
               **{c: {"median": float(R[c].median()), "iqr": R[c].quantile([.25, .75]).tolist(), "pooled_mean": pool(R, c, B=300),
                      "shares": shares(R, c), "split_half": split_half(R, c)} for c in ["w", "mu", "sigma"]},
               "corr": {"w_vs_mu": stats.spearmanr(R.w, R.mu)[0], "w_vs_sigma": stats.spearmanr(R.w, R.sigma)[0], "mu_vs_sigma": stats.spearmanr(R.mu, R.sigma)[0],
                        "w_vs_H": stats.spearmanr(R.w, R.H, nan_policy="omit")[0],
                        "pianist_w_vs_sustain": list(stats.spearmanr(*R.groupby("pianist")[["w", "sus_frac"]].mean().values.T))}}
    print("D4 perf", len(R), "| w median", round(R.w.median(), 3), R.w.quantile([.25, .75]).round(3).tolist(), "| mu", round(R.mu.median(), 1),
          "| sigma", round(R.sigma.median(), 1), "| LR>3.84", round(S["D4"]["share_lr_gt_3.84"], 3))
    for c in ["w", "mu", "sigma"]:
        print("  ", c, "shares", {k: round(v, 3) for k, v in S["D4"][c]["shares"].items()}, "| split-half SB", round(S["D4"][c]["split_half"]["spearman_brown"], 3))
    print("   corr", {k: (round(v, 3) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in S["D4"]["corr"].items()})
save_json(S, out_path)
