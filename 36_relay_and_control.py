"""Step 36 - second round of exploratory extensions (designed after pilots on the same data; p values are descriptive).

relay  The foot waits for the hand at the START of a pedal change (the mirror image of H1): are pedal-ups rare just before a
       chord onset and frequent just after it, compared with a surrogate that keeps each pedal-up locked to the foot's own
       previous press (hold durations permuted within the performance)? Placebo: chord onsets shifted by +0.5 s.
ctrl   Anticipatory and online control: the re-press lag (from the anchor onset) is split into a part predictable from the
       pianist's habit (mean of the previous 5 changes) and an unpredicted residual; slopes of the key release on both, and
       slopes of release QUANTILES (10th-75th percentile) on the residual. Repeated with a same-score-position residual
       (re-press lag after removing score-position and performance fixed effects).
ref    Reference point: dispersion (MAD) of key release relative to the moment the pedal passes controller values 16-112 on its
       down-stroke, vs relative to the hand's own onset, vs a surrogate pairing; pedal depth at release by edition and in the 16
       pianists who played two editions.
pred   Applied: predicting the key release of held notes (two-fold cross-validation within each performance) from the onset,
       from the re-press, or a blend; compared with the error made by equating the sound offset with the key release.

Usage:
    python 36_relay_and_control.py [part=all|relay|ctrl|ref|pred]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import (DER, PLAN, RES, clusters, crossings, demean2, kde_peak, load_cache, load_meta, ped_value, pool, save_json)

PART = dict(a.split("=", 1) for a in sys.argv[1:]).get("part", "all")
M = load_meta().set_index("midi_filename")
out_path = RES / "round2_summary.json"
S = json.loads(out_path.read_text()) if out_path.exists() else {}
rng = np.random.default_rng(PLAN["seed"] + 36)
a0, a1 = np.array(PLAN["avoid_window_ms"]) / 1000; h0, h1 = np.array(PLAN["handover_window_ms"]) / 1000


def window_counts(r, chords):
    """pedal-ups falling 30-210 ms BEFORE a chord onset (avoid) and 0-60 ms AFTER one (handover)."""
    k = np.searchsorted(chords, r, side="right")
    d_before = np.where(k > 0, r - chords[np.maximum(k - 1, 0)], np.inf)
    d_after = np.where(k < len(chords), chords[np.minimum(k, len(chords) - 1)] - r, np.inf)
    return int(((d_after > -a1) & (d_after < -a0)).sum()), int(((d_before >= h0) & (d_before < h1)).sum()), d_before


if PART in ("all", "relay"):
    C = load_cache(); rows, lags = [], []
    for f, d in C.items():
        pr, rl = crossings(d["cc64"], PLAN["pedal_threshold"])
        if len(rl) < PLAN["min_events"] or len(pr) < 2:
            continue
        j = np.searchsorted(pr, rl); ok = j < len(pr)
        r = rl[ok]; nxt = pr[j[ok]]; chg = (nxt - r) < PLAN["change_max_gap_s"]; r = r[chg]
        i = np.searchsorted(pr, r) - 1; has = i >= 0; r = r[has]; pprev = pr[i[has]]
        if len(r) < PLAN["min_events"]:
            continue
        on = d["on"]; cid = clusters(on, PLAN["cluster_gap_s"])
        cnt = np.bincount(cid); first = np.r_[True, np.diff(cid) > 0]
        chords = on[first][cnt >= 2]
        hold = r - pprev
        ra, rh, db = window_counts(r, chords)
        sa, sh = [], []
        for _ in range(PLAN["n_surrogates"]):
            rs = np.sort(pprev + rng.permutation(hold))
            x, y, _ = window_counts(rs, chords); sa.append(x); sh.append(y)
        pa, ph, _ = window_counts(r, chords + 0.5)  # placebo: chords displaced by 0.5 s
        psa, psh = [], []
        for _ in range(20):
            x, y, _ = window_counts(np.sort(pprev + rng.permutation(hold)), chords + 0.5); psa.append(x); psh.append(y)
        rows.append({"midi_filename": f, "n_changes": len(r), "A_foot": np.log((ra + .5) / (np.mean(sa) + .5)), "H_foot": np.log((rh + .5) / (np.mean(sh) + .5)),
                     "A_placebo": np.log((pa + .5) / (np.mean(psa) + .5)), "H_placebo": np.log((ph + .5) / (np.mean(psh) + .5)),
                     "lag_med_ms": float(np.median(db[np.isfinite(db) & (db < 0.5)]) * 1000)})
        lags.append(pd.DataFrame({"midi_filename": f, "lag": db[np.isfinite(db) & (db < 0.5)]}))
    R = pd.DataFrame(rows).join(M[["pianist", "piece", "year"]], on="midi_filename")
    H = pd.read_csv(RES / "h1_performance_indices.csv")[["midi_filename", "A", "H"]]
    R = R.merge(H, on="midi_filename", how="left"); R.to_csv(RES / "relay_foot_indices.csv", index=False)
    L = pd.concat(lags).join(M[["pianist"]], on="midi_filename")
    pk = kde_peak(L.lag.values, grid_ms=[0, 400]); pis = L.pianist.unique(); gi = {k: np.where(L.pianist.values == k)[0] for k in pis}
    bs = [kde_peak(L.lag.values[np.concatenate([gi[k] for k in rng.choice(pis, len(pis))])], grid_ms=[0, 400]) for _ in range(300)]
    pm = R.groupby("pianist")[["H_foot", "H"]].mean()
    S["relay"] = {"n_perf": len(R), "n_changes": int(R.n_changes.sum()), "A_foot": pool(R, "A_foot"), "H_foot": pool(R, "H_foot"),
                  "A_placebo": pool(R, "A_placebo", B=200), "H_placebo": pool(R, "H_placebo", B=200),
                  "share_A_foot_lt0": float((R.A_foot < 0).mean()), "share_H_foot_gt0": float((R.H_foot > 0).mean()),
                  "foot_lag_peak_ms": pk, "foot_lag_peak_ci": np.percentile(bs, [2.5, 97.5]).tolist(), "foot_lag_median_ms": float(R.lag_med_ms.median()),
                  "perf_corr_Hfoot_Hhand": stats.spearmanr(R.H_foot, R.H, nan_policy="omit")[0],
                  "pianist_corr_Hfoot_Hhand": list(stats.spearmanr(pm.H_foot, pm.H, nan_policy="omit"))}
    print("relay: perf", len(R), "| A_foot", round(S["relay"]["A_foot"]["beta"], 3), [round(S["relay"]["A_foot"]["ci_lo"], 3), round(S["relay"]["A_foot"]["ci_hi"], 3)],
          "| H_foot", round(S["relay"]["H_foot"]["beta"], 3), [round(S["relay"]["H_foot"]["ci_lo"], 3), round(S["relay"]["H_foot"]["ci_hi"], 3)],
          "| placebo A/H", round(S["relay"]["A_placebo"]["beta"], 3), round(S["relay"]["H_placebo"]["beta"], 3),
          "| foot lag peak", pk, [round(x, 1) for x in S["relay"]["foot_lag_peak_ci"]], "| corr with hand H (perf, pianist)",
          round(S["relay"]["perf_corr_Hfoot_Hhand"], 3), [round(x, 4) for x in S["relay"]["pianist_corr_Hfoot_Hhand"]])

if PART in ("all", "ctrl"):
    EV = pd.read_pickle(DER / "handover_events.pkl").sort_values(["midi_filename", "anchor"]).copy()
    EV["x"] = EV.p - EV.anchor; EV["y"] = EV.rel_med - EV.anchor; EV["lioi"] = np.log(EV.ioi_next)
    EV = EV[(EV.ioi_next > 0)].copy()
    EV["pred"] = EV.groupby("midi_filename").x.transform(lambda s: s.shift(1).rolling(5).mean()); EV["res"] = EV.x - EV.pred
    rows = []
    for f, g in EV.dropna(subset=["pred", "y"]).groupby("midi_filename"):
        g = g[g.y < 2.0]
        if len(g) < 30:
            continue
        X = np.column_stack([np.ones(len(g)), g.pred, g.res, g.lioi]); b = np.linalg.lstsq(X, g.y, rcond=None)[0]
        rows.append({"midi_filename": f, "slope_pred": b[1], "slope_res": b[2], "diff": b[1] - b[2]})
    R = pd.DataFrame(rows).join(M[["pianist", "piece", "year"]], on="midi_filename")
    NT = pd.read_pickle(DER / "handover_notes.pkl").merge(EV[["midi_filename", "event", "res", "lioi"]], on=["midi_filename", "event"]).dropna(subset=["res"])
    NT["yv"] = NT.off - NT.anchor; NT = NT[NT.yv < 2.0].copy()
    NT["yr"] = NT.groupby("midi_filename", group_keys=False)[["yv", "lioi"]].apply(lambda g: g.yv - np.polyval(np.polyfit(g.lioi, g.yv, 1), g.lioi))
    NT["q"] = NT.groupby("midi_filename").res.transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    qrows = []
    for tau in (0.10, 0.25, 0.50, 0.75):
        t = NT.groupby(["midi_filename", "q"]).agg(yq=("yr", lambda s: s.quantile(tau)), xr=("res", "mean")).reset_index()
        for f, g in t.groupby("midi_filename"):
            if g.q.nunique() >= 4:
                qrows.append({"midi_filename": f, "tau": tau, "slope": np.polyfit(g.xr, g.yq, 1)[0]})
    Q = pd.DataFrame(qrows).join(M[["pianist", "piece", "year"]], on="midi_filename")
    early = NT.groupby("midi_filename").apply(lambda g: np.percentile(g.off - g.p, 10), include_groups=False)
    # same-score-position residual: re-press lag after removing score-position and performance fixed effects
    MT = pd.read_pickle(DER / "handover_events_matched.pkl")
    MT = MT[MT.groupby("set").midi_filename.transform("nunique") >= 2].copy(); MT["x"] = MT.p - MT.anchor
    MT["res_sp"] = demean2({"x": MT.x.values}, MT.set.values, MT.midi_filename.values)["x"]
    NS = pd.read_pickle(DER / "handover_notes.pkl").merge(MT[["midi_filename", "event", "set", "res_sp"]], on=["midi_filename", "event"])
    NS["yv"] = NS.off - NS.anchor; NS = NS[NS.yv < 2.0].copy()
    NS["yr"] = demean2({"y": NS.yv.values}, NS.set.values, NS.midi_filename.values)["y"]
    NS = NS.join(M[["pianist"]], on="midi_filename")
    NS["dec"] = pd.qcut(NS.res_sp, 10, labels=False)

    def sp_slopes(d):
        out = {}
        for tau in (0.10, 0.25, 0.50, 0.75):
            t = d.groupby("dec").agg(yq=("yr", lambda s: s.quantile(tau)), xr=("res_sp", "mean"))
            out[tau] = float(np.polyfit(t.xr, t.yq, 1)[0])
        return out
    sp = sp_slopes(NS); pis = NS.pianist.unique(); gi = {k: np.where(NS.pianist.values == k)[0] for k in pis}; bs = []
    for _ in range(200):
        s_ = np.concatenate([gi[k] for k in rng.choice(pis, len(pis))]); bs.append(sp_slopes(NS.iloc[s_]))
    S["ctrl"] = {"habit": {"n_perf": len(R), "slope_pred": pool(R, "slope_pred"), "slope_res": pool(R, "slope_res"), "pred_minus_res": pool(R, "diff")},
                 "quantiles_habit": {str(t): pool(Q[Q.tau == t], "slope", B=300) | {"share_gt0": float((Q[Q.tau == t].slope > 0).mean())} for t in (0.10, 0.25, 0.50, 0.75)},
                 "p10_release_minus_press_ms": float(early.median() * 1000),
                 "quantiles_same_position": {str(t): {"slope": sp[t], "ci": np.percentile([b[t] for b in bs], [2.5, 97.5]).tolist()} for t in sp},
                 "same_position_n_notes": len(NS), "same_position_n_events": int(NS.groupby(["midi_filename", "event"]).ngroups)}
    print("ctrl habit: slope pred", round(S["ctrl"]["habit"]["slope_pred"]["beta"], 3), "res", round(S["ctrl"]["habit"]["slope_res"]["beta"], 3),
          "diff", round(S["ctrl"]["habit"]["pred_minus_res"]["beta"], 3), [round(S["ctrl"]["habit"]["pred_minus_res"]["ci_lo"], 3), round(S["ctrl"]["habit"]["pred_minus_res"]["ci_hi"], 3)])
    for t in ("0.1", "0.25", "0.5", "0.75"):
        q = S["ctrl"]["quantiles_habit"][t]; s2 = S["ctrl"]["quantiles_same_position"][t]
        print(f"   quantile {t}: habit residual slope {q['beta']:.3f} [{q['ci_lo']:.3f}, {q['ci_hi']:.3f}] share>0 {q['share_gt0']:.2f} | same-position residual slope {s2['slope']:.3f} [{s2['ci'][0]:.3f}, {s2['ci'][1]:.3f}]")
    print("   10th percentile of release minus re-press (median over performances):", round(S["ctrl"]["p10_release_minus_press_ms"], 1), "ms")

if PART in ("all", "ref"):
    C = load_cache(); EV = pd.read_pickle(DER / "handover_events.pkl"); NT = pd.read_pickle(DER / "handover_notes.pkl")
    TH = np.arange(16, 113, 8); rows, depth = [], []
    for f, g in EV.groupby("midi_filename"):
        ct, cv = C[f]["cc64"][:, 0], C[f]["cc64"][:, 1]; cross = {}
        for e, r in zip(g.event.values, g.r.values):
            i = np.searchsorted(ct, r, side="right"); j = np.searchsorted(ct, r + 0.9); st, sv = ct[i:j], cv[i:j]
            up = np.where(sv >= PLAN["pedal_threshold"])[0]
            if len(up) == 0:
                continue
            m = int(np.argmin(sv[:up[0] + 1]))
            if sv[m] >= 16:
                continue
            tt = [st[m + np.where(sv[m:] >= th)[0][0]] if np.any(sv[m:] >= th) else np.nan for th in TH]
            if not np.any(np.isnan(tt)):
                cross[e] = np.array(tt)
        if len(cross) < 30:
            continue
        n = NT[(NT.midi_filename == f) & NT.event.isin(list(cross))]
        T = np.vstack([cross[e] for e in n.event.values]); anc = n.anchor.values; off = n.off.values
        keep = (off > T[:, 0] - 0.2) & (off < T[:, -1] + 0.4)
        if keep.sum() < 60:
            continue
        T, anc, off, ev_ids = T[keep], anc[keep], off[keep], n.event.values[keep]
        L = off[:, None] - T; mad = np.median(np.abs(L - np.median(L, axis=0)), axis=0)
        an = off - anc; mad_on = np.median(np.abs(an - np.median(an)))
        # surrogate: pair each release with the stroke of another event, re-anchored to this event's onset
        ue = np.unique(ev_ids); pos = pd.Series(np.arange(len(ue)), index=ue)[ev_ids].values
        Te = np.vstack([cross[e] for e in ue]); Ae = np.array([g.set_index("event").anchor[e] for e in ue])
        mads = []
        for _ in range(20):
            pm_ = rng.permutation(len(ue)); Ts = (Te[pm_] - Ae[pm_][:, None] + Ae[:, None])[pos]
            Ls = off[:, None] - Ts; mads.append(np.median(np.abs(Ls - np.median(Ls, axis=0)), axis=0))
        mad_sur = np.mean(mads, axis=0)
        # pedal depth at release, for releases during the down-stroke
        during = (off >= T[:, 0]) & (off <= T[:, -1])
        if during.sum() >= 20:
            kk = np.searchsorted(ct, off[during], side="right") - 1
            depth.append({"midi_filename": f, "depth_at_release": float(np.median(cv[np.clip(kk, 0, None)])), "n": int(during.sum())})
        rows.append({"midi_filename": f, "theta_star": int(TH[np.argmin(mad)]), "mad_onset": mad_on * 1000, **{f"mad{t}": v * 1000 for t, v in zip(TH, mad)},
                     **{f"sur{t}": v * 1000 for t, v in zip(TH, mad_sur)}})
    R = pd.DataFrame(rows).join(M[["pianist", "piece", "year"]], on="midi_filename")
    for t in (64, 96):
        R[f"lr_onset_{t}"] = np.log(R[f"mad{t}"] / R.mad_onset); R[f"lr_sur_{t}"] = np.log(R[f"mad{t}"] / R[f"sur{t}"])
    R.to_csv(RES / "reference_point.csv", index=False)
    Dp = pd.DataFrame(depth).join(M[["pianist", "piece", "year"]], on="midi_filename"); Dp.to_csv(RES / "depth_at_release.csv", index=False)
    # editions and repeat entrants
    ed = Dp.groupby("year").depth_at_release.mean()
    pe = Dp.groupby(["pianist", "year"]).depth_at_release.mean().reset_index()
    rep = pe.groupby("pianist").filter(lambda g: len(g) == 2)
    pairs = []
    for p_, g in rep.groupby("pianist"):
        g = g.sort_values("year"); y1, y2 = g.year.values
        others = Dp[Dp.pianist != p_].groupby("year").depth_at_release.mean()
        pairs.append({"pianist": p_, "own_diff": g.depth_at_release.values[1] - g.depth_at_release.values[0], "edition_diff": others[y2] - others[y1]})
    P = pd.DataFrame(pairs)
    S["ref"] = {"n_perf": len(R), "log_mad_ratio_vs_onset_64": pool(R, "lr_onset_64"), "log_mad_ratio_vs_onset_96": pool(R, "lr_onset_96"),
                "log_mad_ratio_vs_surrogate_64": pool(R, "lr_sur_64"), "log_mad_ratio_vs_surrogate_96": pool(R, "lr_sur_96"),
                "median_mad_ms": {**{str(t): float(R[f"mad{t}"].median()) for t in TH}, "onset": float(R.mad_onset.median())},
                "median_sur_ms": {str(t): float(R[f"sur{t}"].median()) for t in TH},
                "theta_star_counts": R.theta_star.value_counts().sort_index().to_dict(),
                "depth": {"n_perf": len(Dp), "median": float(Dp.depth_at_release.median()), "edition_means": ed.round(2).to_dict(),
                          "kruskal_p": float(stats.kruskal(*[g.depth_at_release.values for _, g in Dp.groupby("year")]).pvalue),
                          "edition_share": None},
                "repeat_entrants": {"n": len(P), "r_own_vs_edition": list(stats.pearsonr(P.own_diff, P.edition_diff)) if len(P) > 3 else None,
                                    "sign_agreement": float(np.mean(np.sign(P.own_diff) == np.sign(P.edition_diff))) if len(P) else None,
                                    "slope": float(np.polyfit(P.edition_diff, P.own_diff, 1)[0]) if len(P) > 3 else None}}
    # edition share of variance in depth at release (crossed pianist / piece / edition)
    from release_common import lmm
    y = (Dp.depth_at_release - Dp.depth_at_release.mean()) / Dp.depth_at_release.std()
    v = lmm(y.values, np.ones((len(Dp), 1)), [Dp[c].astype(str).values for c in ["pianist", "piece", "year"]])["vc"]; v = v / v.sum()
    S["ref"]["depth"]["edition_share"] = {"resid": float(v[0]), "pianist": float(v[1]), "piece": float(v[2]), "edition": float(v[3])}
    print("ref: perf", len(R), "| log MAD ratio stroke(64)/onset", round(S["ref"]["log_mad_ratio_vs_onset_64"]["beta"], 3),
          "| stroke(96)/onset", round(S["ref"]["log_mad_ratio_vs_onset_96"]["beta"], 3), "| real/surrogate at 64", round(S["ref"]["log_mad_ratio_vs_surrogate_64"]["beta"], 3),
          "at 96", round(S["ref"]["log_mad_ratio_vs_surrogate_96"]["beta"], 3))
    print("   median MAD ms:", {k: round(v, 1) for k, v in S["ref"]["median_mad_ms"].items()})
    print("   depth at release: median", round(S["ref"]["depth"]["median"], 1), "| edition means", S["ref"]["depth"]["edition_means"], "| KW p", S["ref"]["depth"]["kruskal_p"],
          "| shares", {k: round(v, 3) for k, v in S["ref"]["depth"]["edition_share"].items()})
    print("   repeat entrants:", S["ref"]["repeat_entrants"])

if PART in ("all", "pred"):
    C = load_cache(); EV = pd.read_pickle(DER / "handover_events.pkl"); NT = pd.read_pickle(DER / "handover_notes.pkl")
    NT = NT[(NT.off - NT.anchor) < 2.0].copy(); rows = []
    for f, g in NT.groupby("midi_filename"):
        if g.event.nunique() < 40:
            continue
        pr, rl = crossings(C[f]["cc64"], PLAN["pedal_threshold"])
        # sound offset = key release, unless the pedal is down then: the next pedal release
        down = ped_value(C[f]["cc64"], g.off.values) >= PLAN["pedal_threshold"]
        k = np.searchsorted(rl, g.off.values)
        snd = np.where(down & (k < len(rl)), rl[np.minimum(k, len(rl) - 1)], g.off.values)
        errs = {"sound_offset": [], "onset": [], "pedal": [], "blend": []}
        for tr_par, te_par in ((0, 1), (1, 0)):
            tr = g[g.event % 2 == tr_par]; te = g[g.event % 2 == te_par]; m_te = (g.event % 2 == te_par).values
            d_on = np.median(tr.off - tr.onset); d_p = np.median(tr.off - tr.p)
            p_on_tr = tr.onset + d_on; p_p_tr = tr.p + d_p
            ws = np.linspace(0, 1, 11); w = ws[np.argmin([np.median(np.abs(tr.off - (w_ * p_on_tr + (1 - w_) * p_p_tr))) for w_ in ws])]
            errs["sound_offset"].append(np.abs(snd[m_te] - te.off.values)); errs["onset"].append(np.abs(te.off - (te.onset + d_on)).values)
            errs["pedal"].append(np.abs(te.off - (te.p + d_p)).values); errs["blend"].append(np.abs(te.off - (w * (te.onset + d_on) + (1 - w) * (te.p + d_p))).values)
        r = {k: float(np.median(np.concatenate(v)) * 1000) for k, v in errs.items()}
        rm = {k: float(np.mean(np.concatenate(v)) * 1000) for k, v in errs.items()}
        rows.append({"midi_filename": f, **{f"mae_{k}": v for k, v in r.items()}, **{f"mean_{k}": v for k, v in rm.items()}})
    R = pd.DataFrame(rows).join(M[["pianist", "piece", "year"]], on="midi_filename")
    for k in ("pedal", "blend", "sound_offset"):  # median absolute error (+1 ms so that an error of 0 stays finite) and mean absolute error
        R[f"lr_{k}"] = np.log((R[f"mae_{k}"] + 1) / (R.mae_onset + 1)); R[f"lrm_{k}"] = np.log(R[f"mean_{k}"] / R.mean_onset)
    R.to_csv(RES / "release_prediction.csv", index=False)
    S["pred"] = {"n_perf": len(R), "median_mae_ms": {k: float(R[f"mae_{k}"].median()) for k in ("sound_offset", "onset", "pedal", "blend")},
                 "median_mean_abs_error_ms": {k: float(R[f"mean_{k}"].median()) for k in ("sound_offset", "onset", "pedal", "blend")},
                 "log_ratio_vs_onset": {k: pool(R, f"lr_{k}", B=300) for k in ("pedal", "blend", "sound_offset")},
                 "log_ratio_vs_onset_mean_error": {k: pool(R, f"lrm_{k}", B=300) for k in ("pedal", "blend", "sound_offset")},
                 "share_improved": {k: float((R[f"mae_{k}"] < R.mae_onset).mean()) for k in ("pedal", "blend")}}
    print("pred: perf", len(R), "| median MAE ms", {k: round(v, 1) for k, v in S["pred"]["median_mae_ms"].items()},
          "| ratio vs onset (pedal, blend)", round(np.exp(S["pred"]["log_ratio_vs_onset"]["pedal"]["beta"]), 3), round(np.exp(S["pred"]["log_ratio_vs_onset"]["blend"]["beta"]), 3),
          "| improved", S["pred"]["share_improved"])
save_json(S, out_path)
