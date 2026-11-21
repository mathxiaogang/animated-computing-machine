"""Step 23 - H1: hand-foot handover at sustain-pedal changes (avoidance, handover, peak latency), the una corda control,
and E3 (which notes are cut before the foot catches them).

Usage:
    python 23_handover.py [thr=64] [gap=0.6] [tag=]

Writes results/h1_performance_indices<tag>.csv, results/h1_summary<tag>.json and results/derived/h1_lags<tag>.pkl.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import (DER, PLAN, RES, handover_events, handover_indices, kde_curve, kde_peak, load_cache, load_meta,
                            pool, save_json)

OPTS = dict(a.split("=", 1) for a in sys.argv[1:])
THR = float(OPTS.get("thr", PLAN["pedal_threshold"])); GAP = float(OPTS.get("gap", PLAN["change_max_gap_s"]))
TAG = OPTS.get("tag", ""); SFX = f"_{TAG}" if TAG else ""
SHIFT = OPTS.get("surrogate", "permute")  # robustness: "shift" = re-press moved by a random +-0.5 s instead
C = load_cache(); M = load_meta().set_index("midi_filename")
rng = np.random.default_rng(PLAN["seed"])

rows, lags, sur_lags, soft_rows = [], [], [], []
for f, d in C.items():
    E = handover_events(d, thr=THR, gap_max=GAP)
    if len(E) and E.event.nunique() >= PLAN["min_events"]:
        if SHIFT == "shift":
            # alternative surrogate: keep every event's own press but move it by a random offset of up to +-0.5 s
            ev = E.drop_duplicates("event")
            a0, a1 = np.array(PLAN["avoid_window_ms"]) / 1000; h0, h1 = np.array(PLAN["handover_window_ms"]) / 1000
            pos = pd.Series(np.arange(len(ev)), index=ev.event.values)[E.event.values].values
            sa, sh = [], []
            for _ in range(PLAN["n_surrogates"]):
                pp = ev.p.values + rng.uniform(-0.5, 0.5, len(ev))
                lg = E.off.values - pp[pos]; sa.append(((lg > a0) & (lg < a1)).sum()); sh.append(((lg >= h0) & (lg < h1)).sum())
            ra = ((E.lag > a0) & (E.lag < a1)).sum(); rh = ((E.lag >= h0) & (E.lag < h1)).sum()
            r = {"n_events": len(ev), "n_notes": len(E), "A": float(np.log((ra + .5) / (np.mean(sa) + .5))), "H": float(np.log((rh + .5) / (np.mean(sh) + .5))),
                 "cut_share": float((E.lag < 0).mean())}
        else:
            r = handover_indices(E, rng, return_sur=True)
            sur_lags.append(pd.DataFrame({"midi_filename": f, "lag": r.pop("sur_lags")}))
        multi = E[E["rank"] != "single"]
        for rk in ["top", "inner", "bottom"]:
            x = multi[multi["rank"] == rk]
            r[f"cut_{rk}"] = float((x.lag < 0).mean()) if len(x) >= 10 else np.nan
        rows.append({"midi_filename": f, **r})
        lags.append(E.assign(midi_filename=f)[["midi_filename", "event", "lag", "rank", "pitch", "gap"]])
    if not TAG:  # una corda control: same procedure on controller 67 (no dampers involved)
        d67 = dict(d); d67["cc64"] = d["cc67"]
        E67 = handover_events(d67)
        if len(E67) and E67.event.nunique() >= PLAN["min_events"]:
            soft_rows.append({"midi_filename": f, **{k: v for k, v in handover_indices(E67, rng).items()}})

R = pd.DataFrame(rows).join(M[["pianist", "piece", "year", "group", "sus_frac", "era"]], on="midi_filename")
R.to_csv(RES / f"h1_performance_indices{SFX}.csv", index=False)
L = pd.concat(lags, ignore_index=True).join(M[["pianist"]], on="midi_filename"); L.to_pickle(DER / f"h1_lags{SFX}.pkl")
print(f"performances {len(R)} | pianists {R.pianist.nunique()} | events {R.n_events.sum()} | held notes {R.n_notes.sum()}")

S = {"n_perf": len(R), "n_pianists": int(R.pianist.nunique()), "n_events": int(R.n_events.sum()), "n_notes": int(R.n_notes.sum()),
     "median_cut_share": float(R.cut_share.median()), "H1a_A": pool(R, "A"), "H1b_H": pool(R, "H"),
     "share_perf_A_lt0": float((R.A < 0).mean()), "share_perf_H_gt0": float((R.H > 0).mean())}
print("H1a A:", {k: round(v, 4) for k, v in S["H1a_A"].items() if isinstance(v, float)})
print("H1b H:", {k: round(v, 4) for k, v in S["H1b_H"].items() if isinstance(v, float)})

if not TAG:
    # H1c: peak of the pooled release-minus-re-press density, pianist-cluster bootstrap
    peak = kde_peak(L.lag.values)
    SL = pd.concat(sur_lags, ignore_index=True)
    c, real_d = kde_curve(L.lag.values); _, sur_d = kde_curve(SL.lag.values)
    excess = real_d / real_d.sum() - sur_d / sur_d.sum()
    ex_peak = float(c[np.argmax(excess)])
    pianists = L.pianist.unique(); gi = {k: np.where(L.pianist.values == k)[0] for k in pianists}
    bs, bs_ex = [], []
    Lsur = SL.join(M[["pianist"]], on="midi_filename"); gs = {k: np.where(Lsur.pianist.values == k)[0] for k in pianists}
    for _ in range(PLAN["bootstrap_B"]):
        draw = rng.choice(pianists, len(pianists))
        s = np.concatenate([gi[k] for k in draw]); bs.append(kde_peak(L.lag.values[s]))
        ss = np.concatenate([gs[k] for k in draw if k in gs])
        _, rd = kde_curve(L.lag.values[s]); _, sd = kde_curve(SL.lag.values[ss])
        bs_ex.append(float(c[np.argmax(rd / rd.sum() - sd / sd.sum())]))
    S["H1c"] = {"peak_ms": peak, "ci": np.percentile(bs, [2.5, 97.5]).tolist(), "excess_peak_ms": ex_peak,
                "excess_ci": np.percentile(bs_ex, [2.5, 97.5]).tolist(), "bound_ms": PLAN["feedforward_bound_ms"],
                "supported": bool(np.percentile(bs, 97.5) < PLAN["feedforward_bound_ms"])}
    print("H1c:", S["H1c"])
    # pooled curves for the figure
    cw, rw = kde_curve(L.lag.values, grid_ms=[-300, 400]); _, sw = kde_curve(SL.lag.values, grid_ms=[-300, 400])
    pd.DataFrame({"lag_ms": cw, "real": rw / rw.sum(), "surrogate": sw / sw.sum()}).to_csv(RES / "h1_density_curves.csv", index=False)
    # E3: cut share by voice position (within performance, top vs bottom)
    e3 = R.dropna(subset=["cut_top", "cut_bottom"]).copy(); e3["top_minus_bottom"] = e3.cut_top - e3.cut_bottom
    e3["inner_minus_top"] = e3.cut_inner - e3.cut_top
    S["E3"] = {"n_perf": len(e3), "median_cut_top": float(e3.cut_top.median()), "median_cut_inner": float(e3.cut_inner.median()),
               "median_cut_bottom": float(e3.cut_bottom.median()), "top_minus_bottom": pool(e3, "top_minus_bottom", B=200),
               "inner_minus_top": pool(e3.dropna(subset=["inner_minus_top"]), "inner_minus_top", B=200)}
    print("E3:", {k: v for k, v in S["E3"].items() if not isinstance(v, dict)}, "| top-bottom", round(S["E3"]["top_minus_bottom"]["beta"], 4))
    # una corda control
    SR = pd.DataFrame(soft_rows).join(M[["pianist", "piece", "year"]], on="midi_filename")
    SR.to_csv(RES / "h1_unacorda_control_indices.csv", index=False)
    S["control_unacorda"] = {"n_perf": len(SR), "A": pool(SR, "A", B=200), "H": pool(SR, "H", B=200)}
    print("una corda control: A", round(S["control_unacorda"]["A"]["beta"], 4), "H", round(S["control_unacorda"]["H"]["beta"], 4), "n", len(SR))
save_json(S, RES / f"h1_summary{SFX}.json")
