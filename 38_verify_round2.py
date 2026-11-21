"""Step 38 - independent checks for step 36.

1. relay: observed counts of pedal-ups 30-210 ms before and 0-60 ms after a chord onset, by a plain bisect loop, compared with
   the vectorised window_counts of step 36 (all performances).
2. ctrl: same-score-position slope of the 10th release percentile with quintile instead of decile bins (sensitivity of the binning).
3. pred: median absolute error of the onset-only and re-press-only predictors for 25 performances by a plain loop.
4. sensor scale: do edition differences in pedal depth at release follow each edition's overall controller-64 distribution
   (95th percentile of values; median value while down)? If so, instrument regulation and sensor scaling cannot be separated.
"""
import bisect
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import DER, PLAN, RES, clusters, crossings, demean2, load_cache

spec = importlib.util.spec_from_file_location("s36", Path(__file__).resolve().parent / "36_relay_and_control.py")
src = (Path(__file__).resolve().parent / "36_relay_and_control.py").read_text()
ns = {"np": np, "PLAN": PLAN}
exec(src[src.index("a0, a1 = np.array"):src.index('if PART in ("all", "relay")')], ns)
C = load_cache(); out = {}
diff = 0; tot = 0
for f, d in C.items():
    pr, rl = crossings(d["cc64"], 64)
    if len(rl) < 20 or len(pr) < 2:
        continue
    j = np.searchsorted(pr, rl); ok = j < len(pr); r = rl[ok]; r = r[(pr[j[ok]] - r) < PLAN["change_max_gap_s"]]
    r = r[np.searchsorted(pr, r) - 1 >= 0]
    on = d["on"]; cid = clusters(on, 0.03); cnt = np.bincount(cid); first = np.r_[True, np.diff(cid) > 0]; ch = list(on[first][cnt >= 2])
    va, vh, _ = ns["window_counts"](r, np.array(ch))
    la = lh = 0
    for x in r:
        k = bisect.bisect_right(ch, x)
        if k < len(ch) and 0.03 < ch[k] - x < 0.21:
            la += 1
        if k > 0 and 0 <= x - ch[k - 1] < 0.06:
            lh += 1
    diff += abs(la - va) + abs(lh - vh); tot += la + lh
out["relay_counts_abs_diff"] = int(diff); out["relay_counts_total"] = int(tot)
print("1. relay window counts, loop vs vectorised: total |difference|", diff, "of", tot)

S = json.loads((RES / "round2_summary.json").read_text())
MT = pd.read_pickle(DER / "handover_events_matched.pkl"); MT = MT[MT.groupby("set").midi_filename.transform("nunique") >= 2].copy()
MT["res_sp"] = demean2({"x": (MT.p - MT.anchor).values}, MT.set.values, MT.midi_filename.values)["x"]
NS = pd.read_pickle(DER / "handover_notes.pkl").merge(MT[["midi_filename", "event", "set", "res_sp"]], on=["midi_filename", "event"])
NS = NS[(NS.off - NS.anchor) < 2.0].copy(); NS["yr"] = demean2({"y": (NS.off - NS.anchor).values}, NS.set.values, NS.midi_filename.values)["y"]
NS["q5"] = pd.qcut(NS.res_sp, 5, labels=False)
t = NS.groupby("q5").agg(yq=("yr", lambda s: s.quantile(0.10)), xr=("res_sp", "mean"))
out["ctrl_p10_slope_quintiles"] = float(np.polyfit(t.xr, t.yq, 1)[0]); out["ctrl_p10_slope_deciles_step36"] = S["ctrl"]["quantiles_same_position"]["0.1"]["slope"]
print(f"2. same-position 10th-percentile slope: quintile bins {out['ctrl_p10_slope_quintiles']:.3f} vs decile bins (step 36) {out['ctrl_p10_slope_deciles_step36']:.3f}")

P = pd.read_csv(RES / "release_prediction.csv"); NT = pd.read_pickle(DER / "handover_notes.pkl"); NT = NT[(NT.off - NT.anchor) < 2.0]
rng = np.random.default_rng(3); dd = []
for f in rng.choice(P.midi_filename.values, 25, replace=False):
    g = NT[NT.midi_filename == f]; e_on, e_p = [], []
    for tr_par in (0, 1):
        tr = g[g.event % 2 == tr_par]; te = g[g.event % 2 != tr_par]
        don = sorted(tr.off - tr.onset)[len(tr) // 2] if len(tr) % 2 else np.median(tr.off - tr.onset); dp = np.median(tr.off - tr.p)
        e_on += [abs(o - (s + don)) for o, s in zip(te.off, te.onset)]; e_p += [abs(o - (p + dp)) for o, p in zip(te.off, te.p)]
    row = P[P.midi_filename == f].iloc[0]
    dd.append((np.median(e_on) * 1000 - row.mae_onset, np.median(e_p) * 1000 - row.mae_pedal))
out["pred_max_abs_diff_ms"] = float(np.max(np.abs(dd)))
print("3. prediction errors, loop vs step 36 (25 performances): max |difference|", round(out["pred_max_abs_diff_ms"], 6), "ms")
from scipy import stats
from release_common import load_meta
M = load_meta().set_index("midi_filename"); rows = []
for f, d in C.items():
    v = d["cc64"][:, 1]
    if len(v) >= 50:
        rows.append({"midi_filename": f, "p95": np.percentile(v, 95), "med_down": np.median(v[v >= 64]) if np.any(v >= 64) else np.nan})
E = pd.DataFrame(rows).join(M[["year"]], on="midi_filename").groupby("year")[["p95", "med_down"]].median()
E["depth_at_release"] = pd.read_csv(RES / "depth_at_release.csv").groupby("year").depth_at_release.mean()
out["sensor_r_p95"] = float(stats.pearsonr(E.p95, E.depth_at_release)[0]); out["sensor_r_med_down"] = float(stats.pearsonr(E.med_down, E.depth_at_release)[0])
print(f"4. editions: depth at release vs 95th percentile of controller values r = {out['sensor_r_p95']:.2f}; vs median value while down r = {out['sensor_r_med_down']:.2f}")
(RES / "verification_round2.json").write_text(json.dumps(out, indent=2))
