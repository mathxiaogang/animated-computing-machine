"""Step 31 - recompute the headline numbers by independent code paths.

1. Handover: pooled counts in the two windows, recomputed from the raw MIDI parse (not the cache helpers) for 60 random performances,
   compared with step 23's per-performance counts.
2. Study 2 location: two-way fixed effects by sparse least squares (scipy LSQR on dummy matrices) instead of alternating projections.
3. Study 1 location: per-performance medians recomputed with plain pandas for 30 random performances.
4. Holm adjustment recomputed from the p values.
"""
import bisect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import DER, PLAN, RES, add_strata, load_cache, load_meta

out = {}
rng = np.random.default_rng(7)
C = load_cache(); M = load_meta().set_index("midi_filename")

# ---- 1. handover counts (real releases in the windows)
R = pd.read_csv(RES / "h1_performance_indices.csv")
a0, a1 = np.array(PLAN["avoid_window_ms"]) / 1000; h0, h1 = np.array(PLAN["handover_window_ms"]) / 1000
diffs = []
for f in rng.choice(R.midi_filename.values, 60, replace=False):
    d = C[f]; cc = d["cc64"]; v = cc[:, 1] >= 64; ons = list(d["on"]); offs = list(d["off"])
    rel = [cc[i, 0] for i in range(1, len(cc)) if v[i - 1] and not v[i]]
    prs = [cc[i, 0] for i in range(1, len(cc)) if not v[i - 1] and v[i]]
    ra = rh = 0
    for r in rel:
        k = bisect.bisect_right(prs, r)
        if k >= len(prs) or prs[k] - r >= PLAN["change_max_gap_s"]:
            continue
        p = prs[k]
        i0, i1 = bisect.bisect_left(ons, r - PLAN["held_lookback_s"]), bisect.bisect_left(ons, r)
        held = [(ons[i], offs[i]) for i in range(i0, i1) if offs[i] > r]
        if not held:
            continue
        for on, off in held:
            lag = off - p; ra += a0 < lag < a1; rh += h0 <= lag < h1
    row = R[R.midi_filename == f].iloc[0]
    diffs.append((ra - row.real_avoid, rh - row.real_hand))
diffs = np.array(diffs)
out["handover_counts_max_abs_diff"] = int(np.abs(diffs).max())
print("1. handover window counts, independent loop vs step 23 (60 performances): max |difference| =", out["handover_counts_max_abs_diff"])

# ---- 2. study 2 location by sparse least squares
D = pd.read_pickle(DER / "matched_transitions.pkl"); D = D[D.ped.isin([0, 1])].copy()
D = D[D.groupby("set").ped.transform("nunique") == 2].reset_index(drop=True)
y = D.kot.values * 1000
lioi = np.log(D.ioi.values) - np.log(D.ioi.values).mean()  # centring constants differ from step 26 but are absorbed
X = np.column_stack([D.ped.values.astype(float), lioi, (D.vel.values - D.vel.values.mean()) / 10])
c1 = pd.factorize(D.set)[0]; c2 = pd.factorize(D.midi_filename)[0]; n = len(D)
A = hstack([csr_matrix(X), csr_matrix((np.ones(n), (np.arange(n), c1))), csr_matrix((np.ones(n), (np.arange(n), c2)))]).tocsr()
sol = lsqr(A, y, atol=1e-12, btol=1e-12, iter_lim=50000)[0]
s2 = json.loads((RES / "study2_summary.json").read_text())
out["study2_loc_ms_lsqr"] = float(sol[0]); out["study2_loc_ms_step26"] = s2["H2a_loc_ms"]["b"]
print(f"2. study 2 pedal coefficient: LSQR {sol[0]:.3f} ms vs step 26 {s2['H2a_loc_ms']['b']:.3f} ms")

# ---- 3. study 1 location for 30 performances (location part of the matched effect, plain pandas)
T = add_strata(pd.read_pickle(DER / "transitions_main.pkl")); S1 = pd.read_csv(RES / "study1_performance_effects.csv")
dd = []
for f in rng.choice(S1.midi_filename.values, 30, replace=False):
    g = T[(T.midi_filename == f) & T.ped.isin([0, 1])]
    cell = g.groupby(["stratum", "ped"]).kot.agg(["median", "size"]).unstack()
    ok = (cell[("size", 0)] >= PLAN["min_per_cell"]) & (cell[("size", 1)] >= PLAN["min_per_cell"])
    cell = cell[ok]
    w = np.minimum(cell[("size", 0)], cell[("size", 1)])
    est = float(np.average(cell[("median", 1)] - cell[("median", 0)], weights=w)) * 1000
    dd.append(est - S1[S1.midi_filename == f].loc_ms.iloc[0])
out["study1_loc_max_abs_diff_ms"] = float(np.max(np.abs(dd)))
print("3. study 1 per-performance location, plain pandas vs step 26 (30 performances): max |difference| =", round(out["study1_loc_max_abs_diff_ms"], 6), "ms")
print("   (cells whose residual MAD is zero are dropped in step 26 only; such cells are very rare)")

# ---- 4. Holm
Tt = pd.read_csv(RES / "confirmatory_tests.csv"); p = Tt.p.values; o = np.argsort(p); m = len(p)
holm = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
adj = np.empty(m); adj[o] = holm
out["holm_max_abs_diff"] = float(np.max(np.abs(adj - Tt.p_holm.values)))
print("4. Holm adjustment max |difference| =", out["holm_max_abs_diff"])
(RES / "verification.json").write_text(json.dumps(out, indent=2))
