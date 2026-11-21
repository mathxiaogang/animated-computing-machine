"""Step 28 - E2 individual differences and E4 junior vs senior editions (exploratory).

E2: share of variance from the pianist and split-half reliability across a pianist's performances, for the handover indices
    (H, A, cut share) and the study 1 release effects; relation of each pianist's handover to the amount of sustain pedalling.
E4: senior minus junior editions (group is constant within an edition, so the contrast is confounded with edition; descriptive).

Usage:
    python 28_individuality.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import PLAN, RES, lmm, pool, save_json

rng = np.random.default_rng(PLAN["seed"])
H = pd.read_csv(RES / "h1_performance_indices.csv")
S1 = pd.read_csv(RES / "study1_performance_effects.csv")


def shares(d, col):
    d = d.dropna(subset=[col])
    y = (d[col] - d[col].mean()) / d[col].std()
    f = lmm(y.values, np.ones((len(d), 1)), [d[g].astype(str).values for g in ["pianist", "piece", "year"]])
    v = f["vc"] / f["vc"].sum()
    return {"resid": float(v[0]), "pianist": float(v[1]), "piece": float(v[2]), "edition": float(v[3])}


def split_half(d, col, reps=200):
    d = d.dropna(subset=[col]); g = d.groupby("pianist")[col].apply(list); g = g[g.apply(len) >= 2]
    rs = []
    for _ in range(reps):
        a, b = [], []
        for v in g:
            v = rng.permutation(v); h = len(v) // 2
            a.append(np.mean(v[:h])); b.append(np.mean(v[h:]))
        rs.append(stats.pearsonr(a, b)[0])
    r = float(np.mean(rs))
    return {"n_pianists": int(len(g)), "r_half": r, "spearman_brown": 2 * r / (1 + r)}


S = {"E2": {}, "E4": {}}
for col in ["H", "A", "cut_share"]:
    S["E2"][f"handover_{col}"] = {"variance_shares": shares(H, col), "split_half": split_half(H, col)}
for col in ["loc_ms", "logscale"]:
    S["E2"][f"study1_{col}"] = {"variance_shares": shares(S1, col), "split_half": split_half(S1, col)}
pm = H.groupby("pianist")[["H", "A", "cut_share", "sus_frac"]].mean()
S["E2"]["pianist_H_vs_sustain_share"] = dict(zip(["r", "p"], stats.spearmanr(pm.H, pm.sus_frac)))
S["E2"]["pianist_cut_vs_sustain_share"] = dict(zip(["r", "p"], stats.spearmanr(pm.cut_share, pm.sus_frac)))
S["E2"]["n_pianists"] = len(pm)
for k, v in S["E2"].items():
    print("E2", k, v)

for col, d in [("H", H), ("A", H), ("cut_share", H), ("loc_ms", S1), ("logscale", S1)]:
    d = d.dropna(subset=[col]).copy(); d["senior"] = (d.group == "senior").astype(float)
    p = pool(d, col, extra_X=("senior",), B=0)
    S["E4"][col] = {"senior_minus_junior": p["betas"][1], "se": p["ses"][1], "z": p["betas"][1] / p["ses"][1],
                    "p": float(2 * stats.norm.sf(abs(p["betas"][1] / p["ses"][1]))), "n": p["n"],
                    "edition_means": d.groupby(["year", "group"])[col].mean().round(4).reset_index().values.tolist()}
    print("E4", col, {k: round(v, 4) for k, v in S["E4"][col].items() if isinstance(v, float)})
save_json(S, RES / "individuality_summary.json")
