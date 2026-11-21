"""Step 42 - analyses requested in the second internal review (exploratory robustness checks).

1. Placebo-corrected foot indices: the placebo (chord onsets shifted by 0.5 s) gave slightly negative indices, so the surrogate is
   not perfectly calibrated. Per performance, observed minus placebo index, pooled.
2. Precision: per-performance indices rest on very different numbers of changes, while the pooled model weights performances
   equally. (a) Pooled estimates for performances with at least 100 changes; (b) change-weighted means with pianist-cluster
   bootstrap intervals.
3. Latency between early releases and the re-depression. If the foot timed its re-depression by reacting to the hand's early
   releases (feedback), re-depressions should rarely follow those releases by less than a plausible reaction latency. Distribution
   of re-depression minus key release for held notes released before the re-depression.

Output: results/review2_checks.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import DER, PLAN, RES, load_meta, pool, save_json

rng = np.random.default_rng(PLAN["seed"] + 42)
M = load_meta().set_index("midi_filename")
OUT = {}

# 1. placebo-corrected foot indices
R = pd.read_csv(RES / "relay_foot_indices.csv")
R["A_foot_minus_placebo"] = R.A_foot - R.A_placebo; R["H_foot_minus_placebo"] = R.H_foot - R.H_placebo
OUT["foot_placebo_corrected"] = {c: pool(R, c, B=300) for c in ("A_foot_minus_placebo", "H_foot_minus_placebo")}


# 2. precision
def wmean_ci(d, col, w, B=500):
    d = d.dropna(subset=[col]); est = float(np.average(d[col], weights=d[w]))
    pis = d.pianist.unique(); gi = {k: np.where(d.pianist.values == k)[0] for k in pis}; bs = []
    for _ in range(B):
        s = np.concatenate([gi[k] for k in rng.choice(pis, len(pis))]); bs.append(np.average(d[col].values[s], weights=d[w].values[s]))
    return {"est": est, "ci": np.percentile(bs, [2.5, 97.5]).tolist(), "n": len(d)}


H = pd.read_csv(RES / "h1_performance_indices.csv")
OUT["precision"] = {
    "hand_min100": {c: pool(H[H.n_events >= 100], c, B=300) for c in ("A", "H")},
    "hand_weighted": {c: wmean_ci(H, c, "n_events") for c in ("A", "H")},
    "foot_min100": {c: pool(R[R.n_changes >= 100], c, B=300) for c in ("A_foot", "H_foot")},
    "foot_weighted": {c: wmean_ci(R, c, "n_changes") for c in ("A_foot", "H_foot")},
    "n_hand_min100": int((H.n_events >= 100).sum()), "n_foot_min100": int((R.n_changes >= 100).sum())}

# 3. latency from early releases to the re-depression
NT = pd.read_pickle(DER / "handover_notes.pkl")
cut = NT[NT.lag < 0].copy(); cut["d_ms"] = -cut.lag * 1000
per = cut.groupby("midi_filename").d_ms.median()
OUT["early_release_latency"] = {"n_cut_notes": int(len(cut)), "median_ms": float(cut.d_ms.median()),
                                "share_below_100ms": float((cut.d_ms < 100).mean()), "share_below_150ms": float((cut.d_ms < 150).mean()),
                                "median_of_performance_medians_ms": float(per.median())}
first = cut.groupby(["midi_filename", "event"]).d_ms.max()  # earliest release of each change = largest lead
OUT["early_release_latency"]["earliest_release_per_change"] = {"n_changes": int(len(first)), "median_ms": float(first.median()),
                                                               "share_below_100ms": float((first < 100).mean())}
save_json(OUT, RES / "review2_checks.json")
for k, v in OUT["foot_placebo_corrected"].items():
    print(k, round(v["beta"], 3), [round(v["ci_lo"], 3), round(v["ci_hi"], 3)])
for k in ("hand_min100", "foot_min100"):
    print(k, {c: (round(v["beta"], 3), round(v["ci_lo"], 3), round(v["ci_hi"], 3), v["n"]) for c, v in OUT["precision"][k].items()})
for k in ("hand_weighted", "foot_weighted"):
    print(k, {c: (round(v["est"], 3), [round(x, 3) for x in v["ci"]]) for c, v in OUT["precision"][k].items()})
print("latency", OUT["early_release_latency"])
