"""Step 22 - extract melodic transitions (with key overlap time and pedal state), onset clusters (chords) and data-quality counts.

Usage:
    python 22_extract_transitions.py

Reads results/derived/notes_cache.pkl (step 19). Writes results/derived/transitions_<variant>.pkl, chords.pkl and
results/data_quality.csv. The variant "main" follows the frozen plan; the others are the robustness variants of step 29.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import DER, PLAN, RES, clusters, load_cache, pedal_state, transitions

C = load_cache()
VARIANTS = {
    "main": {},
    "thr32": {"thr": 32}, "thr96": {"thr": 96},
    "win025": {"w": 0.25}, "win060": {"w": 0.6},
    "bottom": {"voice": "bottom"}, "separated": {"voice": "separated"},
    "interval12": {"interval_max": 12}, "ioi015_080": {"ioi_range": [0.15, 0.8]}, "kot1000": {"kot_max": 1.0},
}
out = {k: [] for k in VARIANTS}
chords, qual = [], []
for f, d in C.items():
    for k, kw in VARIANTS.items():
        T = transitions(d, **kw)
        T.insert(0, "midi_filename", f)
        out[k].append(T)
    # onset clusters with >= 3 notes spanning < 60 ms (onset precision, H3a)
    on, p, v = d["on"], d["pitch"].astype(int), d["vel"].astype(int)
    cid = clusters(on, PLAN["cluster_gap_s"])
    s = pd.DataFrame({"c": cid, "on": on, "p": p, "v": v})
    g = s.groupby("c").agg(n=("on", "size"), t=("on", "min"), t1=("on", "max"), vel=("v", "mean"), top=("p", "max"), bot=("p", "min"))
    g = g[(g.n >= PLAN["chord_min_notes"]) & ((g.t1 - g.t) < PLAN["chord_span_s"])].copy()
    g["spread"] = g.t1 - g.t
    st, _, _ = pedal_state(d["cc64"], g.t.values, PLAN["pedal_window_s"], PLAN["pedal_threshold"])
    so, _, _ = pedal_state(d["cc67"], g.t.values, PLAN["pedal_window_s"], PLAN["pedal_threshold"])
    g["ped"], g["soft"] = st, so
    g.insert(0, "midi_filename", f)
    chords.append(g.reset_index(drop=True))
    # data quality: notes re-struck before any key release was recorded (the parser closes them at the re-strike)
    order = np.lexsort((on, p)); same = p[order][1:] == p[order][:-1]
    restruck = (np.abs(d["off"][order][:-1] - on[order][1:]) < 1e-9) & same
    qual.append({"midi_filename": f, "notes": len(on), "restruck_without_release": int(restruck.sum()),
                 "neg_or_zero_duration": int((d["off"] <= on).sum())})

for k, lst in out.items():
    T = pd.concat(lst, ignore_index=True)
    T.to_pickle(DER / f"transitions_{k}.pkl")
    print(f"{k:12s} transitions {len(T):8d} | down {int((T.ped == 1).sum()):7d} | up {int((T.ped == 0).sum()):7d} | change {int((T.ped == -1).sum()):7d}")
CH = pd.concat(chords, ignore_index=True); CH.to_pickle(DER / "chords.pkl")
print("chords", len(CH), CH.ped.value_counts().to_dict())
Q = pd.DataFrame(qual); Q.to_csv(RES / "data_quality.csv", index=False)
print("notes", Q.notes.sum(), "| re-struck without release:", Q.restruck_without_release.sum(), f"({Q.restruck_without_release.sum() / Q.notes.sum():.4%})",
      "| non-positive durations:", Q.neg_or_zero_duration.sum())
