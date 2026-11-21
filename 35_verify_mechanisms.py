"""Step 35 - independent recomputation for the mechanism analyses (step 33).

1. D1 and D2 same-score-position slopes by sparse least squares (dummy matrices + LSQR) instead of alternating projections.
2. D3: within-chord contrast computed directly (mean cut of notes with vs without motor need, in chords that contain both),
   compared in sign and size with the fixed-effects estimate.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import DER, RES

S = json.loads((RES / "mechanisms_summary.json").read_text()); out = {}


def lsqr_slope(d, y, x, cov):
    c1 = pd.factorize(d.set)[0]; c2 = pd.factorize(d.midi_filename)[0]; n = len(d)
    A = hstack([csr_matrix(d[[x, cov]].values), csr_matrix((np.ones(n), (np.arange(n), c1))), csr_matrix((np.ones(n), (np.arange(n), c2)))]).tocsr()
    return float(lsqr(A, d[y].values, atol=1e-12, btol=1e-12, iter_lim=100000)[0][0])


MT = pd.read_pickle(DER / "handover_events_matched.pkl"); MT["lioi"] = np.log(MT.ioi_next)
d1 = MT.assign(y1=MT.rel_med - MT.anchor, x1=MT.p - MT.anchor)
d1 = d1[(d1.y1 <= 2.0) & (d1.ioi_next > 0)].dropna(subset=["y1", "x1", "lioi"])
d1 = d1[d1.groupby("set").midi_filename.transform("nunique") >= 2]
out["D1_lsqr"] = lsqr_slope(d1, "y1", "x1", "lioi"); out["D1_step33"] = S["D1"]["same_position"]["b"]
d2 = MT.assign(y2=MT.rel_win_med - MT.t32)
d2 = d2[(d2.D > 0.005) & (d2.D <= 0.3) & (d2.ioi_next > 0)].dropna(subset=["y2", "D", "lioi"])
d2 = d2[d2.groupby("set").midi_filename.transform("nunique") >= 2]
out["D2_lsqr"] = lsqr_slope(d2, "y2", "D", "lioi"); out["D2_step33"] = S["D2"]["same_position"]["b"]
print(f"1. same-position slopes: D1 LSQR {out['D1_lsqr']:.4f} vs step 33 {out['D1_step33']:.4f} | D2 LSQR {out['D2_lsqr']:.4f} vs step 33 {out['D2_step33']:.4f}")

NT = pd.read_pickle(DER / "handover_notes.pkl"); NT["uid"] = NT.midi_filename + "|" + NT.event.astype(str)
g = NT.groupby(["uid", "need"]).cut.mean().unstack().dropna()
out["D3_direct_within_chord_need"] = float((g[1] - g[0]).mean()); out["D3_n_chords_with_both"] = len(g)
out["D3_step33_need"] = S["D3"]["lpm"]["need"]["b"]
print(f"2. within-chord need contrast (unadjusted, {len(g)} chords with both): {out['D3_direct_within_chord_need']:+.4f} vs fixed-effects {out['D3_step33_need']:+.4f}")
(RES / "verification_mechanisms.json").write_text(json.dumps(out, indent=2))
