"""Step 24 - align every performance to a reference performance of the same piece (banded DTW on chroma, as in step 07 of the
una corda study) and SAVE the warping paths, for all pieces with at least two performances.

Usage:
    python 24_alignment_paths.py

Writes results/derived/alignment_paths.pkl ({midi_filename: {"piece", "ref", "match"}}) and results/alignment_quality.csv.
match[i] = frame of the performance matched to frame i of the reference (10 frames per second).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import DER, PLAN, RES, load_cache, load_meta

A = PLAN["align"]; FPS = A["fps"]; BAND = A["band"]
C = load_cache(); M = load_meta()


def chroma_frames(d):
    """12-d chroma at FPS: sustained-note energy plus doubled onset emphasis, log-compressed, unit-normalised (as in step 07)."""
    T = max(d["off"].max(), d["on"].max()); n = int(np.ceil(T * FPS)) + 1
    ch = np.zeros((n, 12))
    f0 = (d["on"] * FPS).astype(int); f1 = np.maximum((d["off"] * FPS).astype(int), f0 + 1)
    pc = d["pitch"].astype(int) % 12; v = d["vel"] / 127
    np.add.at(ch, (f0, pc), 2 * v)
    for a, b, c_, w in zip(f0, f1, pc, v):
        ch[a:b, c_] += w
    ch = np.log1p(ch)
    return ch / (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)


def dtw_path(X, Y, band=BAND):
    """Banded DTW (cosine distance). Returns, for each frame of X, the matched frame of Y, and the normalised path cost."""
    N, Mm = len(X), len(Y)
    W = int(band * max(N, Mm)) + 5
    diag = np.round(np.arange(N) * (Mm - 1) / max(N - 1, 1)).astype(int)
    lo = np.clip(diag - W, 0, Mm - 1); hi = np.clip(diag + W, 0, Mm - 1)
    INF = 1e18; rows = []; prev, plo, phi = None, 0, -1
    for i in range(N):
        js = np.arange(lo[i], hi[i] + 1); c = 1.0 - Y[js] @ X[i]
        if prev is None:
            row = np.cumsum(c) if js[0] == 0 else np.full(len(js), INF)
        else:
            def get(jj):
                out = np.full(len(jj), INF); m = (jj >= plo) & (jj <= phi); out[m] = prev[jj[m] - plo]; return out
            Aa = c + np.minimum(get(js - 1), get(js)); S = np.cumsum(c)
            row = S + np.minimum.accumulate(Aa - S)
        rows.append(row); prev, plo, phi = row, lo[i], hi[i]
    i, j = N - 1, Mm - 1
    if not (lo[i] <= j <= hi[i]) or not np.isfinite(rows[i][j - lo[i]]):
        return None, np.inf
    cost = rows[i][j - lo[i]] / (N + Mm); match = np.zeros(N, dtype=int); match[i] = j

    def val(ii, jj):
        if ii < 0 or jj < 0 or not (lo[ii] <= jj <= hi[ii]):
            return INF
        return rows[ii][jj - lo[ii]]
    while i > 0 or j > 0:
        cands = [(val(i - 1, j - 1), i - 1, j - 1), (val(i - 1, j), i - 1, j), (val(i, j - 1), i, j - 1)]
        _, i, j = min(cands)
        match[i] = j if match[i] == 0 or j < match[i] else match[i]
    return match, cost


def align_piece(pc):
    P = M[M.piece == pc]
    ref = P.iloc[(P.duration - P.duration.median()).abs().argsort().iloc[0]]
    X = chroma_frames(C[ref.midi_filename])
    out, q = {ref.midi_filename: {"piece": pc, "ref": ref.midi_filename, "match": np.arange(len(X))}}, []
    for _, row in P.iterrows():
        if row.midi_filename == ref.midi_filename:
            continue
        ratio = row.duration / ref.duration; r = {"piece": pc, "ref": ref.midi_filename, "perf": row.midi_filename, "ratio": ratio}
        if not (A["ratio"][0] <= ratio <= A["ratio"][1]):
            q.append(r | {"status": "duration ratio out of range", "cost": np.nan}); continue
        match, cost = dtw_path(X, chroma_frames(C[row.midi_filename]))
        if match is None:
            q.append(r | {"status": "no path", "cost": np.nan}); continue
        q.append(r | {"status": "ok", "cost": cost}); out[row.midi_filename] = {"piece": pc, "ref": ref.midi_filename, "match": match}
    return out, q


if __name__ == "__main__":
    cnt = M.groupby("piece").size(); pieces = list(cnt[cnt >= 2].index)
    with Pool(2) as pool:
        res = pool.map(align_piece, pieces, chunksize=4)
    paths, qual = {}, []
    for o, q in res:
        paths.update(o); qual += q
    Q = pd.DataFrame(qual)
    cut = Q[Q.status == "ok"].cost.quantile(A["cost_drop_quantile"])
    bad = set(Q[(Q.status == "ok") & (Q.cost > cut)].perf)
    Q.loc[Q.perf.isin(bad), "status"] = "cost above 90th percentile"
    for b in bad:
        paths.pop(b, None)
    # a piece needs its reference plus at least one aligned performance
    per_piece = pd.Series([v["piece"] for v in paths.values()]).value_counts()
    keep = set(per_piece[per_piece >= 2].index)
    paths = {k: v for k, v in paths.items() if v["piece"] in keep}
    Q.to_csv(RES / "alignment_quality.csv", index=False)
    pd.to_pickle(paths, DER / "alignment_paths.pkl")
    print("pieces >= 2 performances:", len(pieces), "| candidate alignments:", len(Q), "| status:", Q.status.value_counts().to_dict())
    print("cost cut:", round(cut, 4), "| aligned pieces:", len(keep), "| performances incl. references:", len(paths))
