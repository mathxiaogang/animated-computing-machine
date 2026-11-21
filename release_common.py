"""Shared definitions for the key-release study ("Control follows audibility").

All analysis parameters live in PLAN, which 20_synthetic_validation.py freezes (with a SHA-256 hash) before any
confirmatory analysis is run. Other scripts import from here so every step uses the same definitions.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DER = RES / "derived"
FIG = ROOT / "figures"
META = Path("/mnt/user-data/outputs/una_corda_study/data/metadata_clean.csv")
FEAT = Path("/mnt/user-data/outputs/una_corda_study/data/features.csv")

PLAN = {
    "cluster_gap_s": 0.03,            # onsets closer than this form one simultaneity cluster
    "melody_min_pitch": 60,           # previous melody note >= C4
    "ioi_range_s": [0.12, 1.2],       # inter-onset interval of a melodic transition
    "interval_range": [1, 7],         # semitones between successive melody notes
    "kot_abs_max_s": 0.35,            # |key overlap time| kept
    "pedal_window_s": 0.4,            # pedal state judged in [next onset - w, next onset + w]
    "pedal_threshold": 64,            # controller value counted as down
    # 4 x 3 x 2 x 2 = 48 cells; grid and minimum chosen from sample sizes only (304 performances keep >= 1 cell), before any effect was computed
    "strata": {"ioi": [0.12, 0.2, 0.35, 0.6, 1.2], "vel": [0, 55, 75, 128], "pitch": [59, 76, 128], "interval": [0, 2, 7]},
    "min_per_cell": 8,                # transitions per pedal condition in a stratum
    "change_max_gap_s": 0.6,          # release -> re-press within this = a pedal change
    "held_lookback_s": 0.4,           # held notes struck within this before the release
    "n_surrogates": 100,
    "avoid_window_ms": [-210, -30],
    "handover_window_ms": [0, 60],
    "min_events": 20,                 # pedal changes with held notes per performance (H1)
    "peak_grid_ms": [-100, 300],
    "kde_bw_ms": 10,
    "feedforward_bound_ms": 100,
    "sesoi_location_ms": 10,
    "sesoi_log_scale": float(np.log(1.1)),
    "chord_min_notes": 3, "chord_span_s": 0.06,
    "align": {"fps": 10, "band": 0.12, "ratio": [0.75, 1.33], "cost_drop_quantile": 0.9, "match_tol_s": 0.25},
    "bootstrap_B": 1000, "seed": 20261118,
    "confirmatory": ["H1a", "H1b", "H1c", "H2a", "H2b", "H2c", "H3a", "H3b"],
}


def load_cache():
    return pickle.load(open(DER / "notes_cache.pkl", "rb"))


def load_meta():
    m = pd.read_csv(META).merge(pd.read_csv(FEAT)[["midi_filename", "sus_frac", "soft_frac"]], on="midi_filename")
    m["piece"] = (m.composer + " | " + m.canonical_title.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
                  .str.replace(r"\s+", " ", regex=True).str.strip())
    return m


# ------------------------------------------------------------------ pedal helpers
def ped_value(cc, t):
    """controller value at times t (step function)."""
    if len(cc) == 0:
        return np.zeros(len(t))
    k = np.searchsorted(cc[:, 0], t, side="right") - 1
    return np.where(k >= 0, cc[np.clip(k, 0, None), 1], 0.0)


def window_minmax(cc, centre, w, n=17):
    """min and max controller value over [centre - w, centre + w] (sampled densely, plus every event inside)."""
    ts = centre[:, None] + np.linspace(-w, w, n)[None, :]
    v = ped_value(cc, ts.ravel()).reshape(ts.shape)
    lo, hi = v.min(1), v.max(1)
    if len(cc):  # events falling between sample points
        a = np.searchsorted(cc[:, 0], centre - w); b = np.searchsorted(cc[:, 0], centre + w)
        for i in np.where(b > a)[0]:
            seg = cc[a[i]:b[i], 1]; lo[i] = min(lo[i], seg.min()); hi[i] = max(hi[i], seg.max())
    return lo, hi


def pedal_state(cc, centre, w, thr):
    """1 = down throughout, 0 = up throughout, -1 = changes within the window."""
    lo, hi = window_minmax(cc, centre, w)
    return np.where(lo >= thr, 1, np.where(hi < thr, 0, -1)), lo, hi


def crossings(cc, thr):
    if len(cc) == 0:
        return np.array([]), np.array([])
    d = cc[:, 1] >= thr
    ch = np.diff(d.astype(int))
    return cc[1:, 0][ch == 1], cc[1:, 0][ch == -1]


def clusters(on, gap):
    return np.concatenate([[0], np.cumsum(np.diff(on) >= gap)])


# ------------------------------------------------------------------ melodic transitions
def transitions(d, voice="top", plan=PLAN, thr=None, w=None, interval_max=None, ioi_range=None, kot_max=None):
    """Successive notes of the top (or bottom) voice with key overlap time and pedal state."""
    thr = plan["pedal_threshold"] if thr is None else thr
    w = plan["pedal_window_s"] if w is None else w
    imax = plan["interval_range"][1] if interval_max is None else interval_max
    ir = plan["ioi_range_s"] if ioi_range is None else ioi_range
    kmax = plan["kot_abs_max_s"] if kot_max is None else kot_max
    on, off, p, v = d["on"], d["off"], d["pitch"].astype(int), d["vel"].astype(int)
    cid = clusters(on, plan["cluster_gap_s"])
    if voice == "bottom":
        order = np.lexsort((p, cid))
    else:
        order = np.lexsort((-p, cid))
    first = np.r_[True, cid[order][1:] != cid[order][:-1]]
    sk = order[first]
    if voice == "separated":  # top note at least 5 semitones above every other note of its cluster
        second = {}
        for c, i in zip(cid[order], order):
            second.setdefault(c, []).append(p[i])
        keep = np.array([len(second[cid[i]]) < 2 or second[cid[i]][0] - second[cid[i]][1] >= 5 for i in sk])
        sk = sk[keep]
    sk = sk[np.argsort(on[sk])]
    a, b = sk[:-1], sk[1:]
    ioi = on[b] - on[a]; kot = off[a] - on[b]; step = np.abs(p[b] - p[a])
    pitch_ok = (p[a] >= plan["melody_min_pitch"]) if voice != "bottom" else (p[a] < 60)
    ok = (ioi > ir[0]) & (ioi < ir[1]) & (step >= plan["interval_range"][0]) & (step <= imax) & pitch_ok & (np.abs(kot) < kmax)
    a, b = a[ok], b[ok]
    st, lo, hi = pedal_state(d["cc64"], on[b], w, thr)
    sst, _, _ = pedal_state(d["cc67"], on[b], w, thr)
    return pd.DataFrame({"t": on[b], "ioi": on[b] - on[a], "kot": off[a] - on[b], "pa": p[a], "pb": p[b],
                         "interval": np.abs(p[b] - p[a]), "vel": v[a], "ped": st, "ped_min": lo, "ped_max": hi, "soft": sst})


def add_strata(T, plan=PLAN):
    s = plan["strata"]
    T = T.copy()
    T["s_ioi"] = pd.cut(T.ioi, s["ioi"], labels=False)
    T["s_vel"] = pd.cut(T.vel, s["vel"], labels=False)
    T["s_pitch"] = pd.cut(T.pa, s["pitch"], labels=False)
    T["s_int"] = pd.cut(T.interval, s["interval"], labels=False)
    T["stratum"] = T.s_ioi.astype(str) + T.s_vel.astype(str) + T.s_pitch.astype(str) + T.s_int.astype(str)
    return T


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def matched_effect(T, cond_col="ped", y="kot", min_n=None, extra_strata=()):
    """Within one performance: weighted mean over strata of median and log-MAD differences (cond 1 minus cond 0).
    Location uses raw medians. Scale uses residuals from a per-condition OLS of y on log IOI and velocity, so that
    covariate differences left inside a stratum do not masquerade as variability (synthetic validation, step 20)."""
    min_n = PLAN["min_per_cell"] if min_n is None else min_n
    keys = ["stratum", *extra_strata]
    T = T[T[cond_col].isin([0, 1])].copy()
    T["_r"] = np.nan
    for c in (0, 1):
        m = (T[cond_col] == c).values
        if m.sum() >= 5:
            X = np.column_stack([np.ones(m.sum()), np.log(T.ioi.values[m]), T.vel.values[m].astype(float)])
            b, *_ = np.linalg.lstsq(X, T[y].values[m], rcond=None)
            T.loc[m, "_r"] = T[y].values[m] - X @ b
    # the MAD of a small sample is biased low, so the larger group is subsampled to the smaller group's size
    # (mean log MAD over 20 subsamples) and both MADs carry the same bias (synthetic validation, step 20)
    rng = np.random.default_rng(0)
    loc, sc, w = [], [], []
    for _, g in T.groupby(keys):
        a, b = g[g[cond_col] == 1][y].values, g[g[cond_col] == 0][y].values
        if len(a) >= min_n and len(b) >= min_n:
            ra, rb = g[g[cond_col] == 1]._r.values, g[g[cond_col] == 0]._r.values
            n = min(len(ra), len(rb))

            def lmad(r):
                if len(r) == n:
                    m = mad(r)
                    return np.log(m) if m > 0 else np.nan
                vals = [mad(rng.choice(r, n, replace=False)) for _ in range(20)]
                vals = [v for v in vals if v > 0]
                return np.mean(np.log(vals)) if vals else np.nan
            la, lb = lmad(ra), lmad(rb)
            if np.isfinite(la) and np.isfinite(lb):
                loc.append(np.median(a) - np.median(b)); sc.append(la - lb); w.append(n)
    if not w:
        return None
    w = np.array(w, float)
    return {"loc": float(np.average(loc, weights=w)), "logscale": float(np.average(sc, weights=w)), "n_strata": len(w), "weight": float(w.sum())}


# ------------------------------------------------------------------ hand-foot handover at pedal changes (H1)
def handover_events(d, thr=None, gap_max=None, lookback=None, plan=PLAN):
    """One row per note that was struck shortly before a pedal release and still held at it.
    lag = key release minus the foot's re-press (s); anchor = latest onset of the event's held notes."""
    thr = plan["pedal_threshold"] if thr is None else thr
    gap_max = plan["change_max_gap_s"] if gap_max is None else gap_max
    lookback = plan["held_lookback_s"] if lookback is None else lookback
    pr, rl = crossings(d["cc64"], thr)
    if len(rl) == 0 or len(pr) == 0:
        return pd.DataFrame()
    j = np.searchsorted(pr, rl); ok = j < len(pr)
    r = rl[ok]; p = pr[j[ok]]; m = (p - r) < gap_max; r, p = r[m], p[m]
    on, off, pitch = d["on"], d["off"], d["pitch"].astype(int)
    lo = np.searchsorted(on, r - lookback); hi = np.searchsorted(on, r)
    rows = []
    for k in range(len(r)):
        idx = np.arange(lo[k], hi[k]); idx = idx[off[idx] > r[k]]
        if len(idx) == 0:
            continue
        anc = on[idx].max(); ps = pitch[idx]
        rank = np.where(ps == ps.max(), "top", np.where(ps == ps.min(), "bottom", "inner")) if len(idx) > 1 else np.array(["single"])
        for i, rk in zip(idx, rank):
            rows.append((k, i, off[i] - p[k], on[i], off[i], anc, p[k] - anc, p[k] - r[k], r[k], p[k], pitch[i], rk))
    return pd.DataFrame(rows, columns=["event", "note", "lag", "onset", "off", "anchor", "plag", "gap", "r", "p", "pitch", "rank"])


def handover_indices(E, rng, n_sur=None, plan=PLAN, return_sur=False):
    """Avoidance (A) and handover (H) indices: log ratio of real to surrogate release counts in the two windows.
    Surrogates keep every note's onset and release and every event's onset-to-press lag distribution,
    but pair each event with the onset-to-press lag of another, randomly drawn event of the same performance."""
    n_sur = plan["n_surrogates"] if n_sur is None else n_sur
    a0, a1 = np.array(plan["avoid_window_ms"]) / 1000; h0, h1 = np.array(plan["handover_window_ms"]) / 1000
    ev = E.drop_duplicates("event")[["event", "anchor", "plag"]].reset_index(drop=True)
    pos = pd.Series(np.arange(len(ev)), index=ev.event.values)[E.event.values].values
    off = E.off.values; anc = ev.anchor.values; plag = ev.plag.values
    real = E.lag.values
    ra = ((real > a0) & (real < a1)).sum(); rh = ((real >= h0) & (real < h1)).sum()
    sa, sh, keep = [], [], []
    for _ in range(n_sur):
        perm = rng.permutation(len(ev))
        lag = off - (anc + plag[perm])[pos]
        sa.append(((lag > a0) & (lag < a1)).sum()); sh.append(((lag >= h0) & (lag < h1)).sum())
        if return_sur and len(keep) < 5:
            keep.append(lag)
    out = {"n_events": len(ev), "n_notes": len(E), "real_avoid": int(ra), "sur_avoid": float(np.mean(sa)), "real_hand": int(rh),
           "sur_hand": float(np.mean(sh)), "A": float(np.log((ra + 0.5) / (np.mean(sa) + 0.5))),
           "H": float(np.log((rh + 0.5) / (np.mean(sh) + 0.5))), "cut_share": float((real < 0).mean())}
    if return_sur:
        out["sur_lags"] = np.concatenate(keep)
    return out


def kde_curve(lags_s, grid_ms=None, bw_ms=None, plan=PLAN, weights=None):
    """Gaussian KDE of lags on a 1-ms grid (binned, then smoothed)."""
    g0, g1 = plan["peak_grid_ms"] if grid_ms is None else grid_ms
    bw = plan["kde_bw_ms"] if bw_ms is None else bw_ms
    lo, hi = g0 - 5 * bw, g1 + 5 * bw
    x = np.asarray(lags_s) * 1000
    h, e = np.histogram(x, bins=np.arange(lo, hi + 1, 1.0), weights=weights)
    k = np.exp(-0.5 * (np.arange(-5 * bw, 5 * bw + 1) / bw) ** 2); k /= k.sum()
    s = np.convolve(h, k, "same")
    centres = (e[:-1] + e[1:]) / 2
    m = (centres >= g0) & (centres <= g1)
    return centres[m], s[m]


def kde_peak(lags_s, **kw):
    c, s = kde_curve(lags_s, **kw)
    return float(c[np.argmax(s)])


# ------------------------------------------------------------------ crossed random-intercept LMM (REML) with GLS fixed effects
def _Z(groups):
    Zs, sizes = [], []
    for g in groups:
        codes = pd.factorize(g)[0]
        Z = np.zeros((len(codes), codes.max() + 1)); Z[np.arange(len(codes)), codes] = 1
        Zs.append(Z); sizes.append(Z.shape[1])
    return np.hstack(Zs), sizes


def lmm(y, X, groups):
    """REML fit of y = X b + sum_k Z_k u_k + e. Returns dict with beta, se, z, p, variance components."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, p = X.shape
    Z, sizes = _Z(groups); q = Z.shape[1]
    ZtZ, ZtX, Zty = Z.T @ Z, Z.T @ X, Z.T @ y
    XtX, Xty, yty = X.T @ X, X.T @ y, y @ y

    def parts(t):
        s = np.exp(t); se = s[0]
        dvec = np.concatenate([np.full(k, s[i + 1]) for i, k in enumerate(sizes)])
        M = se * np.diag(1 / dvec) + ZtZ
        c = linalg.cho_factor(M, lower=True)
        MZX, MZy = linalg.cho_solve(c, ZtX), linalg.cho_solve(c, Zty)
        XVX = (XtX - ZtX.T @ MZX) / se
        XVy = (Xty - ZtX.T @ MZy) / se
        yVy = (yty - Zty @ MZy) / se
        logdetV = n * np.log(se) + np.log(dvec).sum() + 2 * np.log(np.diag(c[0])).sum() - q * np.log(se)
        return XVX, XVy, yVy, logdetV

    def nll(t):
        try:
            XVX, XVy, yVy, ld = parts(t)
        except linalg.LinAlgError:
            return 1e12
        b = np.linalg.solve(XVX, XVy)
        return 0.5 * (ld + np.linalg.slogdet(XVX)[1] + yVy - b @ XVy)

    t0 = np.log(np.full(len(groups) + 1, max(np.var(y), 1e-8) / (len(groups) + 1)))
    o = optimize.minimize(nll, t0, method="L-BFGS-B", bounds=[(-30, 10)] * len(t0))
    XVX, XVy, _, _ = parts(o.x)
    cov = np.linalg.inv(XVX); b = cov @ XVy; se = np.sqrt(np.diag(cov))
    z = b / se
    return {"beta": b, "se": se, "z": z, "p": 2 * stats.norm.sf(np.abs(z)), "vc": np.exp(o.x), "theta": o.x}


def gls_beta(y, X, groups, vc):
    """GLS fixed effects for given variance components (used inside cluster bootstraps)."""
    Z, sizes = _Z(groups)
    se = vc[0]; dvec = np.concatenate([np.full(k, vc[i + 1]) for i, k in enumerate(sizes)])
    M = se * np.diag(1 / dvec) + Z.T @ Z
    c = linalg.cho_factor(M, lower=True)
    ZtX, Zty = Z.T @ X, Z.T @ y
    XVX = (X.T @ X - ZtX.T @ linalg.cho_solve(c, ZtX)) / se
    XVy = (X.T @ y - ZtX.T @ linalg.cho_solve(c, Zty)) / se
    return np.linalg.solve(XVX, XVy)


def pool(df, col, groups=("pianist", "piece", "year"), B=None, seed=None, extra_X=()):
    """Second stage: crossed random intercepts for pianist, piece and edition; Wald test + pianist-cluster bootstrap CI."""
    B = PLAN["bootstrap_B"] if B is None else B
    rng = np.random.default_rng(PLAN["seed"] if seed is None else seed)
    d = df.dropna(subset=[col, *extra_X]).reset_index(drop=True)
    X = np.column_stack([np.ones(len(d))] + [d[c].values - d[c].mean() for c in extra_X])
    g = [d[c].astype(str).values for c in groups]
    f = lmm(d[col].values, X, g)
    pianists = d.pianist.unique(); idx = {k: np.where(d.pianist.values == k)[0] for k in pianists}
    bs = []
    if B == 0:
        bs = [np.nan]
    for _ in range(B):
        draw = rng.choice(pianists, len(pianists))
        s = np.concatenate([idx[k] for k in draw])
        # duplicated pianists are relabelled as distinct clusters
        lab = np.concatenate([np.full(len(idx[k]), j) for j, k in enumerate(draw)])
        gs = [lab.astype(str)] + [gg[s] for gg in g[1:]]
        bs.append(gls_beta(d[col].values[s], X[s], gs, f["vc"])[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": len(d), "pianists": d.pianist.nunique(), "beta": float(f["beta"][0]), "se": float(f["se"][0]), "z": float(f["z"][0]),
            "p": float(f["p"][0]), "ci_lo": float(lo), "ci_hi": float(hi), "vc": f["vc"].tolist(),
            "betas": f["beta"].tolist(), "ses": f["se"].tolist()}


def tost(est, se, bound):
    """Two one-sided tests for equivalence within [-bound, bound]; returns p (max of the two)."""
    p1 = stats.norm.sf((est + bound) / se)
    p2 = stats.norm.cdf((est - bound) / se)
    return float(max(p1, p2))


# ------------------------------------------------------------------ two-way fixed effects with cluster-robust SE
def demean2(cols, g1, g2, it=500, tol=1e-10):
    """Alternating projections: remove g1 and g2 fixed effects from each column."""
    out = {}
    c1 = pd.factorize(g1)[0]; c2 = pd.factorize(g2)[0]
    n1, n2 = np.bincount(c1), np.bincount(c2)
    for k, x in cols.items():
        z = np.asarray(x, float).copy()
        for _ in range(it):
            z0 = z.copy()
            z -= (np.bincount(c1, z) / n1)[c1]
            z -= (np.bincount(c2, z) / n2)[c2]
            if np.max(np.abs(z - z0)) < tol:
                break
        out[k] = z
    return out


def fe_regress(y, Xcols, g1, g2, cluster, cluster2=None):
    """OLS after two-way demeaning; CR1 cluster-robust SE (one-way, or two-way if cluster2 given)."""
    names = list(Xcols)
    D = demean2({"y": y, **Xcols}, g1, g2)
    X = np.column_stack([D[k] for k in names]); yy = D["y"]
    XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ X.T @ yy; e = yy - X @ b

    def meat(cl):
        codes = pd.factorize(cl)[0]; G = codes.max() + 1
        S = np.zeros((G, X.shape[1]))
        np.add.at(S, codes, X * e[:, None])
        return S.T @ S * G / (G - 1), G

    M1, G1 = meat(cluster)
    V = XtX_inv @ M1 @ XtX_inv
    if cluster2 is not None:
        M2, _ = meat(cluster2)
        M12, _ = meat(pd.Series(cluster).astype(str).values + "|" + pd.Series(cluster2).astype(str).values)
        V = XtX_inv @ (M1 + M2 - M12) @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return {k: {"b": float(b[i]), "se": float(se[i]), "z": float(b[i] / se[i]), "p": float(2 * stats.norm.sf(abs(b[i] / se[i])))} for i, k in enumerate(names)} | {"_resid": e, "_G": G1}


def fe_location_scale(y, Xcols, g1, g2, cluster, cluster2=None, scale_cols=None):
    """Location: two-way FE regression of y on Xcols. Scale: two-way FE regression of log|e| on the same columns,
    with residuals inflated by 1/sqrt(1 - h) (approximate leverage of the two fixed effects and the regressors)
    so that observations in small groups are not mistaken for less variable ones (synthetic validation, step 20)."""
    loc = fe_regress(y, Xcols, g1, g2, cluster, cluster2)
    e = loc["_resid"]
    c1 = pd.factorize(g1)[0]; c2 = pd.factorize(g2)[0]
    h = 1 / np.bincount(c1)[c1] + 1 / np.bincount(c2)[c2]
    D = demean2(Xcols, g1, g2)
    for k in Xcols:
        h = h + D[k] ** 2 / np.sum(D[k] ** 2)
    h = np.clip(h, 0, 0.95)
    z = np.log(np.abs(e) / np.sqrt(1 - h) + 1e-3)
    sc = fe_regress(z, Xcols if scale_cols is None else scale_cols, g1, g2, cluster, cluster2)
    return loc, sc


def cluster_bootstrap_fe(df, ycol, xcols, g1, g2, cluster="piece", B=200, seed=0, focus="ped", calib=None):
    """Percentile CIs for the location and (calibrated) scale coefficient of `focus` from a bootstrap over clusters
    (pieces by default: resampled pieces get fresh labels so duplicated sets and performances stay distinct)."""
    rng = np.random.default_rng(seed)
    ids = df[cluster].unique(); idx = {k: np.where(df[cluster].values == k)[0] for k in ids}
    a, b = (0.0, 1.0) if calib is None else (calib["intercept"], calib["slope"])
    L, S = [], []
    for _ in range(B):
        draw = rng.choice(ids, len(ids))
        s = np.concatenate([idx[k] for k in draw])
        tag = np.concatenate([np.full(len(idx[k]), j) for j, k in enumerate(draw)]).astype(str)
        d = df.iloc[s]
        X = {c: d[c].values.astype(float) for c in xcols}
        loc, sc = fe_location_scale(d[ycol].values, X, tag + "|" + d[g1].astype(str).values, tag + "|" + d[g2].astype(str).values,
                                    d.pianist.values)
        L.append(loc[focus]["b"]); S.append((sc[focus]["b"] - a) / b)
    return {"loc_ci": np.percentile(L, [2.5, 97.5]).tolist(), "scale_ci": np.percentile(S, [2.5, 97.5]).tolist(), "B": B}


def calibrate_scale(Xcols, g1, g2, cluster, focus="ped", sigma=0.04, gammas=(0.0, 0.3), reps=10, seed=0):
    """Simulation-based calibration of the two-way FE scale estimator on the OBSERVED design (groups and regressors):
    data are simulated with known log-scale effects of `focus`, and the estimator's intercept (bias at 0) and slope
    (attenuation) are returned. Calibrated estimate = (raw - intercept) / slope (synthetic validation, step 20)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(Xcols[focus], float)
    out = []
    for gm in gammas:
        for _ in range(reps):
            e = sigma * np.exp(gm * x) * rng.standard_normal(len(x))
            _, sc = fe_location_scale(e, Xcols, g1, g2, cluster)
            out.append((gm, sc[focus]["b"]))
    o = np.array(out)
    slope, icpt = np.polyfit(o[:, 0], o[:, 1], 1)
    return {"intercept": float(icpt), "slope": float(slope), "sims": o.tolist()}


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=conv))
