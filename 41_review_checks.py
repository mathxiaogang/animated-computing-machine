"""Step 41 - analyses requested in the internal review of the manuscript (exploratory robustness checks).

1. Surrogates that respect local tempo. The H1 surrogate permutes each change's anchor-to-re-depression interval across the whole
   performance, and the foot surrogate permutes hold durations across the whole performance. If both the interval and the
   key-holding time scale with local tempo, whole-performance permutation could widen the surrogate and exaggerate the indices.
   Two stricter surrogates: (a) tempo-matched: permute only within quintiles of local tempo (hand: log IOI from the anchor to the
   next onset; foot: log median inter-cluster IOI within 1 s around the lift); (b) local blocks: permute only within blocks of 10
   consecutive changes. The whole-performance surrogate is recomputed with the same code and number of surrogates (50).
2. Window sensitivity: avoidance / handover windows [-150, -30] / [0, 100] ms and [-300, -50] / [0, 40] ms, whole-performance
   surrogate, both limbs.
3. Shared reference ("anchor") check for the shared-plan slope. Re-depression shift and release were both measured from the
   same anchor onset, so jitter of the anchor could create a positive slope. Re-estimated on held notes struck BEFORE the anchor's
   onset cluster, with the release measured from the note's own onset (hold duration) and the note's head start (anchor minus its
   onset) added to the tempo covariates. Habit and same-score-position versions; 10th-percentile and median slopes.
4. Reliability of the foot's and the hand's handover indices across pianists (random split-half of each pianist's
   performances, Spearman-Brown), and the pianist-level hand-foot correlation corrected for attenuation.

Output: results/review_checks.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import DER, PLAN, RES, clusters, crossings, demean2, load_cache, load_meta, pool, save_json

rng = np.random.default_rng(PLAN["seed"] + 41)
M = load_meta().set_index("midi_filename")
OUT = {}
NS_ = 50
WINS = {"main": ((-0.210, -0.030), (0.0, 0.060)), "alt_150_100": ((-0.150, -0.030), (0.0, 0.100)), "alt_300_40": ((-0.300, -0.050), (0.0, 0.040))}


def counts(lag, win):
    (a0, a1), (h0, h1) = win
    return int(((lag > a0) & (lag < a1)).sum()), int(((lag >= h0) & (lag < h1)).sum())


def perm_within(groups, values):
    out = values.copy()
    for g in np.unique(groups):
        i = np.where(groups == g)[0]
        out[i] = values[rng.permutation(i)]
    return out


def li(o, s):
    return float(np.log((o + 0.5) / (np.mean(s) + 0.5)))


# ------------------------------------------------------------------ 1-2 hand: key releases around the re-depression
EV = pd.read_pickle(DER / "handover_events.pkl").sort_values(["midi_filename", "anchor"])
NT = pd.read_pickle(DER / "handover_notes.pkl")
CACHED = (RES / "review_hand_indices.csv").exists() and (RES / "review_foot_indices.csv").exists()
rows = []
for f, g in ([] if CACHED else NT.groupby("midi_filename", sort=False)):
    ev = EV[EV.midi_filename == f].reset_index(drop=True)
    if len(ev) < PLAN["min_events"]:
        continue
    pos = pd.Series(np.arange(len(ev)), index=ev.event.values)[g.event.values].values
    anc, plag, off = ev.anchor.values, (ev.p - ev.anchor).values, g.off.values
    lio = np.log(ev.ioi_next.where(ev.ioi_next > 0)).values
    tb = np.full(len(ev), -1); ok = np.isfinite(lio)
    if ok.sum() >= 10:
        tb[ok] = pd.qcut(lio[ok], 5, labels=False, duplicates="drop")
    blk = np.arange(len(ev)) // 10
    real = off - (anc + plag)[pos]
    row = {"midi_filename": f}
    for w, win in WINS.items():
        ra, rh = counts(real, win)
        designs = {"whole": None, "tempo": tb, "block": blk} if w == "main" else {"whole": None}
        for dn, grp in designs.items():
            sa, sh = [], []
            for _ in range(NS_):
                pl = rng.permutation(plag) if grp is None else perm_within(grp, plag)
                x, y = counts(off - (anc + pl)[pos], win); sa.append(x); sh.append(y)
            row[f"A_{w}_{dn}"] = li(ra, sa); row[f"H_{w}_{dn}"] = li(rh, sh)
    rows.append(row)
RH = pd.read_csv(RES / "review_hand_indices.csv") if CACHED else pd.DataFrame(rows).join(M[["pianist", "piece", "year"]], on="midi_filename")
OUT["hand"] = {c: pool(RH, c, B=200) for c in RH.columns if c.startswith(("A_", "H_"))}
print("hand:", {c: round(v["beta"], 3) for c, v in OUT["hand"].items()})

# ------------------------------------------------------------------ 1-2 foot: pedal lifts around chord onsets
C = load_cache(); rows = []
for f, d in ([] if CACHED else C.items()):
    pr, rl = crossings(d["cc64"], PLAN["pedal_threshold"])
    if len(rl) < PLAN["min_events"] or len(pr) < 2:
        continue
    j = np.searchsorted(pr, rl); ok = j < len(pr); r = rl[ok]; r = r[(pr[j[ok]] - r) < PLAN["change_max_gap_s"]]
    i = np.searchsorted(pr, r) - 1; has = i >= 0; r = r[has]; pprev = pr[i[has]]
    if len(r) < PLAN["min_events"]:
        continue
    on = d["on"]; cid = clusters(on, PLAN["cluster_gap_s"]); cnt = np.bincount(cid); first = np.r_[True, np.diff(cid) > 0]
    starts = on[first]; ch = starts[cnt >= 2]
    lo = np.searchsorted(starts, r - 1.0); hi = np.searchsorted(starts, r + 1.0)
    loc = np.array([np.median(np.diff(starts[a:b])) if b - a >= 3 else np.nan for a, b in zip(lo, hi)])
    tb = np.full(len(r), -1); okb = np.isfinite(loc) & (loc > 0)
    if okb.sum() >= 10:
        tb[okb] = pd.qcut(np.log(loc[okb]), 5, labels=False, duplicates="drop")
    blk = np.arange(len(r)) // 10; hold = r - pprev

    def wc(x, win):
        (a0, a1), (h0, h1) = win
        k = np.searchsorted(ch, x, side="right")
        db = np.where(k > 0, x - ch[np.maximum(k - 1, 0)], np.inf); da = np.where(k < len(ch), ch[np.minimum(k, len(ch) - 1)] - x, np.inf)
        return int(((da > -a1) & (da < -a0)).sum()), int(((db >= h0) & (db < h1)).sum())
    row = {"midi_filename": f}
    for w, win in WINS.items():
        ra, rh = wc(r, win)
        designs = {"whole": None, "tempo": tb, "block": blk} if w == "main" else {"whole": None}
        for dn, grp in designs.items():
            sa, sh = [], []
            for _ in range(NS_):
                hh = rng.permutation(hold) if grp is None else perm_within(grp, hold)
                x, y = wc(np.sort(pprev + hh), win); sa.append(x); sh.append(y)
            row[f"A_{w}_{dn}"] = li(ra, sa); row[f"H_{w}_{dn}"] = li(rh, sh)
    rows.append(row)
RF = pd.read_csv(RES / "review_foot_indices.csv") if CACHED else pd.DataFrame(rows).join(M[["pianist", "piece", "year"]], on="midi_filename")
OUT["foot"] = {c: pool(RF, c, B=200) for c in RF.columns if c.startswith(("A_", "H_"))}
print("foot:", {c: round(v["beta"], 3) for c, v in OUT["foot"].items()})
RH.to_csv(RES / "review_hand_indices.csv", index=False); RF.to_csv(RES / "review_foot_indices.csv", index=False)

# ------------------------------------------------------------------ 3. anchor-independent shared-plan slope
EV2 = EV[EV.ioi_next > 0].copy(); parts = []
for f, g in EV2.groupby("midi_filename", sort=False):
    on = np.sort(C[f]["on"]); cid = clusters(on, PLAN["cluster_gap_s"]); starts = on[np.r_[True, np.diff(cid) > 0]]
    k = cid[np.clip(np.searchsorted(on, g.anchor.values - 1e-6), 0, len(on) - 1)]
    at = lambda i: np.where((i >= 0) & (i < len(starts)), starts[np.clip(i, 0, len(starts) - 1)], np.nan)
    parts.append(pd.DataFrame({"midi_filename": f, "event": g.event.values, "l_prev": np.log(at(k) - at(k - 1)),
                               "l_next": np.log(at(k + 1) - at(k)), "l_span": np.log(at(k + 2) - at(k - 2))}))
TC = pd.concat(parts).replace([np.inf, -np.inf], np.nan)
EV2 = EV2.merge(TC, on=["midi_filename", "event"], how="left"); EV2["lioi"] = np.log(EV2.ioi_next)
EV2["x"] = EV2.p - EV2.anchor
EV2["pred"] = EV2.groupby("midi_filename").x.transform(lambda s: s.shift(1).rolling(5).mean()); EV2["res"] = EV2.x - EV2.pred
TEMPO = ["lioi", "l_prev", "l_next", "l_span"]
N2 = NT.merge(EV2[["midi_filename", "event", "res", *TEMPO]], on=["midi_filename", "event"]).dropna(subset=["res", *TEMPO])
N2 = N2[(N2.off - N2.anchor) < 2.0].copy()
N2["head"] = N2.anchor - N2.onset; N2["early_note"] = N2["head"] > PLAN["cluster_gap_s"]
N2["y_anchor"] = N2.off - N2.anchor; N2["y_own"] = N2.off - N2.onset
OUT["anchor_check"] = {"share_notes_before_anchor_cluster": float(N2.early_note.mean())}
qrows = []
for f, g in N2.groupby("midi_filename"):
    for label, sub, ycol, covs in (("all_notes_anchor_ref", g, "y_anchor", TEMPO), ("early_notes_anchor_ref", g[g.early_note], "y_anchor", TEMPO),
                                   ("early_notes_own_ref", g[g.early_note], "y_own", TEMPO + ["head"])):
        if sub.event.nunique() < 30:
            continue
        Z = np.column_stack([np.ones(len(sub)), *[sub[c].values for c in covs]])
        xr = sub.res.values - Z @ np.linalg.lstsq(Z, sub.res.values, rcond=None)[0]
        yr = sub[ycol].values - Z @ np.linalg.lstsq(Z, sub[ycol].values, rcond=None)[0]
        q = pd.qcut(xr, 5, labels=False, duplicates="drop")
        if len(np.unique(q)) < 4:
            continue
        t = pd.DataFrame({"q": q, "x": xr, "y": yr})
        for tau in (0.10, 0.50):
            tt = t.groupby("q").agg(yq=("y", lambda s: s.quantile(tau)), xm=("x", "mean"))
            qrows.append({"midi_filename": f, "label": label, "tau": tau, "slope": float(np.polyfit(tt.xm, tt.yq, 1)[0])})
Q = pd.DataFrame(qrows).join(M[["pianist", "piece", "year"]], on="midi_filename")
OUT["anchor_check"]["habit"] = {lab: {str(t): pool(Q[(Q.label == lab) & (Q.tau == t)], "slope", B=200) for t in (0.10, 0.50)} for lab in Q.label.unique()}
print("anchor check, habit:", {lab: {t: round(v["beta"], 3) for t, v in d.items()} for lab, d in OUT["anchor_check"]["habit"].items()})

MT = pd.read_pickle(DER / "handover_events_matched.pkl")
MT = MT[MT.groupby("set").midi_filename.transform("nunique") >= 2].copy()
MT = MT.merge(TC, on=["midi_filename", "event"], how="left"); MT["lioi"] = np.log(MT.ioi_next.where(MT.ioi_next > 0))
NS = NT.merge(MT[["midi_filename", "event", "set", *TEMPO]], on=["midi_filename", "event"]).dropna(subset=TEMPO)
NS = NS[(NS.off - NS.anchor) < 2.0].copy(); NS["x"] = NS.p - NS.anchor; NS["head"] = NS.anchor - NS.onset
NS["y_own"] = NS.off - NS.onset; NS = NS[NS["head"] > PLAN["cluster_gap_s"]].join(M[["pianist"]], on="midi_filename")
NS = NS[NS.groupby("set").midi_filename.transform("nunique") >= 2].reset_index(drop=True)
covs = TEMPO + ["head"]
D = demean2({c: NS[c].values for c in ["x", "y_own", *covs]}, NS.set.values, NS.midi_filename.values)
Z = np.column_stack([D[c] for c in covs]); x = D["x"] - Z @ np.linalg.lstsq(Z, D["x"], rcond=None)[0]; y = D["y_own"] - Z @ np.linalg.lstsq(Z, D["y_own"], rcond=None)[0]
T = pd.DataFrame({"x": x, "y": y, "pianist": NS.pianist.values, "dec": pd.qcut(x, 10, labels=False)})


def dec_slopes(t):
    return {tau: float(np.polyfit(*t.groupby("dec").agg(xm=("x", "mean"), yq=("y", lambda s: s.quantile(tau)))[["xm", "yq"]].values.T, 1)[0]) for tau in (0.10, 0.50)}


est = dec_slopes(T); pis = T.pianist.unique(); gi = {k: np.where(T.pianist.values == k)[0] for k in pis}
bs = [dec_slopes(T.iloc[np.concatenate([gi[k] for k in rng.choice(pis, len(pis))])]) for _ in range(200)]
OUT["anchor_check"]["same_position_early_notes_own_ref"] = {"n_notes": int(len(NS)), **{str(t): {"slope": est[t], "ci": np.percentile([b[t] for b in bs], [2.5, 97.5]).tolist()} for t in est}}
print("anchor check, same position:", OUT["anchor_check"]["same_position_early_notes_own_ref"])

# ------------------------------------------------------------------ 4. reliability and disattenuated hand-foot correlation
R = pd.read_csv(RES / "relay_foot_indices.csv").dropna(subset=["H_foot", "H"])
multi = R.groupby("pianist").filter(lambda g: len(g) >= 2)
rel = {"H_foot": [], "H": []}; cors = []
for _ in range(500):
    halves = []
    for p_, g in multi.groupby("pianist"):
        idx = rng.permutation(len(g)); h = len(g) // 2
        halves.append((g.iloc[idx[:h]][["H_foot", "H"]].mean(), g.iloc[idx[h:]][["H_foot", "H"]].mean()))
    a = pd.DataFrame([x[0] for x in halves]); b = pd.DataFrame([x[1] for x in halves])
    for c in rel:
        rel[c].append(np.corrcoef(a[c], b[c])[0, 1])
sb = {c: float(2 * np.mean(v) / (1 + np.mean(v))) for c, v in rel.items()}
pm = R.groupby("pianist")[["H_foot", "H"]].mean(); r_obs = float(np.corrcoef(pm.H_foot, pm.H)[0, 1])
OUT["reliability"] = {"n_pianists_split": int(multi.pianist.nunique()), "split_half_r": {c: float(np.mean(v)) for c, v in rel.items()},
                      "spearman_brown": sb, "pianist_r_pearson": r_obs, "pianist_r_disattenuated": float(r_obs / np.sqrt(sb["H_foot"] * sb["H"]))}
print("reliability:", OUT["reliability"])
save_json(OUT, RES / "review_checks.json")
