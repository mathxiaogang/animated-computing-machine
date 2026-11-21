"""Step 32 - event and note tables for the handover mechanism analyses (directions 1-4; exploratory extensions designed after
a pilot on the same data).

Per pedal change (event): anchor (latest onset of the held notes), pedal release r, re-press p (controller 64), the down-stroke
crossings t32 and t96 and the stroke duration D = t96 - t32, local IOI (anchor to the next onset cluster), the median key
release of the held notes, and the median release inside the stroke window [t32 - 0.1 s, t32 + 0.4 s].
Per held note: cut (released before p), motor need (a different key within 4 semitones struck after the note and before p),
doubled (another held note of the same pitch class), voice rank, register and velocity.
Events of aligned performances are also matched across performances at the same score position (same piece, same top held
pitch, reference times within 0.25 s).

Usage:
    python 32_handover_events.py

Writes results/derived/handover_events.pkl, handover_notes.pkl and handover_events_matched.pkl.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import DER, PLAN, clusters, handover_events, load_cache

C = load_cache()


def one(f):
    d = C[f]
    E = handover_events(d)
    if len(E) == 0 or E.event.nunique() < PLAN["min_events"]:
        return None, None
    on, pitch, vel = d["on"], d["pitch"].astype(int), d["vel"].astype(int)
    ct, cv = d["cc64"][:, 0], d["cc64"][:, 1]
    con = np.unique(on[np.r_[True, np.diff(clusters(on, PLAN["cluster_gap_s"])) > 0]])
    ev = E.groupby("event").agg(anchor=("anchor", "first"), r=("r", "first"), p=("p", "first"), rel_med=("off", "median"),
                                n_held=("off", "size"), top=("pitch", "max"), bottom=("pitch", "min")).reset_index()
    k = np.searchsorted(con, ev.anchor.values, side="right")
    ev["ioi_next"] = np.where(k < len(con), con[np.minimum(k, len(con) - 1)], np.nan) - ev.anchor
    t32, t96 = [], []
    for r, p in zip(ev.r.values, ev.p.values):
        i = np.searchsorted(ct, r, side="right"); j = np.searchsorted(ct, r + PLAN["change_max_gap_s"] + 0.3)
        st, sv = ct[i:j], cv[i:j]
        up = np.where(sv >= PLAN["pedal_threshold"])[0]
        if len(up) == 0:
            t32.append(np.nan); t96.append(np.nan); continue
        m = int(np.argmin(sv[:up[0] + 1]))  # lowest point of the change, before the re-press
        if sv[m] >= 32:  # shallow change: the pedal never went below 32, so no down-stroke from 32
            t32.append(np.nan); t96.append(np.nan); continue
        a = np.where(sv[m:] >= 32)[0]; b = np.where(sv[m:] >= 96)[0]
        t32.append(st[m + a[0]] if len(a) else np.nan); t96.append(st[m + b[0]] if len(b) else np.nan)
    ev["t32"], ev["t96"] = t32, t96
    ev["D"] = ev.t96 - ev.t32
    # notes
    pcs = E.pitch % 12
    E = E.assign(cut=(E.lag < 0).astype(int), doubled=(E.assign(pc=pcs).groupby(["event", "pc"]).pitch.transform("size") > 1).astype(int),
                 vel=vel[E.note.values])
    lo = np.searchsorted(on, E.onset.values, side="right"); hi = np.searchsorted(on, E.p.values)
    need = np.zeros(len(E), int)
    for n, (a, b, pt) in enumerate(zip(lo, hi, E.pitch.values)):
        if b > a:
            seg = pitch[a:b]
            need[n] = int(np.any((np.abs(seg - pt) <= 4) & (seg != pt)))
    E["need"] = need
    win = E.merge(ev[["event", "t32"]], on="event")
    win = win[(win.off > win.t32 - 0.1) & (win.off < win.t32 + 0.4)]
    ev = ev.merge(win.groupby("event").off.median().rename("rel_win_med"), on="event", how="left")
    ev.insert(0, "midi_filename", f); E.insert(0, "midi_filename", f)
    return ev, E[["midi_filename", "event", "note", "pitch", "rank", "vel", "onset", "off", "lag", "cut", "need", "doubled", "p", "r", "anchor"]]


if __name__ == "__main__":
    with Pool(2) as pool:
        res = pool.map(one, list(C.keys()), chunksize=8)
    EV = pd.concat([r[0] for r in res if r[0] is not None], ignore_index=True)
    NT = pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True)
    EV.to_pickle(DER / "handover_events.pkl"); NT.to_pickle(DER / "handover_notes.pkl")
    print("events", len(EV), "performances", EV.midi_filename.nunique(), "| notes", len(NT),
          "| events with full stroke 32->96:", int((EV.D > 0).sum()), "| median D ms", round(EV.D[EV.D > 0].median() * 1000, 1))
    # same score position matching (aligned performances only)
    paths = pd.read_pickle(DER / "alignment_paths.pkl"); FPS = PLAN["align"]["fps"]; TOL = PLAN["align"]["match_tol_s"]
    A = EV[EV.midi_filename.isin(paths)].copy()
    A["piece"] = A.midi_filename.map(lambda f: paths[f]["piece"])
    ref_t = np.full(len(A), np.nan)
    for f, idx in A.groupby("midi_filename").indices.items():
        m = paths[f]["match"].astype(float) + np.arange(len(paths[f]["match"])) * 1e-6
        ref_t[idx] = np.interp(A.anchor.values[idx] * FPS, m, np.arange(len(m))) / FPS
    A["ref_t"] = ref_t
    A = A.sort_values(["piece", "top", "ref_t"]).reset_index(drop=True)
    same = (A.piece.values[1:] == A.piece.values[:-1]) & (A.top.values[1:] == A.top.values[:-1]) & (np.diff(A.ref_t.values) <= TOL)
    A["set"] = np.cumsum(np.r_[True, ~same])
    A["_d"] = (A.ref_t - A.groupby("set").ref_t.transform("median")).abs()
    A = A.sort_values(["set", "_d"]).drop_duplicates(["set", "midi_filename"]).drop(columns="_d")
    A = A[A.groupby("set").midi_filename.transform("nunique") >= 2]
    A["set"] = A.piece + "|e" + A.set.astype(str)
    A.to_pickle(DER / "handover_events_matched.pkl")
    print("matched events", len(A), "| sets", A.set.nunique(), "| performances", A.midi_filename.nunique(), "| pieces", A.piece.nunique())
