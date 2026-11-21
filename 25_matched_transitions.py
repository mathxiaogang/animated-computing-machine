"""Step 25 - put every melodic transition (and every chord) of the aligned performances on the reference timeline of its
piece, and group transitions that are the same notes at the same score position ("transition sets").

Usage:
    python 25_matched_transitions.py [variant=main]

A transition set = same piece, same pitch pair, reference times chained within 0.25 s; at most one transition per performance
(the one closest to the set's median time); at least two performances. Chords are matched the same way on (top, bottom) pitch.
Writes results/derived/matched_transitions[_variant].pkl and matched_chords.pkl.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_common import DER, PLAN, load_meta

OPTS = dict(a.split("=", 1) for a in sys.argv[1:])
VAR = OPTS.get("variant", "main"); SFX = "" if VAR == "main" else f"_{VAR}"
FPS = PLAN["align"]["fps"]; TOL = PLAN["align"]["match_tol_s"]
paths = pd.read_pickle(DER / "alignment_paths.pkl")


def to_ref_time(df, tcol="t"):
    out = np.full(len(df), np.nan)
    for f, idx in df.groupby("midi_filename").indices.items():
        m = paths[f]["match"].astype(float)
        m = m + np.arange(len(m)) * 1e-6  # strictly increasing for interpolation across plateaus
        out[idx] = np.interp(df[tcol].values[idx] * FPS, m, np.arange(len(m))) / FPS
    return out


def make_sets(df, keys):
    df = df.sort_values(keys + ["ref_t"]).reset_index(drop=True)
    same = np.ones(len(df), bool)
    for k in keys:
        v = df[k].values; same[1:] &= v[1:] == v[:-1]
    same[0] = False
    new = ~same | np.r_[True, np.diff(df.ref_t.values) > TOL]
    df["set"] = np.cumsum(new)
    med = df.groupby("set").ref_t.transform("median")
    df["_d"] = (df.ref_t - med).abs()
    df = df.sort_values(["set", "_d"]).drop_duplicates(["set", "midi_filename"]).drop(columns="_d")
    n = df.groupby("set").midi_filename.transform("nunique")
    return df[n >= 2].reset_index(drop=True)


meta = load_meta().set_index("midi_filename")
T = pd.read_pickle(DER / f"transitions_{VAR}.pkl")
T = T[T.midi_filename.isin(paths)].copy()
T["piece"] = T.midi_filename.map(lambda f: paths[f]["piece"])
T["ref_t"] = to_ref_time(T)
D = make_sets(T, ["piece", "pa", "pb"])
D["set"] = D.piece + "|" + D.set.astype(str)
D.to_pickle(DER / f"matched_transitions{SFX}.pkl")
mixed = D[D.ped.isin([0, 1])].groupby("set").ped.nunique() == 2
print(f"{VAR}: aligned performances {T.midi_filename.nunique()} | transitions {len(T)} | in sets {len(D)} | sets {D.set.nunique()}"
      f" | sets with both pedal states {int(mixed.sum())} | pieces {D.piece.nunique()}")

if VAR == "main":
    CH = pd.read_pickle(DER / "chords.pkl")
    CH = CH[CH.midi_filename.isin(paths)].copy()
    CH["piece"] = CH.midi_filename.map(lambda f: paths[f]["piece"])
    CH["ref_t"] = to_ref_time(CH)
    DC = make_sets(CH, ["piece", "top", "bot"])
    DC["set"] = DC.piece + "|c" + DC.set.astype(str)
    DC.to_pickle(DER / "matched_chords.pkl")
    mixed_c = DC[DC.ped.isin([0, 1])].groupby("set").ped.nunique() == 2
    print(f"chords in sets {len(DC)} | sets {DC.set.nunique()} | with both pedal states {int(mixed_c.sum())}")
