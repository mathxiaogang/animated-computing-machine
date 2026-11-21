"""Step 19 - parse every MAESTRO MIDI file into notes (onset, key release, pitch, velocity, release velocity) and pedal controllers 64/66/67.

Usage:
    python -I 19_build_note_cache.py <maestro-v3.0.0 dir> [<out .pkl>]

MAESTRO MIDI comes from Disklavier key and pedal sensors: a note-off is the key release, not the end of the sound.
Release velocity is not recorded in MAESTRO (every note-off is a note-on with velocity 0), so only release times are used."""
import sys, struct, pickle
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("maestro-v3.0.0")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "results/derived/notes_cache.pkl"
ROOT = ROOT_DIR
META = Path("/mnt/user-data/outputs/una_corda_study/data/metadata_clean.csv")

def _vlq(b, i):
    v = 0
    while True:
        c = b[i]; i += 1; v = (v << 7) | (c & 0x7F)
        if c < 0x80: return v, i

def load(path):
    b = path.read_bytes(); hl = struct.unpack(">I", b[4:8])[0]
    _, ntrk, div = struct.unpack(">HHH", b[8:14]); i, ev, tempo = 8 + hl, [], 500000
    for _ in range(ntrk):
        L = struct.unpack(">I", b[i + 4:i + 8])[0]; j, end, tick, rs = i + 8, i + 8 + L, 0, None
        while j < end:
            d, j = _vlq(b, j); tick += d; s = b[j]
            if s == 0xFF:
                typ = b[j + 1]; l, j = _vlq(b, j + 2)
                if typ == 0x51: tempo = int.from_bytes(b[j:j + 3], "big")
                j += l; continue
            if s in (0xF0, 0xF7):
                l, j = _vlq(b, j + 1); j += l; continue
            if s & 0x80: rs = s; j += 1
            st = rs & 0xF0
            if st in (0xC0, 0xD0): j += 1; continue
            ev.append((tick, st, b[j], b[j + 1])); j += 2
        i = end
    ev = np.array(ev, dtype=float); ev = ev[np.argsort(ev[:, 0], kind="stable")]
    ev[:, 0] *= tempo / 1e6 / div
    return ev

def parse(f):
    ev = load(ROOT / f); T = ev[:, 0].max()
    open_ = {}; notes = []
    for t, st, d1, d2 in ev:
        if st == 0x90 and d2 > 0:
            if d1 in open_:  # re-strike without release: close previous
                t0, v0 = open_.pop(d1); notes.append((t0, t, d1, v0, -1))
            open_[d1] = (t, d2)
        elif st == 0x80 or (st == 0x90 and d2 == 0):
            if d1 in open_:
                t0, v0 = open_.pop(d1); notes.append((t0, t, d1, v0, d2 if st == 0x80 else -1))
    for p, (t0, v0) in open_.items(): notes.append((t0, T, p, v0, -1))
    n = np.array(sorted(notes), dtype=float)
    cc = {k: ev[(ev[:, 1] == 0xB0) & (ev[:, 2] == k)][:, [0, 3]].astype(np.float32) for k in (64, 66, 67)}
    return f, dict(T=T, on=n[:, 0].astype(np.float64), off=n[:, 1].astype(np.float64), pitch=n[:, 2].astype(np.int16),
                   vel=n[:, 3].astype(np.int16), relv=n[:, 4].astype(np.int16), cc64=cc[64], cc66=cc[66], cc67=cc[67])

if __name__ == "__main__":
    meta = pd.read_csv(META)
    with Pool(8) as pool: out = dict(pool.map(parse, list(meta.midi_filename), chunksize=8))
    OUT.parent.mkdir(parents=True, exist_ok=True); pickle.dump(out, open(OUT, "wb"), protocol=4)
    n = sum(len(v["on"]) for v in out.values()); print("files", len(out), "notes", n)
    rv = np.concatenate([v["relv"] for v in out.values()]); print("release velocity values:", np.unique(rv)[:20], "share -1:", (rv == -1).mean())
