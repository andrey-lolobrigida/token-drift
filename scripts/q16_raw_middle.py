"""Q16 on GPT-2: who sits in the middle of each triple, by raw geometry only (no null words)?

q16.json's role swaps say the deficiency vice is usually the most "between" word in GPT-2's
middle layers. That verdict is a percentile against frequency-matched null words, so it mixes
geometry with how the nulls happen to fall. Here: points built as in the pipeline (unit_mean),
each word in turn cast as the middle, seg = its distance to the segment between the other two
(in units of that segment's length). Smallest seg wins. Run twice: on the points as they are,
and on unit-normed points (direction only; a short average can't sit "inside" anything).
Also: which pair of the triple is closest (vice-vice = the two bad words group together, i.e.
a good/bad axis beats Aristotle's too-little/too-much one).

`uv run python scripts/q16_raw_middle.py runs/gpt2_probe`, needs extract/occ.npy, a few min.
"""
import json
import sys
from pathlib import Path

import numpy as np

from token_drift import betweenness as bt, cli

rd = Path(sys.argv[1])
q = json.loads((rd / "metrics" / "q16.json").read_text())
shared = {s["id"] for s in json.loads((rd / "probe_corpus" / "shared_last_piece.json").read_text())}
data = cli._load_probe(rd)
names = q["layer_names"]
slices = [(s, g) for s in ("classical", "everyday") for g in cli.GROUPS]
trip = {sg: [t for t in q["triples"] if t["set"] == sg[0] and t["id"] not in shared
             and "missing" not in t["groups"][sg[1]]] for sg in slices}
ROLES = ("defic", "virtue", "excess")


def segs(P, words, gi):
    """seg of each role as the middle, vs the other two."""
    x = [P[data.w_index[w], gi] for w in words]
    out = []
    for m in range(3):
        a, b = [x[i] for i in range(3) if i != m]
        out.append(float(bt.segment_stats(a, x[m], b)[2]))
    return out


res = {sg: {"raw": [], "unit": []} for sg in slices}
med_seg = {sg: [] for sg in slices}
closest = {sg: [] for sg in slices}
PAIRS = ((0, 1), (1, 2), (0, 2))  # defic-virtue, virtue-excess, vice-vice
for f in range(len(names)):
    P, _, _ = bt.unit_mean(data.occ[f], data.widx, data.gidx, len(data.vocab), len(cli.GROUPS))
    U = P / np.linalg.norm(P, axis=-1, keepdims=True)
    for s, g in slices:
        gi = cli.GROUPS.index(g)
        for kind, M in (("raw", P), ("unit", U)):
            c = [0, 0, 0]
            all_s = []
            for t in trip[(s, g)]:
                sv = segs(M, t["words"], gi)
                c[int(np.argmin(sv))] += 1
                all_s.append(sv)
            res[(s, g)][kind].append(c)
            if kind == "raw":
                med_seg[(s, g)].append(np.median(all_s, axis=0) if all_s else [np.nan] * 3)
                cc = [0, 0, 0]
                for t in trip[(s, g)]:
                    x = [P[data.w_index[w], gi] for w in t["words"]]
                    cc[int(np.argmin([np.linalg.norm(x[i] - x[j]) for i, j in PAIRS]))] += 1
                closest[(s, g)].append(cc)
    print(f"frame {names[f]} done", file=sys.stderr)

for s, g in slices:
    if not trip[(s, g)]:
        continue
    print(f"\n{s}/{g}, n={len(trip[(s, g)])}: who is in the middle (defic/virtue/excess), median seg per role (raw), "
          f"closest pair (defic-virtue/virtue-excess/vice-vice, raw)")
    for f, nm in enumerate(names):
        r, u, m = res[(s, g)]["raw"][f], res[(s, g)]["unit"][f], med_seg[(s, g)][f]
        print(f"  {nm:14s} raw {r[0]:2d}/{r[1]:2d}/{r[2]:2d}   unit {u[0]:2d}/{u[1]:2d}/{u[2]:2d}   "
              f"seg {m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}   closest {closest[(s, g)][f][0]:2d}/{closest[(s, g)][f][1]:2d}/{closest[(s, g)][f][2]:2d}")
