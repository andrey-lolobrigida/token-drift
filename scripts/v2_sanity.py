"""v2 sanity checks (spec 'Sanity checks on the real run'). Reads runs/ directly, like the other scripts.

1. step143000 should reproduce runs/pythia70m (same weights as main), within ~1e-3.
2. step0 vs runs/random_init: similar, not identical (HF's init may not be EleutherAI's).
3. Tokens the Pile never has should show up as embed rows that never got a gradient: same direction
   as at init, shrunk by weight decay alone (see the comment at check 3 for why not "lowest drift").
"""
import json
from pathlib import Path

import numpy as np

KEYS = ("knn_consecutive", "knn_purity", "knn_purity_shuffled", "anisotropy", "top_pc_share", "cka")
CK = Path("runs/pythia70m_ckpt")


def load(p):
    return json.loads(Path(p).read_text())


def gaps(a: dict, b: dict) -> dict:
    assert a["layer_names"] == b["layer_names"], (a["layer_names"], b["layer_names"])
    assert a["subsample_idx"] == b["subsample_idx"], "different metrics subsample: not comparable"
    return {k: round(float(np.nanmax(np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)))), 4)
            for k in KEYS if a.get(k) is not None and b.get(k) is not None}


print("1. step143000 vs runs/pythia70m, max |diff| per curve (expect < ~1e-3):")
print("  ", gaps(load(CK / "step0143000/metrics/metrics.json"), load("runs/pythia70m/metrics/metrics.json")))
print("2. step0 vs runs/random_init, max |diff| per curve (expect similar, not identical):")
print("  ", gaps(load(CK / "step0000000/metrics/metrics.json"), load("runs/random_init/metrics/metrics.json")))
tl = load(CK / "timeline/timeline.json")
# Check 3 was first "lowest drift = never seen", but a row that never gets a gradient doesn't stay
# put: AdamW's weight decay shrinks it toward 0, so its drift -> 1, while trained rows rotate to
# ~orthogonal and land near 1.5. Never-trained = same direction as at init, length collapsed.
fin = CK / f"step{tl['steps'][-1]:07d}"
toks = load(fin / "frames/tokens.json")
fb = np.load(fin / "frames/freq_bins.npy")
w0 = np.load(CK / "step0000000/weights/embed.npy").astype(np.float32)
wf = np.load(fin / "weights/embed.npy").astype(np.float32)
n0, nf = np.linalg.norm(w0, axis=1), np.linalg.norm(wf, axis=1)
cos = (w0 * wf).sum(1) / (n0 * nf + 1e-12)
dead = np.flatnonzero((cos > 0.99) & (nf < 0.01 * n0))
print(f"3. embed rows that only decayed by {tl['final']} (cos(init, final) > 0.99, norm < 1% of init): "
      f"{len(dead)} of {len(n0)}")
print(f"   their median shrink factor {np.median(nf[dead] / n0[dead]):.2e}; "
      f"everyone else: median cos {np.median(np.delete(cos, dead)):.3f}, "
      f"median norm ratio {np.median(np.delete(nf / n0, dead)):.2f}")
print("   freq bins of the decayed rows:", dict(zip(*[a.tolist() for a in np.unique(fb[dead], return_counts=True)])))
for i in dead[:25]:
    print(f"   id {i:>5}  bin {fb[i]}  cos {cos[i]:.3f}  shrink {nf[i] / n0[i]:.1e}  {toks[i]!r}")
print("half-way steps:", json.dumps(tl["half_way"], indent=1))
