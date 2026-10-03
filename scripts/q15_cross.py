"""FINDINGS 11.4 / Q15: how far apart are v0 (token alone) and v1 (corpus-averaged)?

One-off check, not part of the pipeline. Per-token cross overlap (Jaccard of the 10
neighbours in v0 vs v1, same frame, same 10k tokens), under both normalization orders
(section 8 / 11.3: center-first scrambles L6pre), split by corpus-frequency bin and by
category. Trained vs random-init. `uv run python scripts/q15_cross.py`, a few min on CPU.
"""
import json
import numpy as np
from token_drift.cli import corpus_eligibility
from token_drift.labels import CATEGORIES
from token_drift.metrics import _per_row_change, knn_indices
from token_drift.normalize import normalize_layer

R = "runs/"
PAIRS = {"trained": ("pythia70m", "pythia70m_corpus"), "random": ("random_init", "random_init_corpus")}
K = 10


def frames(run, fit, rnf):
    a = np.load(R + run + "/extract/acts.npy", mmap_mode="r")
    return [normalize_layer(a[i], center=True, unit_norm=True, drop_top_pcs=0,
                            row_norm_first=rnf, fit_rows=fit) for i in range(a.shape[0])]


for name, (v0, v1) in PAIRS.items():
    m1 = json.load(open(R + v1 + "/metrics/metrics.json"))
    idx = np.array(m1["subsample_idx"])
    counts = np.load(R + v1 + "/extract/counts.npy")
    _, cb = corpus_eligibility(counts, 1)
    fb, lab = cb[idx], np.load(R + v1 + "/extract/labels.npy")[idx]
    names = [n.replace(" (embed)", "").replace("L6 (pre-LN)", "L6pre").replace("L6 (post-LN)", "L6post")
             for n in m1["layer_names"][:-1]]
    print(f"== {name}   frames: {' '.join(f'{n:>6}' for n in names)}")
    for rnf in (False, True):
        f0, f1 = frames(v0, None, rnf), frames(v1, counts > 0, rnf)
        sim = np.array([1 - _per_row_change(knn_indices(a[idx], K), knn_indices(b[idx], K))
                        for a, b in zip(f0, f1)])  # (frames, tokens)
        print(f"  {'unit-norm 1st' if rnf else 'center 1st':14} all        {' '.join(f'{v:6.2f}' for v in sim.mean(1))}")
        if not rnf:
            continue
        for b in range(5):
            print(f"  {'':14} freq bin {b}  {' '.join(f'{v:6.2f}' for v in sim[:, fb == b].mean(1))}")
        for c in np.unique(lab):
            if (lab == c).sum() >= 50:
                print(f"  {'':14} {CATEGORIES[c][:9]:9}  {' '.join(f'{v:6.2f}' for v in sim[:, lab == c].mean(1))}  n={(lab == c).sum()}")
