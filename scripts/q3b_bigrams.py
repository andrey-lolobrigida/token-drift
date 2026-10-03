"""FINDINGS 11.5 / Q3 (b): is the last hidden state "what follows t" and the unembed
"what t follows"?

One-off check, not part of the pipeline. Model-free token vectors from corpus bigrams:
successor vector = which tokens follow t, predecessor vector = which tokens precede t.
PPMI-weighted (raw counts are all " the" and ","), SVD to 256-d, then the usual
center -> unit-norm and k=10 kNN overlap against model frames on the metrics tokens.
`uv run python scripts/q3b_bigrams.py`, a few min on CPU.
"""
import json
import sys
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from token_drift.cli import corpus_eligibility
from token_drift.metrics import _per_row_change, knn_indices, knn_overlap
from token_drift.normalize import normalize_layer

R = "runs/"
K, SEED = 10, 0
DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 256  # robustness: try 100 / 256 / 500
ALPHA = 0.75  # context-distribution smoothing (Levy et al. 2015): stops rare contexts getting huge PMI

stream = np.load(R + "pythia70m_corpus/corpus/windows.npy").ravel().astype(np.int64)
idx = np.array(json.load(open(R + "pythia70m_corpus/metrics/metrics.json"))["subsample_idx"])
acts = np.load(R + "pythia70m_corpus/normalize/acts_norm.npy", mmap_mode="r")
V = acts.shape[1]
counts = np.load(R + "pythia70m_corpus/extract/counts.npy")
C = sp.csr_matrix((np.ones(len(stream) - 1, dtype=np.float64), (stream[:-1], stream[1:])), shape=(V, V))
C.sum_duplicates()  # C[a, b] = how often b follows a


def ppmi_svd(M):
    """Rows = the metrics tokens, columns = their contexts. PPMI then SVD."""
    total = M.sum()
    row = np.asarray(M.sum(1)).ravel()
    col = np.asarray(M.sum(0)).ravel() ** ALPHA
    col /= col.sum()
    M = M[idx].tocoo()
    pmi = np.log(M.data / total) - np.log(row[idx][M.row] / total) - np.log(col[M.col])
    keep = pmi > 0
    P = sp.csr_matrix((pmi[keep], (M.row[keep], M.col[keep])), shape=(len(idx), V))
    return TruncatedSVD(DIM, random_state=SEED).fit_transform(P)


def knn(x):
    return knn_indices(normalize_layer(x, center=True, unit_norm=True, drop_top_pcs=0), K)


ctx = {"successors": knn(ppmi_svd(C)), "predecessors": knn(ppmi_svd(C.T.tocsr()))}
v0 = np.load(R + "pythia70m/normalize/acts_norm.npy", mmap_mode="r")
frames = {"embed (L0)": acts[0], "v0 last (L6post)": v0[7], "v1 last (L6post)": acts[7], "unembed": acts[8]}
fb = corpus_eligibility(counts, 1)[1][idx]  # 0 = most frequent ... 4 = rarest
print(f"SVD dim {DIM}. Overlap, all tokens, then by corpus-frequency bin 0..4 (chance ~0.0005)")
print(f"{'':18} {'successors':>11} {'predecessors':>13}   succ by bin / pred by bin")
for name, f in frames.items():
    nb = knn_indices(np.asarray(f[idx], dtype=np.float32), K)
    s_, p_ = (1 - _per_row_change(nb, ctx[c]) for c in ("successors", "predecessors"))
    sb = " ".join(f"{s_[fb == b].mean():.2f}" for b in range(5))
    pb = " ".join(f"{p_[fb == b].mean():.2f}" for b in range(5))
    print(f"{name:18} {knn_overlap(nb, ctx['successors']):11.2f} {knn_overlap(nb, ctx['predecessors']):13.2f}   {sb} / {pb}")
print(f"{'succ vs pred':18} {knn_overlap(ctx['successors'], ctx['predecessors']):11.2f}")
print("min count in metrics tokens:", counts[idx].min())
