"""v2: how training moves the vocab geometry. Arrays in, numbers out.

"with final" = how close a checkpoint's frame already is to the finished model's same frame.
kNN overlap asks it locally (does each token have its final neighbours yet?), CKA globally
(is the whole cloud in its final shape?). They can disagree, and that split is interesting
(cf. FINDINGS 11.2).
"""
from __future__ import annotations

import numpy as np

from token_drift.metrics import knn_indices, knn_overlap, linear_cka


def overlap_with_final(x: np.ndarray, x_final: np.ndarray, k: int) -> float:
    """Mean Jaccard of each row's k-NN set in `x` vs in `x_final`. Rows = the same tokens."""
    return knn_overlap(knn_indices(x, k), knn_indices(x_final, k))


def cka_with_final(x: np.ndarray, x_final: np.ndarray) -> float:
    return linear_cka(x, x_final)


def row_drift(w_t: np.ndarray, w_0: np.ndarray) -> np.ndarray:
    """||W_t[i] - W_0[i]|| / ||W_0[i]|| per row: how far each row moved, in units of its own
    starting length. Upcast first: the matrices sit on disk in float16."""
    a = np.asarray(w_t, dtype=np.float32)
    b = np.asarray(w_0, dtype=np.float32)
    num = np.linalg.norm(a - b, axis=1)
    den = np.linalg.norm(b, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan).astype(np.float32)  # a zero init row has no scale


def median_by_bin(values: np.ndarray, bins: np.ndarray, n_bins: int) -> list[float]:
    """Median of `values` inside each bin 0..n_bins-1. Median, not mean: a handful of rows that
    blow up (an attention-sink token, say) shouldn't drag a whole frequency bin."""
    values, bins = np.asarray(values, dtype=float), np.asarray(bins)
    out = []
    for b in range(n_bins):
        v = values[(bins == b) & ~np.isnan(values)]
        out.append(float(np.median(v)) if len(v) else float("nan"))
    return out


def half_way_step(values, steps: list[int], frac: float = 0.5) -> int | None:
    """First step where `values` has covered `frac` of the way from its first to its last value.

    The 'when did X appear' number for FINDINGS 13, fixed before looking at any curve so the
    step isn't picked by eye. None when the curve ends where it started (nothing to cover).
    """
    v = np.asarray(values, dtype=float)
    span = v[-1] - v[0]
    if not np.isfinite(span) or span == 0:
        return None
    reached = (v - v[0]) / span >= frac
    return int(steps[int(np.argmax(reached))])  # the last point always reaches, so argmax finds one
