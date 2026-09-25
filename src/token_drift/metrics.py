"""Layer-to-layer drift metrics. Everything here takes normalized (vocab, d) arrays.

The headline number is kNN overlap between consecutive layers: no projection, no
clustering hyperparameters, just "did this token's neighbors change".
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def knn_indices(x: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
    """(n, d) unit rows -> (n, k) indices of the k nearest neighbors by cosine, self excluded.

    Brute force via matmul in chunks: for 10k x 512 this is a 10k x 10k float32 sim
    matrix built 2048 rows at a time, which is faster than any tree at d=512.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    out = np.empty((n, k), dtype=np.int32)
    for start in range(0, n, chunk):
        sims = x[start : start + chunk] @ x.T  # rows are unit-norm so this is cosine
        rows = np.arange(sims.shape[0])
        sims[rows, rows + start] = -np.inf  # never your own neighbor
        # argpartition gives the top-k unordered in O(n); we don't need them sorted
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        out[start : start + chunk] = top
    return out


def knn_sorted(x: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
    """Like `knn_indices`, but each row is ordered nearest first.

    The k sweep needs this: the first 5 of a sorted top-100 *are* the top-5, so one
    neighbour search at the biggest k serves every smaller k.
    """
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    out = np.empty((n, k), dtype=np.int32)
    for start in range(0, n, chunk):
        sims = x[start : start + chunk] @ x.T
        rows = np.arange(sims.shape[0])
        sims[rows, rows + start] = -np.inf
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
        out[start : start + chunk] = np.take_along_axis(top, order, axis=1)
    return out


def knn_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Jaccard of per-row neighbor sets. a, b: (n, k) index arrays."""
    k = a.shape[1]
    inter = np.array([len(np.intersect1d(ra, rb)) for ra, rb in zip(a, b)])
    return float(np.mean(inter / (2 * k - inter)))


def knn_purity(neighbors: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of each token's k neighbors that share its label, averaged over tokens.

    Much more sensitive than silhouette in high dimension: silhouette wants compact,
    convex clusters, while purity only asks whether *local* neighborhoods are pure.
    On pythia-70m silhouette sits at ~0 for every layer while purity is ~0.7 vs ~0.2
    shuffled, so this is the number to look at for "does surface form cluster".
    """
    labels = np.asarray(labels)
    return float((labels[neighbors] == labels[:, None]).mean())


def anisotropy(x: np.ndarray, *, n_pairs: int = 4000, seed: int = 0) -> float:
    """Ethayarajh (2019): mean cosine between random pairs of rows. 0 = isotropic, 1 = a cone.

    Meant for *raw* activations: after centering + unit-norm this is ~0 by construction.
    """
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, x.shape[0], n_pairs)
    j = rng.integers(0, x.shape[0], n_pairs)
    keep = i != j
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return float((xn[i[keep]] * xn[j[keep]]).sum(1).mean())


def top_pc_share(x: np.ndarray) -> float:
    """Fraction of total variance carried by the top principal component, after centering.

    The shape check mean-cosine anisotropy can't do. A shared offset makes every random
    pair's cosine high but vanishes under centering; a single stretched coordinate (a
    "massive activation" dimension) survives centering and shows up here instead.
    """
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean(0)
    s = np.linalg.svd(x, compute_uv=False)  # singular values; s**2 are the PC variances
    return float(s[0] ** 2 / (s**2).sum())


def _per_row_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    k = a.shape[1]
    inter = np.array([len(np.intersect1d(ra, rb)) for ra, rb in zip(a, b)])
    return 1.0 - inter / (2 * k - inter)


def knn_change_by_bin(a: np.ndarray, b: np.ndarray, bins: np.ndarray, n_bins: int) -> list[float]:
    """Mean per-token neighborhood change (1 - Jaccard) grouped by an integer bin label.

    Voita et al. (2019) fig. 4: in a trained LM, frequent tokens change more per layer.
    Empty bins come back as NaN rather than crashing the whole metrics stage.
    """
    change = _per_row_change(a, b)
    bins = np.asarray(bins)
    return [float(change[bins == k].mean()) if (bins == k).any() else float("nan") for k in range(n_bins)]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019). Invariant to rotation and isotropic scaling."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(0)
    y = y - y.mean(0)
    # ||X^T Y||_F^2 / (||X^T X||_F ||Y^T Y||_F), the (d x d) form: cheaper than (n x n)
    hsic = np.linalg.norm(x.T @ y) ** 2
    return float(hsic / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    # Rows are unit-norm, so euclidean is monotone in cosine; euclidean is what sklearn
    # does fastest. sklearn wants >= 2 distinct labels; a tiny forced-eligible subsample
    # (v1's "too few eligible tokens" fallback) can land on just one, so NaN it instead
    # of crashing the whole metrics stage.
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(x, labels, metric="euclidean"))


def kmeans_ari(x: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    """ARI between k-means partitions of two layers. Cheap cross-check on kNN overlap."""
    la = KMeans(n_clusters=k, random_state=seed, n_init=3).fit_predict(x)
    lb = KMeans(n_clusters=k, random_state=seed, n_init=3).fit_predict(y)
    return float(adjusted_rand_score(la, lb))


def compute_all(
    layers: list[np.ndarray],
    layer_names: list[str],
    labels: np.ndarray,
    *,
    knn_k: int,
    kmeans_k: int,
    seed: int,
    subsample: int | None,
    raw_layers: list[np.ndarray] | None = None,
    freq_bins: np.ndarray | None = None,
    n_freq_bins: int = 5,
    freq_bins_source: str | None = None,
    eligible: np.ndarray | None = None,
    subsample_idx: np.ndarray | None = None,
    merge_rank_bins: np.ndarray | None = None,
    n_merge_rank_bins: int = 6,
    self_sim: np.ndarray | None = None,
    self_sim_baseline: list[float] | None = None,
) -> dict:
    """All per-layer curves in one JSON-serializable dict.

    `layers` should already include the unembed pseudo-layer as the last entry.
    One random subsample of tokens is drawn once and reused for every layer.
    `raw_layers` (uncentered, same order) is only used for the anisotropy curve;
    `freq_bins` (one int per token) enables the change-by-frequency table.
    v1 extras: `eligible` limits the draw (count >= min_count), `subsample_idx` skips it
    (v0v1 reuses v1's tokens), `merge_rank_bins` adds a second Voita table next to the
    corpus one, `self_sim` (extract frames x vocab) + baseline add Ethayarajh's curves.
    """
    rng = np.random.default_rng(seed)
    n = layers[0].shape[0]
    pool = np.arange(n) if eligible is None else np.flatnonzero(eligible)
    if subsample_idx is not None:
        idx = np.asarray(subsample_idx)
    elif subsample is not None and subsample < len(pool):
        # draw positions in the pool, not ids: with pool = every token this is the v0 draw exactly
        idx = np.sort(pool[rng.choice(len(pool), size=subsample, replace=False)])
    else:
        if subsample is not None and eligible is not None and subsample > len(pool):
            warnings.warn(f"only {len(pool)} eligible tokens, fewer than subsample={subsample}; "
                          "using all of them", stacklevel=2)
        idx = pool
    xs = [np.asarray(layer[idx], dtype=np.float32) for layer in layers]
    lab = np.asarray(labels)[idx]
    shuffled = rng.permutation(lab)  # control: silhouette on permuted labels should be ~0

    knn = [knn_indices(x, knn_k) for x in xs]
    L = len(xs)
    aniso = pc_share = None
    if raw_layers is not None:
        raws = [np.asarray(r[idx], dtype=np.float32) for r in raw_layers]
        aniso = [anisotropy(r, seed=seed) for r in raws]
        pc_share = [top_pc_share(r) for r in raws]  # centers internally; raw in, shape out
    by_freq = None
    if freq_bins is not None:
        fb = np.asarray(freq_bins)[idx]
        by_freq = [knn_change_by_bin(knn[i], knn[i + 1], fb, n_freq_bins) for i in range(L - 1)]
    by_rank = None
    if merge_rank_bins is not None:
        mb = np.asarray(merge_rank_bins)[idx]
        by_rank = [knn_change_by_bin(knn[i], knn[i + 1], mb, n_merge_rank_bins) for i in range(L - 1)]
    ss = ss_base = ss_adj = None
    if self_sim is not None:
        # one value per extract frame; the unembed pseudo-layer has no occurrences
        ss = [float(np.nanmean(np.asarray(row)[idx])) for row in self_sim]
        ss_base = [float(b) for b in self_sim_baseline]
        ss_adj = [s - b for s, b in zip(ss, ss_base)]  # Ethayarajh's anisotropy-adjusted self-sim
    cka = [[linear_cka(xs[i], xs[j]) for j in range(L)] for i in range(L)]
    return {
        "layer_names": list(layer_names),
        "knn_k": knn_k,
        "kmeans_k": kmeans_k,
        "subsample_idx": idx.tolist(),
        "knn_consecutive": [knn_overlap(knn[i], knn[i + 1]) for i in range(L - 1)],
        "knn_vs_first": [knn_overlap(knn[0], knn[i]) for i in range(L)],
        "knn_vs_last": [knn_overlap(knn[-1], knn[i]) for i in range(L)],
        "cka": cka,
        "knn_purity": [knn_purity(nb, lab) for nb in knn],
        "knn_purity_shuffled": [knn_purity(nb, shuffled) for nb in knn],
        "silhouette": [silhouette(x, lab) for x in xs],
        "silhouette_shuffled": [silhouette(x, shuffled) for x in xs],
        "kmeans_ari_consecutive": [
            kmeans_ari(xs[i], xs[i + 1], kmeans_k, seed) for i in range(L - 1)
        ],
        "anisotropy": aniso,
        "top_pc_share": pc_share,
        "knn_change_by_freq": by_freq,
        "n_freq_bins": n_freq_bins if freq_bins is not None else None,
        "freq_bins_source": (freq_bins_source or "merge_rank") if freq_bins is not None else None,
        "knn_change_by_merge_rank": by_rank,
        "n_merge_rank_bins": n_merge_rank_bins if merge_rank_bins is not None else None,
        "eligible_n": int(len(pool)),
        "self_sim": ss,
        "self_sim_baseline": ss_base,
        "self_sim_adjusted": ss_adj,
    }


def knn_sweep(
    layers: list[np.ndarray],
    layer_names: list[str],
    labels: np.ndarray,
    *,
    ks: list[int],
    idx: np.ndarray,
    seed: int,
) -> dict:
    """The k-dependent curves of `compute_all` (overlaps, purity) at several k (Q10).

    `idx` should be the subsample `compute_all` used (metrics.json's `subsample_idx`),
    so the sweep at the config's k reproduces metrics.json exactly.
    """
    xs = [np.asarray(layer[idx], dtype=np.float32) for layer in layers]
    lab = np.asarray(labels)[idx]
    # same permutation compute_all draws: its rng does choice() first, then permutation()
    rng = np.random.default_rng(seed)
    if len(idx) < layers[0].shape[0]:
        rng.choice(layers[0].shape[0], size=len(idx), replace=False)
    shuffled = rng.permutation(lab)
    ks = sorted(int(k) for k in ks)
    full = [knn_sorted(x, ks[-1]) for x in xs]
    n, L = len(idx), len(xs)
    by_k = {}
    for k in ks:
        knn = [nb[:, :k] for nb in full]
        # two random k-subsets of n-1 tokens share k^2/(n-1) on average
        inter = k * k / (n - 1)
        by_k[str(k)] = {
            "knn_consecutive": [knn_overlap(knn[i], knn[i + 1]) for i in range(L - 1)],
            "knn_vs_first": [knn_overlap(knn[0], knn[i]) for i in range(L)],
            "knn_vs_last": [knn_overlap(knn[-1], knn[i]) for i in range(L)],
            "knn_purity": [knn_purity(nb, lab) for nb in knn],
            "knn_purity_shuffled": [knn_purity(nb, shuffled) for nb in knn],
            "chance_overlap": inter / (2 * k - inter),
        }
    return {"layer_names": list(layer_names), "ks": ks, "n": int(n), "by_k": by_k}


def cross_overlap(
    a_layers: list[np.ndarray], b_layers: list[np.ndarray], idx: np.ndarray, k: int
) -> list[float]:
    """Per frame: kNN overlap between run A's frame f and run B's frame f, same tokens.

    1 = the two ways of building a vocab matrix (v0 alone vs v1 in context) agree on every
    token's neighbours at that depth. Q6 reads "does context take over?" off how it falls.
    """
    out = []
    for a, b in zip(a_layers, b_layers):
        na = knn_indices(np.asarray(a[idx], dtype=np.float32), k)
        nb = knn_indices(np.asarray(b[idx], dtype=np.float32), k)
        out.append(knn_overlap(na, nb))
    return out
