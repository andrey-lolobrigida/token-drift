"""kNN overlap, linear CKA, silhouette, k-means ARI. All on normalized rows."""
import numpy as np
import pytest

from token_drift.metrics import (
    compute_all,
    knn_indices,
    knn_overlap,
    kmeans_ari,
    linear_cka,
    silhouette,
)


def _unit(x):
    x = x - x.mean(0)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def clustered(rng):
    """200 points, 4 tight well-separated clusters in 16-d, plus integer labels."""
    centers = rng.normal(size=(4, 16)) * 10
    labels = np.repeat(np.arange(4), 50)
    x = centers[labels] + rng.normal(size=(200, 16)) * 0.1
    return _unit(x).astype(np.float32), labels


def test_knn_indices_shape_and_excludes_self(clustered):
    x, _ = clustered
    idx = knn_indices(x, k=10)
    assert idx.shape == (200, 10)
    assert not np.any(idx == np.arange(200)[:, None])


def test_knn_indices_finds_same_cluster(clustered):
    x, labels = clustered
    idx = knn_indices(x, k=10)
    assert np.all(labels[idx] == labels[:, None])


def test_knn_overlap_identical_is_one(clustered):
    x, _ = clustered
    idx = knn_indices(x, k=10)
    assert knn_overlap(idx, idx) == pytest.approx(1.0)


def test_knn_overlap_random_is_near_zero(rng):
    a = knn_indices(_unit(rng.normal(size=(500, 16))).astype(np.float32), k=10)
    b = knn_indices(_unit(rng.normal(size=(500, 16))).astype(np.float32), k=10)
    assert knn_overlap(a, b) < 0.05


def test_knn_overlap_is_mean_jaccard():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2, 9], [4, 5, 6]])  # jaccard 2/4 and 3/3
    assert knn_overlap(a, b) == pytest.approx((0.5 + 1.0) / 2)


def test_cka_self_is_one_and_rotation_invariant(clustered, rng):
    x, _ = clustered
    q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
    assert linear_cka(x, x) == pytest.approx(1.0)
    assert linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-5)


def test_cka_unrelated_is_small(rng):
    x = rng.normal(size=(300, 16)).astype(np.float32)
    y = rng.normal(size=(300, 16)).astype(np.float32)
    assert linear_cka(x, y) < 0.2


def test_silhouette_high_for_true_labels_low_for_shuffled(clustered, rng):
    x, labels = clustered
    assert silhouette(x, labels) > 0.8
    assert abs(silhouette(x, rng.permutation(labels))) < 0.1


def test_kmeans_ari_identical_is_one(clustered):
    x, _ = clustered
    assert kmeans_ari(x, x, k=4, seed=0) == pytest.approx(1.0)


def test_kmeans_ari_is_seeded(clustered, rng):
    x, _ = clustered
    y = _unit(rng.normal(size=x.shape)).astype(np.float32)
    assert kmeans_ari(x, y, k=4, seed=0) == kmeans_ari(x, y, k=4, seed=0)


def test_compute_all_shapes(clustered, rng):
    x, labels = clustered
    layers = [x, x @ np.linalg.qr(rng.normal(size=(16, 16)))[0], _unit(rng.normal(size=x.shape))]
    names = ["L0", "L1", "unembed"]
    m = compute_all(layers, names, labels, knn_k=5, kmeans_k=4, seed=0, subsample=100)
    assert m["layer_names"] == names
    assert len(m["knn_consecutive"]) == 2  # pairs (0,1), (1,2)
    assert len(m["knn_vs_first"]) == 3 and m["knn_vs_first"][0] == pytest.approx(1.0)
    assert len(m["knn_vs_last"]) == 3 and m["knn_vs_last"][-1] == pytest.approx(1.0)
    assert np.asarray(m["cka"]).shape == (3, 3)
    assert len(m["silhouette"]) == 3
    assert len(m["silhouette_shuffled"]) == 3
    assert len(m["kmeans_ari_consecutive"]) == 2
    assert m["subsample_idx"] is not None and len(m["subsample_idx"]) == 100
    # rotation doesn't change neighborhoods, so layer 0 -> 1 should be near-perfect
    assert m["knn_consecutive"][0] > 0.95
    # everything in there must be JSON-serializable plain python
    import json
    json.dumps(m)
