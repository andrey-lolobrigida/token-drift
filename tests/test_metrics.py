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


def test_knn_purity_is_fraction_of_neighbors_sharing_label():
    from token_drift.metrics import knn_purity

    nb = np.array([[1, 2], [0, 2], [0, 1]])
    labels = np.array([0, 0, 1])
    # row0: neighbors labels (0,1) -> 0.5; row1: (0,1) -> 0.5; row2: (0,0) -> 0.0
    assert knn_purity(nb, labels) == pytest.approx((0.5 + 0.5 + 0.0) / 3)


def test_compute_all_includes_purity_and_shuffled_control(clustered):
    x, labels = clustered
    m = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=None)
    assert m["knn_purity"] == pytest.approx([1.0, 1.0])
    assert len(m["knn_purity_shuffled"]) == 2
    assert m["knn_purity_shuffled"][0] < 0.5


# ---- anisotropy (Ethayarajh 2019) and change-by-frequency (Voita et al. 2019) ----

def test_anisotropy_is_mean_cosine_of_random_pairs(rng):
    from token_drift.metrics import anisotropy

    iso = rng.normal(size=(2000, 32))
    assert abs(anisotropy(iso, n_pairs=5000, seed=0)) < 0.05
    cone = iso + 10.0  # a big shared offset: everything points the same way
    assert anisotropy(cone, n_pairs=5000, seed=0) > 0.9


def test_top_pc_share_sees_a_stretch_but_not_an_offset(rng):
    from token_drift.metrics import anisotropy, top_pc_share

    d = 64
    iso = rng.normal(size=(3000, d))
    assert top_pc_share(iso) < 3 / d  # isotropic: every PC ~1/d of the variance
    stretched = iso.copy()
    stretched[:, 0] *= 30  # one massive dimension
    assert top_pc_share(stretched) > 0.9
    # a shared offset makes mean-cos anisotropy fire but is NOT a variance direction
    offset = iso + 50.0
    assert anisotropy(offset, n_pairs=2000) > 0.9
    assert top_pc_share(offset) < 3 / d


def test_anisotropy_is_seeded(rng):
    from token_drift.metrics import anisotropy

    x = rng.normal(size=(500, 8))
    assert anisotropy(x, n_pairs=100, seed=1) == anisotropy(x, n_pairs=100, seed=1)


def test_knn_change_by_bin_groups_per_token_change():
    from token_drift.metrics import knn_change_by_bin

    a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    b = np.array([[1, 2], [3, 9], [0, 0], [7, 8]])  # jaccard 1, 1/3, 0, 1 -> change 0, 2/3, 1, 0
    bins = np.array([0, 0, 1, 1])
    out = knn_change_by_bin(a, b, bins, n_bins=2)
    assert out == pytest.approx([(0 + 2 / 3) / 2, (1 + 0) / 2])


def test_knn_change_by_bin_empty_bin_is_nan():
    from token_drift.metrics import knn_change_by_bin

    a = np.array([[1, 2], [3, 4]])
    out = knn_change_by_bin(a, a, np.array([0, 0]), n_bins=3)
    assert out[0] == 0.0 and np.isnan(out[1]) and np.isnan(out[2])


def test_compute_all_takes_raw_layers_and_freq_bins(clustered, rng):
    x, labels = clustered
    raw = [x * 3 + 1, x]  # uncentered versions; anisotropy is computed on these
    bins = rng.integers(0, 3, size=len(labels))
    m = compute_all(
        [x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=None,
        raw_layers=raw, freq_bins=bins, n_freq_bins=3,
    )
    assert len(m["anisotropy"]) == 2 and m["anisotropy"][0] > m["anisotropy"][1]
    # scale + shift leaves the top-PC share alone: it's about shape after centering
    assert len(m["top_pc_share"]) == 2 and abs(m["top_pc_share"][0] - m["top_pc_share"][1]) < 1e-4
    assert np.asarray(m["knn_change_by_freq"]).shape == (1, 3)  # transitions x bins
    assert m["n_freq_bins"] == 3
    import json; json.dumps(m)


def test_compute_all_without_extras_still_works(clustered):
    x, labels = clustered
    m = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=None)
    assert m["anisotropy"] is None and m["top_pc_share"] is None and m["knn_change_by_freq"] is None


# ---- k sweep (Q10): same neighbourhood metrics at several k ----

def test_knn_sorted_is_nearest_first_and_same_set_as_knn_indices(clustered):
    from token_drift.metrics import knn_sorted

    x, _ = clustered
    nb = knn_sorted(x, k=10)
    assert {frozenset(r) for r in nb} == {frozenset(r) for r in knn_indices(x, k=10)}
    sims = np.take_along_axis(x @ x.T, nb.astype(np.int64), axis=1)
    assert np.all(np.diff(sims, axis=1) <= 1e-6)  # nearest first


def test_knn_sweep_at_the_config_k_matches_compute_all(clustered, rng):
    from token_drift.metrics import knn_sweep

    x, labels = clustered
    layers = [x, _unit(x + rng.normal(size=x.shape) * 0.5), _unit(rng.normal(size=x.shape))]
    names = ["L0", "L1", "unembed"]
    m = compute_all(layers, names, labels, knn_k=5, kmeans_k=4, seed=0, subsample=150)
    s = knn_sweep(layers, names, labels, ks=[3, 5, 20], idx=np.array(m["subsample_idx"]), seed=0)
    assert s["ks"] == [3, 5, 20]
    at5 = s["by_k"]["5"]
    for key in ("knn_consecutive", "knn_vs_first", "knn_vs_last", "knn_purity", "knn_purity_shuffled"):
        assert at5[key] == pytest.approx(m[key]), key
    # chance Jaccard of two random k-sets grows with k but stays tiny next to real overlap
    assert 0 < s["by_k"]["3"]["chance_overlap"] < s["by_k"]["20"]["chance_overlap"] < 0.1
    import json
    json.dumps(s)


# ---- v1: eligibility, forced subsample, merge-rank table, self-sim ----

def test_subsample_is_drawn_from_eligible_only(clustered):
    x, labels = clustered
    eligible = np.zeros(200, bool)
    eligible[::2] = True
    m = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=50, eligible=eligible)
    assert len(m["subsample_idx"]) == 50 and set(m["subsample_idx"]) <= set(np.flatnonzero(eligible))
    assert m["eligible_n"] == 100


def test_all_eligible_reproduces_the_v0_draw(clustered):
    x, labels = clustered
    a = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=50)
    b = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=50,
                    eligible=np.ones(200, bool))
    assert a["subsample_idx"] == b["subsample_idx"]


def test_fewer_eligible_than_subsample_warns_and_uses_all(clustered):
    x, labels = clustered
    eligible = np.zeros(200, bool)
    eligible[:30] = True
    with pytest.warns(UserWarning, match="eligible"):
        m = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=50, eligible=eligible)
    assert m["subsample_idx"] == list(range(30))


def test_subsample_idx_overrides_the_draw(clustered):
    x, labels = clustered
    m = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=50,
                    subsample_idx=np.arange(10, 60))
    assert m["subsample_idx"] == list(range(10, 60))


def test_self_sim_curves_are_means_over_the_subsample(clustered):
    x, labels = clustered
    ss = np.full((2, 200), 0.5, np.float32)
    ss[1] = 0.3
    ss[1, :100] = np.nan  # never drawn below: only rows >= 100 are eligible
    eligible = np.arange(200) >= 100
    m = compute_all([x, x, x], ["a", "b", "unembed"], labels, knn_k=5, kmeans_k=4, seed=0,
                    subsample=None, eligible=eligible, self_sim=ss, self_sim_baseline=[0.1, 0.2])
    assert m["self_sim"] == pytest.approx([0.5, 0.3])  # one per extract frame, no unembed
    assert m["self_sim_baseline"] == pytest.approx([0.1, 0.2])
    assert m["self_sim_adjusted"] == pytest.approx([0.4, 0.1])
    import json
    json.dumps(m)


def test_corpus_bins_with_the_merge_rank_table_riding_along(clustered, rng):
    x, labels = clustered
    m = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=None,
                    freq_bins=rng.integers(0, 5, 200), n_freq_bins=5, freq_bins_source="corpus",
                    merge_rank_bins=rng.integers(0, 6, 200), n_merge_rank_bins=6)
    assert m["freq_bins_source"] == "corpus"
    assert np.asarray(m["knn_change_by_freq"]).shape == (1, 5)
    assert np.asarray(m["knn_change_by_merge_rank"]).shape == (1, 6)


def test_v1_keys_are_null_on_a_v0_style_call(clustered):
    x, labels = clustered
    m = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=None)
    for k in ("self_sim", "self_sim_baseline", "self_sim_adjusted", "knn_change_by_merge_rank", "freq_bins_source"):
        assert m[k] is None, k
    assert m["eligible_n"] == 200
