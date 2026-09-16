"""Projection + flipbook + metric plots. Tiny inputs; we only check shapes and files."""
import json

import numpy as np
import pytest

from token_drift.viz import plot_flipbook, plot_metrics, project_layers

N, D, L = 80, 8, 3


@pytest.fixture
def layers(rng):
    base = rng.normal(size=(N, D))
    base[:40] += 6  # two blobs so UMAP has something to find
    return [base + rng.normal(size=(N, D)) * 0.1 for _ in range(L)]


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("method", ["stacked_umap", "aligned_umap"])
def test_project_layers_shape(layers, method):
    coords = project_layers(layers, method=method, n_neighbors=5, min_dist=0.1, seed=0)
    assert coords.shape == (L, N, 2)
    assert np.isfinite(coords).all()


def test_project_layers_rejects_unknown_method(layers):
    with pytest.raises(ValueError):
        project_layers(layers, method="tsne", n_neighbors=5, min_dist=0.1, seed=0)


def test_plot_flipbook_writes_frames_gif_and_coords(tmp_path, rng):
    coords = rng.normal(size=(L, N, 2))
    labels = rng.integers(0, 10, size=N)
    names = ["L0 (embed)", "L1", "unembed"]
    traj = {" bank": 3, " river": 7}
    files = plot_flipbook(coords, labels, names, tmp_path, trajectories=traj)
    for i in range(L):
        assert (tmp_path / f"umap_layer_{i}.png").exists()
    assert (tmp_path / "flipbook.gif").exists()
    assert (tmp_path / "umap_coords.npy").exists()
    assert np.load(tmp_path / "umap_coords.npy").shape == (L, N, 2)
    assert (tmp_path / "trajectories.png").exists()
    assert len(files) >= L + 2


def _fake_metrics(rng, L=4):
    names = [f"L{i}" for i in range(L - 1)] + ["unembed"]
    return {
        "layer_names": names,
        "knn_consecutive": rng.uniform(size=L - 1).tolist(),
        "knn_vs_first": rng.uniform(size=L).tolist(),
        "knn_vs_last": rng.uniform(size=L).tolist(),
        "cka": rng.uniform(size=(L, L)).tolist(),
        "silhouette": rng.uniform(size=L).tolist(),
        "silhouette_shuffled": (rng.uniform(size=L) * 0.01).tolist(),
        "kmeans_ari_consecutive": rng.uniform(size=L - 1).tolist(),
    }


def test_plot_metrics_single_and_overlay(tmp_path, rng):
    m1, m2 = _fake_metrics(rng), _fake_metrics(rng)
    plot_metrics({"pythia70m": m1}, tmp_path / "one.png")
    plot_metrics({"pythia70m": m1, "random_init": m2}, tmp_path / "two.png")
    assert (tmp_path / "one.png").exists() and (tmp_path / "two.png").exists()
    json.dumps(m1)  # sanity: the fake matches what metrics.json holds


def test_plot_cka_heatmap(tmp_path, rng):
    from token_drift.viz import plot_cka_heatmap

    m = _fake_metrics(rng)
    plot_cka_heatmap(m, tmp_path / "cka.png")
    assert (tmp_path / "cka.png").exists()
