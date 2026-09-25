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
        "knn_purity": rng.uniform(size=L).tolist(),
        "knn_purity_shuffled": (rng.uniform(size=L) * 0.2).tolist(),
        "silhouette": rng.uniform(size=L).tolist(),
        "silhouette_shuffled": (rng.uniform(size=L) * 0.01).tolist(),
        "kmeans_ari_consecutive": rng.uniform(size=L - 1).tolist(),
        "anisotropy": rng.uniform(size=L).tolist(),
        "top_pc_share": rng.uniform(size=L).tolist(),
        "knn_change_by_freq": rng.uniform(size=(L - 1, 5)).tolist(),
        "n_freq_bins": 5,
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


def test_plot_metrics_has_six_panels_and_tolerates_missing_extras(tmp_path, rng):
    import matplotlib.pyplot as plt

    m = _fake_metrics(rng)
    fig = plot_metrics({"r": m}, tmp_path / "six.png", return_fig=True)
    assert len(fig.axes) == 6
    plt.close(fig)
    m["anisotropy"] = None
    m["knn_change_by_freq"] = None
    plot_metrics({"r": m}, tmp_path / "four.png")  # old-format metrics.json must still plot
    assert (tmp_path / "four.png").exists()


def test_anisotropy_panel_gets_top_pc_share_line_when_present(tmp_path, rng):
    import matplotlib.pyplot as plt

    m = _fake_metrics(rng)
    fig = plot_metrics({"r": m}, tmp_path / "a.png", return_fig=True)
    ax_aniso = fig.axes[4]
    labels = [ln.get_label() for ln in ax_aniso.get_lines()]
    assert len(labels) == 2 and any("raw" in l for l in labels) and any("centered" in l for l in labels)
    plt.close(fig)
    del m["top_pc_share"]  # metrics.json from before the field existed
    fig = plot_metrics({"r": m}, tmp_path / "b.png", return_fig=True)
    assert len(fig.axes[4].get_lines()) == 1
    plt.close(fig)


def _hue(c):
    import colorsys

    from matplotlib.colors import to_rgb

    return colorsys.rgb_to_hls(*to_rgb(c))[0]


def test_freq_panel_ramp_follows_each_runs_own_hue(tmp_path, rng):
    # colour follows the entity: run 3's frequency lines must be run 3's hue, not a
    # recycled second-run orange
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    from token_drift.viz import RUN_COLORS

    runs = {f"r{i}": _fake_metrics(rng) for i in range(3)}
    fig = plot_metrics(runs, tmp_path / "three.png", return_fig=True)
    lines = fig.axes[5].get_lines()
    per_run = len(lines) // 3
    for i in range(3):
        mine = lines[i * per_run : (i + 1) * per_run]
        for ln in mine:
            assert abs(_hue(ln.get_color()) - _hue(RUN_COLORS[i])) < 0.03
        # light -> dark with bin index (bin is a magnitude)
        light = [sum(to_rgb(ln.get_color())) for ln in mine]
        assert light == sorted(light, reverse=True)
    plt.close(fig)


def test_tick_labels_tilt_once_the_pre_ln_frame_makes_them_crowded(tmp_path, rng):
    import matplotlib.pyplot as plt

    m = _fake_metrics(rng, L=9)  # pythia's frame count since the pre-LN frame
    m["layer_names"] = ["L0 (embed)"] + [f"L{i}" for i in range(1, 6)] + ["L6 (pre-LN)", "L6 (post-LN)", "unembed"]
    fig = plot_metrics({"r": m}, tmp_path / "t.png", return_fig=True)
    for ax in fig.axes:
        assert all(t.get_rotation() == 45 for t in ax.get_xticklabels())
    # tilted transition labels go on one line, or neighbours overlap
    assert all("\n" not in t.get_text() for t in fig.axes[0].get_xticklabels())
    plt.close(fig)


def test_one_figure_legend_below_the_grid_and_none_on_the_panels(tmp_path, rng):
    # loc="best" per panel landed on top of data lines; one shared legend can't
    import matplotlib.pyplot as plt

    runs = {"a": _fake_metrics(rng), "b": _fake_metrics(rng)}
    fig = plot_metrics(runs, tmp_path / "leg.png", return_fig=True)
    assert all(ax.get_legend() is None for ax in fig.axes)
    texts = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
    assert "a" in texts and "b" in texts  # runs named once, not once per panel
    assert any("shuffled" in t for t in texts) and any("top-PC" in t for t in texts)
    plt.close(fig)


def test_self_sim_row_appears_only_when_a_run_has_it(tmp_path, rng):
    import matplotlib.pyplot as plt

    v0 = _fake_metrics(rng)
    v1 = _fake_metrics(rng)
    v1["self_sim"] = rng.uniform(size=3).tolist()  # extract frames only: no unembed
    v1["self_sim_baseline"] = (rng.uniform(size=3) * 0.3).tolist()
    v1["self_sim_adjusted"] = (np.array(v1["self_sim"]) - np.array(v1["self_sim_baseline"])).tolist()
    v1["freq_bins_source"] = "corpus"
    fig = plot_metrics({"v0": v0, "v1": v1}, tmp_path / "ss.png", return_fig=True)
    assert len(fig.axes) == 8
    assert len(fig.axes[6].get_lines()) == 2  # self-sim + its anisotropy baseline, v1 only
    assert fig.axes[7].get_ylim()[0] < 0  # adjusted self-sim can go negative
    texts = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
    assert any("baseline" in t for t in texts)
    plt.close(fig)
    fig = plot_metrics({"v0": v0}, tmp_path / "no_ss.png", return_fig=True)
    assert len(fig.axes) == 6
    plt.close(fig)


def test_plot_cross_overlap(tmp_path):
    from token_drift.viz import plot_cross_overlap

    p = plot_cross_overlap([1.0, 0.5, 0.4, 1.0], ["L0 (embed)", "L1", "L2", "unembed"], tmp_path / "c.png", title="v0 vs v1")
    assert p.exists()
