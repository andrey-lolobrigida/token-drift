"""Projection + flipbook + metric plots. Tiny inputs; we only check shapes and files."""
import json

import numpy as np
import pytest

from token_drift import viz
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


def test_plot_trajectories_labels_paths_with_the_given_text(tmp_path, rng):
    from token_drift.viz import plot_trajectories

    coords = rng.normal(size=(L, N, 2))
    labels = rng.integers(0, 10, size=N)
    p = plot_trajectories(coords, labels, ["a", "b", "c"], {"' vice' n=812": 3, "' envy' n=40": 7},
                          tmp_path / "trajectories_vice.png", title="vice")
    assert p.exists()


def test_trajectories_use_a_legend_not_labels_on_the_plot(tmp_path, rng):
    # real groups (virtue: 20 words) often end in one tight clump; labels drawn at the
    # endpoints were unreadable even after un-stacking them (Andrey, 2026-10-03), so the
    # words live in a legend outside the axes and each path is told apart by colour+marker.
    import matplotlib.pyplot as plt

    from token_drift.viz import plot_trajectories

    coords = rng.normal(size=(L, N, 2))
    coords[-1, :12] = 0.0  # 12 paths, same last-frame point
    traj = {f"' word{i}' n={100 * i}": i for i in range(12)}
    fig = plot_trajectories(coords, rng.integers(0, 10, size=N), ["a", "b", "c"], traj,
                            tmp_path / "t.png", return_fig=True)
    ax = fig.axes[0]
    assert len(ax.texts) == 0
    legend = fig.legends[0] if fig.legends else ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == list(traj)
    plt.close(fig)


def test_group_frames_one_zoomed_panel_per_frame_with_every_word(tmp_path, rng):
    # the trajectories plot is 9 layers of spaghetti; this is the comic-strip version:
    # one panel per frame, zoomed onto the group, words keyed by a shared legend.
    import matplotlib.pyplot as plt

    from token_drift.viz import plot_group_frames

    coords = rng.normal(size=(L, N, 2)) * 10
    group = {"' vice'": 3, "' envy'": 7, "' pride'": 11}
    # real groups end up in one clump (that's why the trajectories plot was unreadable)
    coords[:, list(group.values())] = coords[:, [3]] + rng.normal(size=(L, 3, 2))
    fig = plot_group_frames(coords, ["a", "b", "c"], group, tmp_path / "group_vice.png",
                            title="vice", return_fig=True)
    assert (tmp_path / "group_vice.png").exists()
    panels = [ax for ax in fig.axes if ax.get_title(loc="left")]
    assert [ax.get_title(loc="left") for ax in panels] == ["a", "b", "c"]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == list(group)
    for f, ax in enumerate(panels):
        assert len(ax.texts) == 0
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        pts = coords[f, list(group.values())]
        assert (pts[:, 0] > x0).all() and (pts[:, 0] < x1).all()
        assert (pts[:, 1] > y0).all() and (pts[:, 1] < y1).all()
        # zoomed: the window is much smaller than the full scatter's spread
        assert (x1 - x0) < np.ptp(coords[f, :, 0])
    plt.close(fig)


def test_group_frames_zoom_ignores_a_far_straggler(tmp_path, rng):
    # one far-off word used to set the window and squash the rest into a corner
    import matplotlib.pyplot as plt

    from token_drift.viz import plot_group_frames

    coords = rng.normal(size=(L, N, 2)) * 10
    rows = list(range(10))
    coords[:, rows] = rng.normal(size=(L, 10, 2)) * 0.5
    coords[:, 9] = 40.0  # the straggler
    group = {f"w{r}": r for r in rows}
    fig = plot_group_frames(coords, ["a", "b", "c"], group, tmp_path / "g.png", return_fig=True)
    for ax in [ax for ax in fig.axes if ax.get_title(loc="left")]:
        assert np.diff(ax.get_xlim())[0] < 10  # zoomed on the core, not stretched to 40
        assert len(ax.collections) == 1 + 10  # background + every word, straggler pinned
    plt.close(fig)


def test_trajectory_styles_are_pairwise_distinct_for_big_groups():
    # with a colour-only legend, 28 hues off a continuous ramp (the old turbo version)
    # are indistinguishable neighbours; 10 qualitative colours x marker shapes stay apart.
    from token_drift.viz import RUN_COLORS, _trajectory_styles

    assert [c for c, _ in _trajectory_styles(4)] == RUN_COLORS[:4]
    for n in (10, 28):
        styles = _trajectory_styles(n)
        assert len(styles) == n
        assert len({(tuple(np.round(c, 6)) if not isinstance(c, str) else c, m) for c, m in styles}) == n
        assert len({tuple(c) for c, _ in styles}) == min(n, 10)  # colours repeat, shapes don't


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


def test_anisotropy_panel_title_flags_corpus_mean_sources(tmp_path, rng):
    import matplotlib.pyplot as plt

    # a corpus run's "raw" acts are per-token means (unit_mean/raw_mean), not occurrence-
    # level vectors, so the title must not claim it's Ethayarajh's raw-acts number.
    m_acts = _fake_metrics(rng)
    m_acts["anisotropy_source"] = "acts"
    fig = plot_metrics({"r": m_acts}, tmp_path / "acts.png", return_fig=True)
    assert "RAW acts" in fig.axes[4].get_title(loc="left")
    plt.close(fig)

    m_corpus = _fake_metrics(rng)
    m_corpus["anisotropy_source"] = "unit_mean"
    fig = plot_metrics({"r": m_corpus}, tmp_path / "corpus.png", return_fig=True)
    title = fig.axes[4].get_title(loc="left")
    assert "RAW acts" not in title and "self-sim" in title
    plt.close(fig)

    # mixed run set (v0v1-style overlay): fall back to the plain title, don't overclaim
    fig = plot_metrics({"a": m_acts, "b": m_corpus}, tmp_path / "mixed.png", return_fig=True)
    assert "RAW acts" in fig.axes[4].get_title(loc="left")
    plt.close(fig)


def test_panel_titles_stay_inside_their_own_panel(tmp_path, rng):
    # the corpus anisotropy title used to run on into the Voita panel's title next to it
    import matplotlib.pyplot as plt

    for source in ("acts", "unit_mean"):
        m = _fake_metrics(rng)
        m["anisotropy_source"] = source
        fig = plot_metrics({"r": m}, tmp_path / f"{source}.png", return_fig=True)
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            if not ax.get_title(loc="left"):
                continue
            title_box = ax._left_title.get_window_extent(renderer)
            assert title_box.x1 <= ax.get_window_extent(renderer).x1 + 5, ax.get_title(loc="left")
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


def _fake_q16(n_frames=3):
    rng = np.random.default_rng(0)
    present = lambda: {  # noqa: E731
        "counts": [30, 40, 50], "null_words": ["a", "b"],
        "t": [0.5] * n_frames, "d": [0.3] * n_frames, "seg": [0.3] * n_frames,
        "null_pct": [0.0, 40.0, 90.0][:n_frames], "swap_pct": [[50.0, 0.0, 60.0]] * n_frames,
        "td": [[[1.2, 0.8], [0.5, 0.3], [-0.2, 0.9]]] * n_frames,
        "null_td": [rng.uniform(0, 1, size=(2, 2)).tolist() for _ in range(n_frames)],
        "beats_null": [True, False, False][:n_frames], "best_of_three": [True, True, False][:n_frames],
    }
    triples = [
        {"id": "classical/fear/noun/x,y,z", "set": "classical", "concept": "fear", "pos": "noun",
         "words": ["x", "y", "z"], "groups": {"books": present(), "pile": {"missing": ["z"]}}},
        {"id": "everyday/fear/noun/p,q,r", "set": "everyday", "concept": "fear", "pos": "noun",
         "words": ["p", "q", "r"], "groups": {"books": {"missing": ["p"]}, "pile": {"missing": ["p"]}}},
    ]
    return {"layer_names": ["L0 (embed)", "L1 (pre-LN)", "L1 (post-LN)"][:n_frames],
            "groups": ["books", "pile"], "triples": triples}


def test_plot_q16_summary_writes_png(tmp_path):
    p = viz.plot_q16_summary(_fake_q16(), tmp_path / "s.png")
    assert p.exists()


def test_plot_q16_triples_shares_axes_and_skips_all_missing(tmp_path):
    q = _fake_q16()
    fig = viz.plot_q16_triples(q["triples"][:1], q["layer_names"], tmp_path / "t.png", title="x", return_fig=True)
    axes = fig.axes[:3]  # one present row x 3 frames
    assert len({ax.get_xlim() for ax in axes}) == 1 and len({ax.get_ylim() for ax in axes}) == 1
    assert viz.plot_q16_triples(q["triples"][1:], q["layer_names"], tmp_path / "u.png", title="x") is None


def test_plot_q16_occ_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    rows = [("x / y / z", {"books": [(rng.normal(0.5, 0.2, 40), rng.uniform(0, 1, 40))] * 2,
                           "pile": [(rng.normal(0.4, 0.3, 60), rng.uniform(0, 1, 60))] * 2})]
    assert viz.plot_q16_occ(rows, ["L0 (embed)", "L1"], tmp_path / "o.png", title="x").exists()


def _degenerate_q16():
    # frame 0 degenerate (two words share a last piece): no verdict, NaNs written as None
    q = _fake_q16()
    e = q["triples"][0]["groups"]["books"]
    e["null_pct"][0], e["swap_pct"] = None, [[None, None, None]] + e["swap_pct"][1:]
    e["beats_null"][0], e["best_of_three"][0] = None, None
    e["t"][0] = e["d"][0] = e["seg"][0] = None
    e["td"] = [[[None, None], [0.5, 0.3], [None, None]]] + e["td"][1:]
    e["null_td"] = [[[None, None], [None, None]]] + e["null_td"][1:]
    return q


def test_plot_q16_summary_cross_hatches_degenerate_cells(tmp_path):
    fig = viz.plot_q16_summary(_degenerate_q16(), tmp_path / "s.png", return_fig=True)
    hatches = [p.get_hatch() for ax in fig.axes for p in ax.patches]
    assert "\\\\\\\\" in hatches and "degenerate" in fig._suptitle.get_text()
    books = fig.axes[0]
    dots = np.concatenate([c.get_offsets() for c in books.collections])
    assert 0 not in dots[:, 0]  # best_of_three None at frame 0 -> no dot there


def test_plot_q16_triples_survives_degenerate_frames(tmp_path):
    q = _degenerate_q16()
    assert viz.plot_q16_triples(q["triples"][:1], q["layer_names"], tmp_path / "t.png", title="x").exists()


def test_plot_q16_occ_survives_an_all_nan_frame(tmp_path):
    # degenerate frame: the two vice points coincide, so every occurrence's t and d is NaN
    rng = np.random.default_rng(0)
    nan = np.full(40, np.nan)
    rows = [("x / y / z", {"books": [(nan, nan), (rng.normal(0.5, 0.2, 40), rng.uniform(0, 1, 40))]})]
    assert viz.plot_q16_occ(rows, ["L0 (embed)", "L1"], tmp_path / "o.png", title="x").exists()


def _fake_timeline(rng, with_gaps=True):
    steps = [0, 1, 8, 64]
    names = ["L0 (embed)", "L1", "L2", "L3", "L4 (pre-LN)", "L4 (post-LN)", "unembed"]  # a 4-layer toy
    S, F, B = len(steps), len(names), 6
    curves = {k: rng.uniform(0, 1, size=(S, F)).tolist() for k in
              ("knn_purity", "knn_purity_shuffled", "silhouette", "silhouette_shuffled", "anisotropy", "top_pc_share")}
    curves["knn_consecutive"] = rng.uniform(0, 1, size=(S, F - 1)).tolist()
    drift = {w: rng.uniform(1e-4, 1, size=(S, B)).tolist() for w in ("embed", "unembed")}
    if with_gaps:  # what the JSON round trip gives: None where a value was NaN
        curves["silhouette"][0][2] = None
        drift["embed"][0] = [0.0] * B
        drift["embed"][1][3] = None
    tl_frames, tl_ids = ["L0 (embed)", "L2", "unembed"], [0, 2, 6]
    return {
        "revisions": [f"step{s}" for s in steps], "steps": steps, "final": "step64", "layer_names": names,
        "timeline_frames": tl_frames, "timeline_frame_ids": tl_ids, "curves": curves,
        "with_final": {n: {"knn": rng.uniform(0, 1, S).tolist(), "cka": rng.uniform(0, 1, S).tolist()} for n in tl_frames},
        "drift": drift, "n_freq_bins": B,
    }


def test_plot_timeline_six_panels_and_survives_nones(tmp_path, rng):
    fig = viz.plot_timeline(_fake_timeline(rng), tmp_path / "t.png", return_fig=True)
    assert (tmp_path / "t.png").exists()
    assert len(fig.axes) == 6
    ax_cons = fig.axes[2]
    # 4-layer toy: the middle block is transitions L1->L2 and L2->L3
    assert [l.get_label() for l in ax_cons.get_lines()] == ["L1 -> L2", "L2 -> L3"]


def test_plot_drift_one_line_per_matrix_and_bin(tmp_path, rng):
    t = _fake_timeline(rng)
    fig = viz.plot_drift(t, tmp_path / "d.png", return_fig=True)
    assert (tmp_path / "d.png").exists()
    labels = [l.get_label() for l in fig.axes[0].get_lines()]
    assert sum(lab.startswith("embed:") for lab in labels) == t["n_freq_bins"]
    assert sum(lab.startswith("unembed:") for lab in labels) == t["n_freq_bins"]
