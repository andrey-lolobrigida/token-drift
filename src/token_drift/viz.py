"""Aligned UMAP flipbook and metric plots.

Never compare coordinates across separately fitted UMAPs. AlignedUMAP fits all frames
jointly with an identity relation between consecutive layers (same token index), so a
point moving between frames means the token moved, not that the projection rotated.
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")  # headless; we only ever write files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from token_drift.labels import CATEGORIES  # noqa: E402

# Categorical hues in fixed order (never cycled), from the dataviz reference palette.
# The two "leftover" buckets get ink/gray so they recede instead of stealing a hue.
CATEGORY_COLORS = {
    "special": "#0b0b0b",
    "byte": "#e34948",
    "nonascii": "#4a3aa7",
    "digit": "#e87ba4",
    "punct": "#008300",
    "space_cap": "#2a78d6",
    "space_lower": "#eb6834",
    "cap": "#1baf7a",
    "lower": "#eda100",
    "mixed": "#898781",
}
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # one per run, fixed order
_INK, _MUTED, _GRID = "#0b0b0b", "#898781", "#e1e0d9"


def project_layers(
    layers: list[np.ndarray], *, method: str, n_neighbors: int, min_dist: float, seed: int
) -> np.ndarray:
    """list of (n, d) -> (L, n, 2) coordinates in one shared 2-d space."""
    import umap  # slow import (numba JIT); keep it out of module load

    xs = [np.asarray(x, dtype=np.float32) for x in layers]
    n = xs[0].shape[0]
    if method == "aligned_umap":
        # relation i maps row j of frame i to row j of frame i+1: same token, next layer.
        relations = [{j: j for j in range(n)} for _ in range(len(xs) - 1)]
        # alignment_regularisation: pulls same-token points together across frames; the
        # default (1e-2) is a mild tether. alignment_window_size=2 = only look at
        # neighbouring frames, which is what a flipbook needs.
        reducer = umap.AlignedUMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=seed,
            alignment_window_size=2,
        )
        reducer.fit(xs, relations=relations)
        return np.stack([np.asarray(e) for e in reducer.embeddings_], axis=0)
    if method == "stacked_umap":
        # Fallback: one UMAP over all layers stacked, then split. Shared space by
        # construction; frames are less crisp because the fit has to serve every layer.
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=seed
        )
        emb = reducer.fit_transform(np.concatenate(xs, axis=0))
        return emb.reshape(len(xs), n, 2)
    raise ValueError(f"unknown viz method {method!r}; use aligned_umap or stacked_umap")


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(True, color=_GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def plot_flipbook(
    coords: np.ndarray,
    labels: np.ndarray,
    layer_names: list[str],
    out_dir: str | Path,
    *,
    trajectories: dict[str, int] | None = None,
    frame_seconds: float = 0.8,
) -> list[Path]:
    """One PNG per layer with shared axis limits, stitched into flipbook.gif.

    Also saves the raw coords (umap_coords.npy) and a trajectory plot for a few
    hand-picked tokens if `trajectories` (token string -> row index) is given.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(coords)
    labels = np.asarray(labels)
    np.save(out_dir / "umap_coords.npy", coords)

    # Same limits for every frame, otherwise the eye reads a zoom as motion.
    lo, hi = coords.min(axis=(0, 1)), coords.max(axis=(0, 1))
    pad = 0.03 * (hi - lo)
    xlim, ylim = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])

    # Draw big categories first so small ones (digits, punct) sit on top and stay visible.
    order = np.argsort(-np.bincount(labels, minlength=len(CATEGORIES)))
    written: list[Path] = []
    for i, name in enumerate(layer_names):
        fig, ax = plt.subplots(figsize=(8, 6.5))
        for c in order:
            mask = labels == c
            if not mask.any():
                continue
            ax.scatter(
                coords[i, mask, 0], coords[i, mask, 1],
                s=2, alpha=0.6, linewidths=0, color=CATEGORY_COLORS[CATEGORIES[c]],
                label=f"{CATEGORIES[c]} ({mask.sum()})",
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"{name}", color=_INK, fontsize=12, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        _style_axes(ax)
        ax.grid(False)
        leg = ax.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8,
            markerscale=5, title="category", title_fontsize=8,
        )
        for t in leg.get_texts():
            t.set_color(_INK)
        fig.tight_layout()
        p = out_dir / f"umap_layer_{i}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        written.append(p)

    frames = [iio.imread(p) for p in written]
    gif = out_dir / "flipbook.gif"
    iio.imwrite(gif, frames, duration=frame_seconds * 1000, loop=0)
    written.append(gif)

    if trajectories:
        written.append(_plot_trajectories(coords, labels, layer_names, trajectories, xlim, ylim, out_dir))
    return written


def _plot_trajectories(coords, labels, layer_names, trajectories, xlim, ylim, out_dir) -> Path:
    """Paths of a few tokens across frames, on top of a faint last-layer scatter."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(coords[-1, :, 0], coords[-1, :, 1], s=1, alpha=0.08, color=_MUTED, linewidths=0)
    L = coords.shape[0]
    alphas = np.linspace(0.25, 1.0, L)  # fade in: early layers faint, last layer solid
    for j, (tok, idx) in enumerate(trajectories.items()):
        color = RUN_COLORS[j % len(RUN_COLORS)] if j < len(RUN_COLORS) else CATEGORY_COLORS[CATEGORIES[labels[idx]]]
        path = coords[:, idx, :]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.2, alpha=0.7)
        for i in range(L):
            ax.scatter(path[i, 0], path[i, 1], s=18, color=color, alpha=alphas[i], linewidths=0)
        ax.annotate(repr(tok), path[-1], fontsize=7, color=_INK, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    _style_axes(ax)
    ax.grid(False)
    ax.set_title(f"token trajectories, {layer_names[0]} -> {layer_names[-1]} (faint = early)", loc="left", fontsize=11, color=_INK)
    fig.tight_layout()
    p = Path(out_dir) / "trajectories.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def _run_ramp(color: str, n: int) -> list[tuple[float, float, float]]:
    """n shades of one run's hue, light -> dark. Colour follows the run (its identity);
    lightness carries the bin (a magnitude). Hue and saturation stay fixed."""
    h, _, s = colorsys.rgb_to_hls(*to_rgb(color))
    return [colorsys.hls_to_rgb(h, lum, s) for lum in np.linspace(0.80, 0.25, n)]


def _transition_labels(names: list[str], *, one_line: bool = False) -> list[str]:
    # tilted two-line labels run into their neighbours; tilted single lines stay parallel
    sep = " -> " if one_line else "\n->"
    return [f"{a}{sep}{b}" for a, b in zip(names[:-1], names[1:])]


def plot_metrics(runs: dict[str, dict], out_path: str | Path, *, return_fig: bool = False):
    """Curves over layers. `runs` maps run name -> metrics.json dict; all on shared axes.

    Six panels: the four drift/purity curves, plus anisotropy (Ethayarajh) and
    change-by-frequency (Voita) when the metrics carry them (older metrics.json may not).
    A 4th row (self-similarity + adjusted self-similarity, Ethayarajh) shows up only when
    at least one run is a corpus-mode v1 run with self_sim in its metrics.json.
    """
    has_ss = any(m.get("self_sim") is not None for m in runs.values())
    fig, axes = plt.subplots(4 if has_ss else 3, 2, figsize=(12, 16 if has_ss else 12))
    (ax_cons, ax_drift), (ax_sil, ax_ari), (ax_aniso, ax_freq) = axes[:3]
    ax_ss, ax_ssa = axes[3] if has_ss else (None, None)
    styles = ["-", "--", ":", "-."]
    for r, (run, m) in enumerate(runs.items()):
        names = m["layer_names"]
        color, ls = RUN_COLORS[r % len(RUN_COLORS)], styles[r % len(styles)]
        kw = dict(color=color, linewidth=2, marker="o", markersize=5)
        x_t = np.arange(len(names) - 1)
        ax_cons.plot(x_t, m["knn_consecutive"], ls, label=f"{run}: kNN overlap", **kw)
        ax_ari.plot(x_t, m["kmeans_ari_consecutive"], ls, label=f"{run}: k-means ARI", **kw)
        x_l = np.arange(len(names))
        ax_drift.plot(x_l, m["knn_vs_first"], ls, label=f"{run}: vs {names[0]}", **kw)
        ax_drift.plot(x_l, m["knn_vs_last"], ls, label=f"{run}: vs {names[-1]}", **{**kw, "marker": "s", "alpha": 0.55})
        # purity, not silhouette: silhouette is ~0 everywhere in 512-d even when local
        # neighborhoods are clearly organized by category. silhouette stays in the json.
        ax_sil.plot(x_l, m["knn_purity"], ls, label=f"{run}: kNN category purity", **kw)
        ax_sil.plot(x_l, m["knn_purity_shuffled"], ls, label=f"{run}: shuffled labels", **{**kw, "marker": "x", "alpha": 0.55})
        if m.get("anisotropy") is not None:
            ax_aniso.plot(x_l, m["anisotropy"], ls, label=f"{run}: mean cos, random pairs (raw)", **kw)
        if m.get("top_pc_share") is not None:
            # same colour + linestyle as the run's mean-cos line, triangle marker + faded: the
            # panel's other series, like "vs last" in the drift panel. Not a separate dash style,
            # because "--" already means "second run".
            ax_aniso.plot(x_l, m["top_pc_share"], ls, label=f"{run}: top-PC variance share (centered)",
                          **{**kw, "marker": "^", "alpha": 0.55})
        if m.get("knn_change_by_freq") is not None:
            # one line per frequency bin, single hue light->dark: bin is a magnitude (rank), not an identity
            by_freq = np.asarray(m["knn_change_by_freq"], dtype=float)  # (transitions, bins)
            nb = by_freq.shape[1]
            ramp = _run_ramp(color, nb)
            corpus_bins = m.get("freq_bins_source") == "corpus"
            for b in range(nb):
                if corpus_bins:  # 0 = most frequent ... nb-1 = rarest, no base/byte bin
                    lab = f"corpus bin {b}" + (" (most frequent)" if b == 0 else " (rarest)" if b == nb - 1 else "")
                    show = b in (0, nb - 1)
                else:
                    lab = "base/byte" if b == 0 else f"freq bin {b}" + (" (most frequent)" if b == 1 else " (rarest)" if b == nb - 1 else "")
                    show = b in (0, 1, nb - 1)
                ax_freq.plot(x_t, by_freq[:, b], ls, color=ramp[b], linewidth=1.5, marker="o", markersize=3.5,
                             label=f"{run}: {lab}" if show else None)
        if m.get("self_sim") is not None:
            x_s = np.arange(len(m["self_sim"]))  # extract frames only: the unembed has no occurrences
            ax_ss.plot(x_s, m["self_sim"], ls, label=f"{run}: self-similarity", **kw)
            # same colour + linestyle, faded triangle-down: "--" already means "second run" here
            ax_ss.plot(x_s, m["self_sim_baseline"], ls, label=f"{run}: baseline (any two occurrences)",
                       **{**kw, "marker": "v", "alpha": 0.55})
            ax_ssa.plot(x_s, m["self_sim_adjusted"], ls, label=f"{run}: adjusted self-sim", **kw)
    first = next(iter(runs.values()))
    names = first["layer_names"]
    # 8 short frames fit upright; with the "(pre-LN)"/"(post-LN)" frames even Pythia's 9
    # collide, so tilt from 9 on
    rot = dict(rotation=45, ha="right") if len(names) > 8 else {}
    for ax in (ax_cons, ax_ari, ax_freq):
        ax.set_xticks(np.arange(len(names) - 1), _transition_labels(names, one_line=bool(rot)), fontsize=7, **rot)
    for ax in (ax_drift, ax_sil, ax_aniso):
        ax.set_xticks(np.arange(len(names)), names, fontsize=7, **rot)
    ax_cons.set_title("kNN overlap, consecutive layers (higher = less reorganization)", loc="left", fontsize=10)
    ax_drift.set_title("kNN overlap vs first layer and vs unembed", loc="left", fontsize=10)
    ax_sil.set_title("kNN purity of surface-form categories (frac. of neighbors with same label)", loc="left", fontsize=10)
    ax_ari.set_title("k-means ARI, consecutive layers", loc="left", fontsize=10)
    ax_aniso.set_title("anisotropy: mean cos of RAW acts (Ethayarajh 2019) / top-PC share after centering", loc="left", fontsize=10)
    sources = {m.get("freq_bins_source", "merge_rank") for m in runs.values() if m.get("knn_change_by_freq") is not None}
    if sources == {"corpus"}:
        freq_title = ("neighborhood change by corpus-count bin (Voita et al. 2019)\n"
                      "light -> dark: bin 0 (most frequent) ... bin 4 (rarest)")
    else:
        freq_title = ("neighborhood change by token frequency bin (Voita et al. 2019)\n"
                      "light -> dark: base/byte, bin 1 (most frequent) ... bin 5 (rarest)")
    ax_freq.set_title(freq_title, loc="left", fontsize=10)
    if has_ss:
        for ax in (ax_ss, ax_ssa):
            ax.set_xticks(np.arange(len(names) - 1), names[:-1], fontsize=7, **rot)
        ax_ss.set_title("self-similarity: mean cos between a token's occurrences (Ethayarajh 2019)",
                        loc="left", fontsize=10)
        ax_ssa.set_title("adjusted self-similarity = self-sim - baseline", loc="left", fontsize=10)
    for ax in axes.flat:
        _style_axes(ax)
        ax.set_ylim(0, 1.02)
    if has_ss:
        ax_ssa.set_ylim(-0.2, 1.02)  # adjusted self-sim dips below 0 when occurrences are less alike than random pairs
    # One legend for the whole figure, under the grid. Per-panel loc="best" kept landing
    # on data lines. Row 1: which run (colour + dash). Row 2: what the marker shape means,
    # in neutral grey because it's the same for every run.
    run_handles = [
        Line2D([], [], color=RUN_COLORS[r % len(RUN_COLORS)], linestyle=styles[r % len(styles)], linewidth=2)
        for r in range(len(runs))
    ]
    fig.legend(run_handles, list(runs), loc="lower center", bbox_to_anchor=(0.5, 0.025),
               ncol=len(runs), frameon=False, fontsize=9)
    shape_key = [("o", "main series"), ("s", "vs unembed (top right)"),
                 ("x", "shuffled-label baseline (purity)"), ("^", "top-PC variance share (anisotropy)")]
    if has_ss:
        shape_key.append(("v", "self-sim baseline"))
    shape_handles = [Line2D([], [], color=_MUTED, marker=mk, linestyle="none", markersize=6) for mk, _ in shape_key]
    fig.legend(shape_handles, [t for _, t in shape_key], loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(shape_key), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))  # leave the bottom strip for the legends
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120)
    if return_fig:
        return fig
    plt.close(fig)
    return out_path


def plot_ksweep(sweeps: dict[str, dict], out_path: str | Path) -> Path:
    """Q10: the k-dependent curves at several k. One row per run, one line per k.

    k is a magnitude, so it gets the run's own hue light (small k) -> dark (big k),
    like the frequency panel. What to look for is whether the *shape* moves with k,
    not the level: bigger k gives bigger overlaps almost by construction.
    """
    fig, axes = plt.subplots(len(sweeps), 3, figsize=(15, 3.6 * len(sweeps)), squeeze=False)
    for r, (run, s) in enumerate(sweeps.items()):
        names, ks = s["layer_names"], s["ks"]
        ramp = _run_ramp(RUN_COLORS[r % len(RUN_COLORS)], len(ks))
        ax_cons, ax_last, ax_pur = axes[r]
        x_t, x_l = np.arange(len(names) - 1), np.arange(len(names))
        for j, k in enumerate(ks):
            m = s["by_k"][str(k)]
            kw = dict(color=ramp[j], linewidth=1.8, marker="o", markersize=4, label=f"k={k}")
            ax_cons.plot(x_t, m["knn_consecutive"], **kw)
            ax_last.plot(x_l, m["knn_vs_last"], **kw)
            ax_pur.plot(x_l, m["knn_purity"], **kw)
            ax_pur.plot(x_l, m["knn_purity_shuffled"], color=ramp[j], linewidth=1, marker="x",
                        markersize=4, alpha=0.55)
        rot = dict(rotation=45, ha="right")
        ax_cons.set_xticks(x_t, _transition_labels(names, one_line=True), fontsize=7, **rot)
        for ax in (ax_last, ax_pur):
            ax.set_xticks(x_l, names, fontsize=7, **rot)
        ax_cons.set_title(f"{run}: kNN overlap, consecutive layers", loc="left", fontsize=10)
        ax_last.set_title(f"{run}: kNN overlap vs {names[-1]}", loc="left", fontsize=10)
        ax_pur.set_title(f"{run}: category purity (x = shuffled labels)", loc="left", fontsize=10)
        ax_cons.legend(frameon=False, fontsize=8, loc="lower left")
        for ax in axes[r]:
            _style_axes(ax)
            ax.set_ylim(0, 1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_cka_heatmap(metrics: dict, out_path: str | Path) -> Path:
    """Linear CKA between every pair of layers. Block structure = phases."""
    names = metrics["layer_names"]
    cka = np.asarray(metrics["cka"])
    fig, ax = plt.subplots(figsize=(6, 5))
    # single-hue sequential ramp: CKA is a magnitude in [0, 1]
    im = ax.imshow(cka, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cka[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if cka[i, j] > 0.6 else _INK)
    fig.colorbar(im, ax=ax, fraction=0.046, label="linear CKA")
    ax.set_title("layer x layer linear CKA", loc="left", fontsize=10)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_cross_overlap(cross: list[float], layer_names: list[str], out_path: str | Path, *, title: str) -> Path:
    """One line: v0-vs-v1 kNN overlap at each frame, same tokens."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(layer_names))
    ax.plot(x, cross, color=RUN_COLORS[0], linewidth=2, marker="o", markersize=5)
    ax.set_xticks(x, layer_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, loc="left", fontsize=10)
    _style_axes(ax)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
