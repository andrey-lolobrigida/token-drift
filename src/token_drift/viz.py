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
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

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


def _shared_limits(coords: np.ndarray):
    # Same limits for every frame, otherwise the eye reads a zoom as motion.
    lo, hi = coords.min(axis=(0, 1)), coords.max(axis=(0, 1))
    pad = 0.03 * (hi - lo)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])


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
    png_names: list[str] | None = None,
    gif_name: str = "flipbook.gif",
    coords_name: str = "umap_coords.npy",
) -> list[Path]:
    """One PNG per layer with shared axis limits, stitched into flipbook.gif.

    Also saves the raw coords (umap_coords.npy) and a trajectory plot for a few
    hand-picked tokens if `trajectories` (token string -> row index) is given.
    `png_names` / `gif_name` / `coords_name` let several flipbooks share one folder (v2: one
    per timeline frame, pages = checkpoints instead of layers).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if png_names is not None and len(png_names) != len(layer_names):
        raise ValueError(f"png_names has {len(png_names)} entries for {len(layer_names)} pages")
    png_names = png_names or [f"umap_layer_{i}.png" for i in range(len(layer_names))]
    coords = np.asarray(coords)
    labels = np.asarray(labels)
    np.save(out_dir / coords_name, coords)

    xlim, ylim = _shared_limits(coords)

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
        p = out_dir / png_names[i]
        fig.savefig(p, dpi=110)
        plt.close(fig)
        written.append(p)

    frames = [iio.imread(p) for p in written]
    gif = out_dir / gif_name
    iio.imwrite(gif, frames, duration=frame_seconds * 1000, loop=0)
    written.append(gif)

    if trajectories:
        written.append(plot_trajectories(
            coords, labels, layer_names, {repr(t): i for t, i in trajectories.items()},
            out_dir / "trajectories.png",
        ))
    return written


_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _trajectory_styles(n: int) -> list[tuple]:
    """One (colour, marker) per path. The words live in a colour legend, so colours must
    be told apart at a glance: RUN_COLORS for a handful, then tab10's 10 qualitative hues,
    with the marker shape changing every 10. (A continuous ramp like turbo gives 28 hues
    where neighbours are indistinguishable, which is fine for labelled lines, not for a key.)"""
    if n <= len(RUN_COLORS):
        return [(c, "o") for c in RUN_COLORS[:n]]
    tab = plt.get_cmap("tab10").colors
    return [(tab[j % 10], _MARKERS[(j // 10) % len(_MARKERS)]) for j in range(n)]


def _style_legend(leg):
    leg.get_frame().set_edgecolor(_GRID)
    for t in leg.get_texts():
        t.set_color(_INK)


def _word_legend(fig, texts: list[str], styles: list[tuple]):
    """The words, keyed by colour+marker, in a column to the right of the plot(s)."""
    handles = [Line2D([], [], color=c, marker=m, linestyle="-", linewidth=1.2, markersize=6)
               for c, m in styles]
    ncol = 1 if len(texts) <= 20 else 2  # 28 rows at 7pt would run off the bottom
    leg = fig.legend(handles, texts, loc="center left", bbox_to_anchor=(1.0, 0.5),
                     fontsize=7, frameon=True, ncol=ncol)
    _style_legend(leg)
    return leg


def plot_trajectories(
    coords: np.ndarray, labels: np.ndarray, layer_names: list[str], trajectories: dict[str, int],
    out_path: str | Path, *, title: str | None = None, return_fig: bool = False,
):
    """Paths of a few tokens across frames, on top of a faint last-layer scatter.

    `trajectories` maps the legend text for each path (e.g. "' vice' n=812") to its
    row in `coords`.
    """
    xlim, ylim = _shared_limits(coords)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(coords[-1, :, 0], coords[-1, :, 1], s=1, alpha=0.08, color=_MUTED, linewidths=0)
    L = coords.shape[0]
    alphas = np.linspace(0.25, 1.0, L)  # fade in: early layers faint, last layer solid
    styles = _trajectory_styles(len(trajectories))
    for (color, marker), idx in zip(styles, trajectories.values()):
        path = coords[:, idx, :]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.2, alpha=0.7)
        ax.scatter(path[:, 0], path[:, 1], s=18, color=color, marker=marker, alpha=alphas, linewidths=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    _style_axes(ax)
    ax.grid(False)
    head = f"{title}: " if title else ""
    ax.set_title(f"{head}token trajectories, {layer_names[0]} -> {layer_names[-1]} (faint = early)",
                 loc="left", fontsize=11, color=_INK)
    fig.tight_layout()
    _word_legend(fig, list(trajectories), styles)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")  # tight: the legend sits outside the figure box
    if return_fig:
        return fig
    plt.close(fig)
    return out_path


def plot_group_frames(
    coords: np.ndarray, layer_names: list[str], group: dict[str, int], out_path: str | Path,
    *, title: str | None = None, return_fig: bool = False,
):
    """The comic-strip version of plot_trajectories: one small panel per frame, each zoomed
    onto where the group sits in that frame, no paths. Words are keyed by the same
    colour+marker legend as plot_trajectories.

    Each panel has its own window (the group drifts across the map, a shared window would
    zoom back out). AlignedUMAP keeps frames roughly aligned, but read positions *within* a
    panel, not across panels. `group` maps legend text to its row in `coords`.
    """
    n = coords.shape[0]
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    styles = _trajectory_styles(len(group))
    rows = list(group.values())
    for f in range(n):
        ax = axes[f]
        pts = coords[f, rows]
        core = _core_mask(pts)
        lo, hi = pts[core].min(axis=0), pts[core].max(axis=0)
        # floor on the span: a group that collapsed to one point would zoom to nothing
        span = max((hi - lo).max(), 0.05 * np.ptp(coords[f], axis=0).max())
        pad = 0.15 * span
        xlim, ylim = (lo[0] - pad, hi[0] + pad), (lo[1] - pad, hi[1] + pad)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.scatter(coords[f, :, 0], coords[f, :, 1], s=3, alpha=0.15, color=_MUTED, linewidths=0)
        # stragglers get pinned just inside the edge they're off of, hollow:
        # "it's out there, that way", without letting them set the zoom
        inset = 0.03 * span
        for j, r in enumerate(rows):
            color, marker = styles[j]
            xy = coords[f, r]
            if core[j]:
                ax.scatter(*xy, s=26, color=color, marker=marker, linewidths=0, zorder=3)
            else:
                xy = np.clip(xy, [xlim[0] + inset, ylim[0] + inset], [xlim[1] - inset, ylim[1] - inset])
                ax.scatter(*xy, s=26, marker=marker, facecolors="none", edgecolors=color,
                           linewidths=1, zorder=3)
        ax.set_xticks([])
        ax.set_yticks([])
        _style_axes(ax)
        ax.grid(False)
        ax.set_title(layer_names[f], loc="left", fontsize=9, color=_INK)
    for ax in axes[n:]:
        ax.axis("off")
    head = f"{title}: " if title else ""
    fig.suptitle(f"{head}each frame zoomed onto the group (compare positions within a panel; "
                 "hollow = off-panel, pinned to the edge)", x=0.01, ha="left", fontsize=11, color=_INK)
    fig.tight_layout()
    _word_legend(fig, list(group), styles)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    if return_fig:
        return fig
    plt.close(fig)
    return out_path


def _core_mask(pts: np.ndarray, k: float = 2.5) -> np.ndarray:
    """Which points form the group's core: within k x the median distance to the group's
    median point. Robust to a couple of far stragglers, which would otherwise set the zoom
    (in the virtue group, ' good' and ' hope' squashed the other 15 words into a corner)."""
    if len(pts) < 5:  # a median of 3 distances isn't robust to anything; keep them all
        return np.ones(len(pts), bool)
    d = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
    med = np.median(d)
    return d <= k * med if med > 0 else np.ones(len(pts), bool)


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
    # .get(..., "acts"): older metrics.json predates this field and was always vocab-run acts
    aniso_sources = {m.get("anisotropy_source", "acts") for m in runs.values() if m.get("anisotropy") is not None}
    if aniso_sources and aniso_sources <= {"unit_mean", "raw_mean"}:
        # corpus runs' "raw" acts are already per-token means (unit_mean or raw_mean), not
        # occurrence-level vectors - Ethayarajh's actual number lives in the self-sim panel.
        aniso_title = ("anisotropy: mean cos between per-token MEANS, uncentered\n"
                       "(not Ethayarajh's occurrence-level number: see self-sim baseline)\n"
                       "/ top-PC share after centering")
    else:
        aniso_title = "anisotropy: mean cos of RAW acts (Ethayarajh 2019)\n/ top-PC share after centering"
    ax_aniso.set_title(aniso_title, loc="left", fontsize=10)
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


# ---------- Q16 (milestone B) ----------

# Sequential blue from the dataviz reference ramp, dark -> light: the interesting end (low null
# percentile = the virtue sits between its vices) is the dark, salient one.
_SEQ_BLUE = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
_PLACEMENTS = [  # (legend text, colour, filled) for word 0 / 1 / 2 of the triple in the middle
    ("deficiency in the middle", RUN_COLORS[1], False),
    ("virtue (mean) in the middle", RUN_COLORS[0], True),
    ("excess in the middle", RUN_COLORS[2], False),
]
_GROUP_COLORS = {"books": RUN_COLORS[0], "pile": RUN_COLORS[1]}


def _triple_label(t: dict) -> str:
    return f"{t['set']} {t['concept']} {t['pos']}: {' / '.join(t['words'])}"


def _save(fig, out_path, return_fig):
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")  # tight: legends sit outside the axes
    if return_fig:
        return fig
    plt.close(fig)
    return out_path


def plot_q16_summary(q16: dict, out_path, *, return_fig: bool = False):
    """One heatmap per source group: rows = triples, columns = frames, colour = the virtue's
    null percentile. Dot = virtue beats both role swaps; hatched = a word below min_count;
    cross-hatched cell = degenerate frame (two words share a point, so no verdict)."""
    rows, names, groups = q16["triples"], q16["layer_names"], q16["groups"]
    cmap = LinearSegmentedColormap.from_list("q16", _SEQ_BLUE)
    fig, axes = plt.subplots(1, len(groups), squeeze=False, sharey=True, layout="constrained",
                             figsize=(3.8 + 0.45 * len(names) * len(groups), 1.6 + 0.26 * len(rows)))
    im = None
    for ax, g in zip(axes[0], groups):
        M = np.full((len(rows), len(names)), np.nan)
        for i, t in enumerate(rows):
            if "missing" not in t["groups"][g]:
                M[i] = np.array(t["groups"][g]["null_pct"], dtype=float)  # None (degenerate) -> NaN
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
        for i, t in enumerate(rows):
            e = t["groups"][g]
            if "missing" in e:
                ax.add_patch(Rectangle((-0.5, i - 0.5), len(names), 1, facecolor="none",
                                       edgecolor=_MUTED, hatch="////", linewidth=0))
                continue
            for f, pct in enumerate(e["null_pct"]):
                if pct is None:  # degenerate: same blank as missing, other hatch so they don't read alike
                    ax.add_patch(Rectangle((f - 0.5, i - 0.5), 1, 1, facecolor="none",
                                           edgecolor=_MUTED, hatch="\\\\\\\\", linewidth=0))
            best = np.flatnonzero([b is True for b in e["best_of_three"]])  # None = no verdict = no dot
            ax.scatter(best, np.full(len(best), i), s=14, color="white", edgecolors=_INK, linewidths=0.6, zorder=3)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.tick_params(colors=_MUTED, labelsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(g, loc="left", fontsize=10, color=_INK)
    axes[0][0].set_yticks(range(len(rows)), [_triple_label(t) for t in rows])
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.6)
    cb.set_label("virtue's null percentile (0 = beats every null word)", fontsize=8, color=_INK)
    # two lines: one long line ran into the colorbar label
    fig.suptitle("Q16: does the virtue sit between its vices?\n"
                 "dot = virtue beats both role swaps, hatched = a word below min_count,\n"
                 "cross-hatched = degenerate: two words share a point",
                 x=0.01, ha="left", fontsize=9, color=_INK)
    return _save(fig, out_path, return_fig)


def plot_q16_triples(entries: list[dict], layer_names: list[str], out_path, *, title: str,
                     return_fig: bool = False):
    """The exact (t, d) plane per frame: t = position along the vice-vice line (0, 1 = the two
    ends), d = distance off it, both in vice-vice units. Three points always span a plane, so
    unlike a UMAP nothing here is a projection artefact. Rows = triple x group, columns = frames."""
    rows = [(t, g) for t in entries for g, e in t["groups"].items() if "missing" not in e]
    if not rows:
        return None
    F = len(layer_names)
    # shared axes on every panel: same flipbook rule as the UMAPs, a zoom must not read as motion
    fig, axes = plt.subplots(len(rows), F, squeeze=False, sharex=True, sharey=True,
                             figsize=(1.7 * F + 1.5, 1.6 * len(rows) + 0.8))
    for r, (t, g) in enumerate(rows):
        e = t["groups"][g]
        for f in range(F):
            ax = axes[r, f]
            _style_axes(ax)
            for x in (0, 1):
                ax.axvline(x, color=_MUTED, linewidth=0.8, linestyle="--")
            nt = np.asarray(e["null_td"][f], dtype=float)  # None (degenerate frame) -> NaN, not drawn
            ax.scatter(nt[:, 0], nt[:, 1], s=8, color=_MUTED, alpha=0.5, linewidths=0)
            for (_, col, filled), (tt, dd) in zip(_PLACEMENTS, np.asarray(e["td"][f], dtype=float)):
                ax.scatter([tt], [dd], s=36, facecolors=col if filled else "none", edgecolors=col,
                           linewidths=1.4, zorder=3)
            if r == 0:
                ax.set_title(layer_names[f], fontsize=8, color=_INK)
        # words stacked one per line: the slash version was wider than the panel is tall
        axes[r, 0].set_ylabel("\n".join(t["words"]) + f"\n[{g}]\nd", fontsize=7, color=_INK)
    axes[0, 0].set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel("t", fontsize=8, color=_MUTED)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=c if fl else "none",
                      markeredgecolor=c) for _, c, fl in _PLACEMENTS]
    handles.append(Line2D([], [], marker="o", linestyle="", markersize=4, color=_MUTED, alpha=0.5))
    fig.suptitle(f"{title}: d (off the line) vs t (along it)", x=0.01, ha="left", fontsize=10, color=_INK)
    fig.tight_layout()
    leg = fig.legend(handles, [p[0] for p in _PLACEMENTS] + ["null words (virtue swapped out)"],
                     loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    _style_legend(leg)
    return _save(fig, out_path, return_fig)


def plot_q16_occ(rows, frame_names: list[str], out_path, *, title: str, return_fig: bool = False):
    """Per-occurrence t and d of the virtue against its two vice points, books vs Pile overlaid.
    A mean point can sit between the vices while its occurrences are two clumps; this shows it.
    rows = [(triple label, {group: [(t array, d array) per frame]})]."""
    F = len(frame_names)
    fig, axes = plt.subplots(len(rows), 2 * F, squeeze=False, figsize=(3.6 * F + 1.5, 1.6 * len(rows) + 0.8))
    for r, (label, per_group) in enumerate(rows):
        for f in range(F):
            for j, stat in enumerate(("t", "d")):
                ax = axes[r, 2 * f + j]
                _style_axes(ax)
                vals = np.concatenate([per_group[g][f][j] for g in per_group])
                vals = vals[np.isfinite(vals)]  # NaN = degenerate frame (the two vice points coincide)
                if len(vals):
                    bins = np.linspace(vals.min(), vals.max() + 1e-9, 30)
                    for g in per_group:
                        x = np.asarray(per_group[g][f][j])
                        ax.hist(x[np.isfinite(x)], bins=bins, density=True, histtype="step",
                                color=_GROUP_COLORS[g], linewidth=1.4)
                else:
                    ax.text(0.5, 0.5, "degenerate", transform=ax.transAxes, ha="center", va="center",
                            fontsize=7, color=_MUTED)
                if stat == "t":
                    for x in (0, 1):
                        ax.axvline(x, color=_MUTED, linewidth=0.8, linestyle="--")
                ax.set_yticks([])
                ax.ticklabel_format(axis="x", useOffset=False, style="plain")  # no 1e-9+0.8 offset text
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))  # near-constant d (tiny runs) otherwise piles up tick labels
                if r == 0:
                    ax.set_title(f"{frame_names[f]}: {stat}", fontsize=8, color=_INK)
        axes[r, 0].set_ylabel(label.replace(" / ", "\n"), fontsize=7, color=_INK)
    fig.suptitle(f"{title}: per-occurrence position of the virtue", x=0.01, ha="left", fontsize=10, color=_INK)
    fig.tight_layout()
    groups = [g for g in _GROUP_COLORS if any(g in pg for _, pg in rows)]
    leg = fig.legend([Line2D([], [], color=_GROUP_COLORS[g], linewidth=1.4) for g in groups], groups,
                     loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    _style_legend(leg)
    return _save(fig, out_path, return_fig)


# ---------- v2 timeline (training checkpoints) ----------

F16_RESOLUTION = 2.0**-11  # float16 relative spacing: Pythia's checkpoints are stored in fp16


def _as_float(rows) -> np.ndarray:
    """JSON lists (None where a NaN was) -> float array with NaN; matplotlib skips NaN points."""
    return np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)


def _step_axis(ax, steps: list[int]):
    # symlog = linear in [-1, 1] and log beyond, so step 0 gets a spot at the left edge
    # (plain log can't show 0) and 1 .. 143000 still spread out evenly per decade
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(left=-0.5)  # steps start at 0; autoscale would leave ~4 empty decades on the left
    ax.set_xticks(steps, [str(s) for s in steps], fontsize=7, rotation=45, ha="right")
    ax.minorticks_off()
    ax.set_xlabel("training step", fontsize=8, color=_MUTED)


def plot_timeline(tl: dict, out_path, *, return_fig: bool = False):
    """v2 panels A + B against training step.

    The timeline frames (config timeline.frames) get a colour each and a legend entry; every
    other frame is a thin grey context line, so the eye lands on the frames the spec asks about.
    """
    steps, names = tl["steps"], tl["layer_names"]
    hi = dict(zip(tl["timeline_frame_ids"], RUN_COLORS))
    c = {k: _as_float(v) for k, v in tl["curves"].items()}
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    (ax_pur, ax_sil), (ax_cons, ax_aniso), (ax_knn, ax_cka) = axes

    def per_frame(ax, y, ls="-", labelled=True):
        for f in range(y.shape[1]):
            if f in hi:
                ax.plot(steps, y[:, f], ls, color=hi[f], linewidth=2, marker="o", markersize=4,
                        label=names[f] if labelled else None)
            else:
                ax.plot(steps, y[:, f], ls, color=_MUTED, linewidth=0.8, alpha=0.5)

    per_frame(ax_pur, c["knn_purity"] - c["knn_purity_shuffled"])
    per_frame(ax_sil, c["silhouette"])
    per_frame(ax_aniso, c["anisotropy"])
    per_frame(ax_aniso, c["top_pc_share"], ls=":", labelled=False)  # same colours, dotted: the panel's 2nd series
    n_layers = len(names) - 3  # frames = embed, blocks 1..n-1, block n pre-LN, post-LN, unembed
    block = list(range(1, n_layers - 1))  # transitions L1->L2 .. L(n-2)->L(n-1): v0's stable middle block
    ramp = _run_ramp(RUN_COLORS[2], max(len(block), 1))  # depth is ordered, so one hue light -> dark
    for j, i in enumerate(block):
        ax_cons.plot(steps, c["knn_consecutive"][:, i], color=ramp[j], linewidth=1.8, marker="o", markersize=4,
                     label=f"{names[i]} -> {names[i + 1]}")
    for name, f in zip(tl["timeline_frames"], tl["timeline_frame_ids"]):
        wf = tl["with_final"][name]
        kw = dict(color=hi[f], linewidth=2, marker="o", markersize=4)
        ax_knn.plot(steps, _as_float([wf["knn"]])[0], **kw)
        ax_cka.plot(steps, _as_float([wf["cka"]])[0], **kw)

    ax_pur.set_title("A. kNN category purity minus shuffled-label purity", loc="left", fontsize=10)
    ax_sil.set_title("A. silhouette on surface-form categories", loc="left", fontsize=10)
    ax_cons.set_title("A. kNN overlap, consecutive layers, middle block", loc="left", fontsize=10)
    ax_aniso.set_title("A. anisotropy (solid: mean cos of raw acts, dotted: top-PC share)", loc="left", fontsize=10)
    ax_knn.set_title(f"B. kNN overlap with {tl['final']} (local)", loc="left", fontsize=10)
    ax_cka.set_title(f"B. linear CKA with {tl['final']} (global)", loc="left", fontsize=10)
    for ax in axes.flat:
        _style_axes(ax)
        _step_axis(ax, steps)
    for ax in (ax_cons, ax_aniso, ax_knn, ax_cka):
        ax.set_ylim(0, 1.02)
    ax_pur.set_ylim(-0.05, 1.02)  # a frame with no surface-form structure sits at ~0, maybe a hair below
    if block:
        ax_cons.legend(frameon=False, fontsize=7, loc="lower right")
    handles, labels = ax_pur.get_legend_handles_labels()
    handles.append(Line2D([], [], color=_MUTED, linewidth=0.8, alpha=0.5))
    labels.append("other frames")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels),
               frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))  # leave the bottom strip for the legend
    return _save(fig, out_path, return_fig)


def plot_drift(tl: dict, out_path, *, return_fig: bool = False):
    """v2 panel C: median row drift from init per merge-rank bin. Embed solid, unembed dashed.

    The prediction (spec): unembed rows move early and together (the softmax pushes every row at
    every position); embed rows only move when their token is in the batch, so rare bins lag.
    """
    steps, nb = tl["steps"], tl["n_freq_bins"]
    ramp = _run_ramp(RUN_COLORS[0], nb)  # a bin is a magnitude (rarity), so one hue light -> dark
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for w, ls, mk in (("embed", "-", "o"), ("unembed", "--", "s")):
        d = _as_float(tl["drift"][w])  # (steps, bins)
        for b in range(nb):
            lab = "base/byte" if b == 0 else f"bin {b}" + (
                " (most frequent)" if b == 1 else " (rarest)" if b == nb - 1 else "")
            ax.plot(steps, d[:, b], ls, color=ramp[b], linewidth=1.6, marker=mk, markersize=4, label=f"{w}: {lab}")
    ax.axhline(F16_RESOLUTION, color=_MUTED, linewidth=0.8, linestyle=":")
    ax.text(steps[-1], F16_RESOLUTION, "float16 resolution", fontsize=7, color=_MUTED, ha="right", va="bottom")
    _step_axis(ax, steps)
    # symlog y as well: step 0 is exactly 0 by definition, early drift is ~1e-4, late drift ~1
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylim(bottom=0)  # drift is a norm ratio, never negative; autoscale shows ~3 empty decades below 0
    ax.set_ylabel("median ||W_t[i] - W_0[i]|| / ||W_0[i]||", fontsize=8, color=_MUTED)
    ax.set_title("C. row drift from init by merge-rank bin (embed solid, unembed dashed)", loc="left", fontsize=10)
    _style_axes(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7)
    return _save(fig, out_path, return_fig)
