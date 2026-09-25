"""`token-drift extract|normalize|metrics|viz|all --config configs/x.yaml`, plus `compare`.

This is the only module that touches files from another stage. Everything else is
arrays in, arrays out.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import typer
import yaml

from token_drift import extract as ex
from token_drift import metrics as mt
from token_drift import viz
from token_drift.labels import categorize_all, freq_bins
from token_drift.normalize import normalize_all

N_FREQ_BINS = 5  # quantiles of merge rank; bin 0 is reserved for base/byte tokens

app = typer.Typer(add_completion=False, help="Watch a small LM's vocab geometry drift.")


# ---------- config / paths ----------

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def run_dir(cfg: dict) -> Path:
    return Path(cfg["out_dir"]) / cfg["run_name"]


def stage_dir(rd: Path, stage: str) -> Path:
    """runs/<run>/<stage>/ - one subfolder per pipeline stage so a run dir isn't a pile."""
    d = rd / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prepare_run_dir(cfg: dict) -> Path:
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    # Every run is reproducible from its yaml + seed, so the yaml travels with the outputs.
    src = Path(cfg["_config_path"])
    if src.exists():
        shutil.copy(src, rd / "config.yaml")
    return rd


def layer_names(n_layers: int) -> list[str]:
    """Human labels for the n_layers+3 frames: embedding, blocks, unembed.

    The last block shows up twice: the raw residual (grabbed by a hook on the final
    LayerNorm) and HF's hidden_states[-1], which is *after* that LN. Labelled so nobody
    has to remember which is which.
    """
    names = ["L0 (embed)"] + [f"L{i}" for i in range(1, n_layers)]
    names += [f"L{n_layers} (pre-LN)", f"L{n_layers} (post-LN)"]
    return names + ["unembed"]


def _load_json(p: Path):
    return json.loads(p.read_text())


# ---------- stages ----------

def stage_extract(cfg: dict) -> Path:
    rd = _prepare_run_dir(cfg)
    device = ex.pick_device(cfg["device"])
    typer.echo(f"[extract] {cfg['model']} random_init={cfg['random_init']} on {device}")
    t0 = time.time()
    model, tok = ex.build_model(
        cfg["model"], random_init=cfg["random_init"], seed=cfg["seed"], device=device
    )
    tokens = ex.vocab_tokens(tok)
    ids = np.arange(len(tokens))
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    acts = ex.extract_activations(
        model, ids, bos_id=bos_id, batch_size=cfg["extract"]["batch_size"], device=device
    )
    embed, unembed = ex.get_embed_unembed(model)
    ln_gain, ln_bias = ex.get_final_ln(model)
    n_layers = model.config.num_hidden_layers
    del model  # free it: everything downstream is numpy
    out = stage_dir(rd, "extract")
    # embed/unembed matrices are padded past the tokenizer's vocab; keep only real rows
    np.save(out / "acts.npy", acts)
    np.save(out / "embed.npy", embed[: len(tokens)])
    np.save(out / "unembed.npy", unembed[: len(tokens)])
    np.savez(out / "final_ln.npz", gain=ln_gain, bias=ln_bias)
    np.save(out / "labels.npy", categorize_all(tokens, special=set(tok.all_special_tokens)))
    ranks = ex.vocab_freq_ranks(tok)
    np.save(out / "freq_ranks.npy", ranks)
    np.save(out / "freq_bins.npy", freq_bins(ranks, n_bins=N_FREQ_BINS))
    (out / "tokens.json").write_text(json.dumps(tokens))
    (out / "layer_names.json").write_text(json.dumps(layer_names(n_layers)))
    typer.echo(f"[extract] acts {acts.shape} in {time.time() - t0:.0f}s -> {out}")
    return rd


def stage_normalize(cfg: dict) -> Path:
    rd = _prepare_run_dir(cfg)
    ex = rd / "extract"
    acts = np.load(ex / "acts.npy", mmap_mode="r")  # float16 on disk; upcast per layer
    unembed = np.load(ex / "unembed.npy")
    # The unembedding matrix rides along as pseudo-layer L+1 from here on. Same rows
    # (tokens), and it's the cleanest "identity vs prediction" comparison we have.
    stack = np.concatenate([acts, unembed[None]], axis=0)
    n = cfg["normalize"]
    row_first = n.get("row_norm_first", False)  # configs from before 2026-09-24 don't have it
    norm = normalize_all(
        stack, center=n["center"], unit_norm=n["unit_norm"], drop_top_pcs=n["drop_top_pcs"],
        row_norm_first=row_first,
    )
    np.save(stage_dir(rd, "normalize") / "acts_norm.npy", norm)
    typer.echo(f"[normalize] {norm.shape} center={n['center']} drop_top_pcs={n['drop_top_pcs']} row_norm_first={row_first}")
    return rd


def stage_metrics(cfg: dict) -> Path:
    rd = _prepare_run_dir(cfg)
    ex_dir = rd / "extract"
    norm = np.load(rd / "normalize" / "acts_norm.npy", mmap_mode="r")
    labels = np.load(ex_dir / "labels.npy")
    names = _load_json(ex_dir / "layer_names.json")
    # raw (uncentered) frames for the anisotropy curve; same order as the normalized stack
    raw_acts = np.load(ex_dir / "acts.npy", mmap_mode="r")
    raw = [raw_acts[i] for i in range(raw_acts.shape[0])] + [np.load(ex_dir / "unembed.npy")]
    fb_path = ex_dir / "freq_bins.npy"
    fb = np.load(fb_path) if fb_path.exists() else None  # runs extracted before this existed
    m = cfg["metrics"]
    t0 = time.time()
    result = mt.compute_all(
        [norm[i] for i in range(norm.shape[0])], names, labels,
        knn_k=m["knn_k"], kmeans_k=m["kmeans_k"], seed=cfg["seed"], subsample=m["subsample"],
        raw_layers=raw, freq_bins=fb, n_freq_bins=(int(fb.max()) + 1) if fb is not None else N_FREQ_BINS + 1,
    )
    out = stage_dir(rd, "metrics")
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    viz.plot_metrics({cfg["run_name"]: result}, out / "metrics.png")
    viz.plot_cka_heatmap(result, out / "cka.png")
    typer.echo(f"[metrics] knn_consecutive={np.round(result['knn_consecutive'], 3).tolist()}")
    typer.echo(f"[metrics] knn_purity={np.round(result['knn_purity'], 3).tolist()}")
    typer.echo(f"[metrics] anisotropy={np.round(result['anisotropy'], 3).tolist()}")
    typer.echo(f"[metrics] top_pc_share={np.round(result['top_pc_share'], 3).tolist()} ({time.time() - t0:.0f}s)")
    return rd


def stage_viz(cfg: dict) -> Path:
    rd = _prepare_run_dir(cfg)
    norm = np.load(rd / "normalize" / "acts_norm.npy", mmap_mode="r")
    labels = np.load(rd / "extract" / "labels.npy")
    names = _load_json(rd / "extract" / "layer_names.json")
    tokens = _load_json(rd / "extract" / "tokens.json")
    v = cfg["viz"]
    out = stage_dir(rd, "viz")

    # AlignedUMAP on 50k x 8 frames takes a long time; reuse the metrics subsample and
    # force the hand-picked trajectory tokens into it so they're always drawn.
    metrics = _load_json(rd / "metrics" / "metrics.json")
    idx = set(metrics["subsample_idx"])
    tok_to_row = {t: i for i, t in enumerate(tokens)}
    traj: dict[str, int] = {}
    for t in v.get("trajectory_tokens", []):
        if t in tok_to_row:
            traj[t] = tok_to_row[t]
            idx.add(tok_to_row[t])
        else:
            typer.echo(f"[viz] trajectory token {t!r} not in vocab, skipping")
    idx = np.array(sorted(idx))
    np.save(out / "viz_idx.npy", idx)
    pos = {row: k for k, row in enumerate(idx)}
    traj = {t: pos[r] for t, r in traj.items()}

    t0 = time.time()
    layers = [np.asarray(norm[i][idx], dtype=np.float32) for i in range(norm.shape[0])]
    coords = viz.project_layers(
        layers, method=v["method"], n_neighbors=v["n_neighbors"], min_dist=v["min_dist"],
        seed=cfg["seed"],
    )
    typer.echo(f"[viz] {v['method']} on {len(idx)} tokens x {len(layers)} frames in {time.time() - t0:.0f}s")
    viz.plot_flipbook(coords, labels[idx], names, out, trajectories=traj)
    typer.echo(f"[viz] wrote flipbook -> {out / 'flipbook.gif'}")
    return rd


def compare_runs(run_dirs: list[Path], out: Path) -> Path:
    runs = {Path(rd).name: _load_json(Path(rd) / "metrics" / "metrics.json") for rd in run_dirs}
    return viz.plot_metrics(runs, out)


def run_ksweep(cfgs: list[dict], ks: list[int], out: Path) -> Path:
    """Q10: re-run the kNN metrics at several k on each run's metrics subsample."""
    sweeps = {}
    for cfg in cfgs:
        rd = run_dir(cfg)
        norm = np.load(rd / "normalize" / "acts_norm.npy", mmap_mode="r")
        idx = np.array(_load_json(rd / "metrics" / "metrics.json")["subsample_idx"])
        t0 = time.time()
        s = mt.knn_sweep(
            [norm[i] for i in range(norm.shape[0])], _load_json(rd / "extract" / "layer_names.json"),
            np.load(rd / "extract" / "labels.npy"), ks=ks, idx=idx, seed=cfg["seed"],
        )
        (stage_dir(rd, "metrics") / "ksweep.json").write_text(json.dumps(s, indent=1))
        typer.echo(f"[ksweep] {cfg['run_name']} k={s['ks']} ({time.time() - t0:.0f}s)")
        sweeps[cfg["run_name"]] = s
    return viz.plot_ksweep(sweeps, out)


# ---------- typer commands ----------

_CONFIG = typer.Option(..., "--config", "-c", help="path to a yaml config")


@app.command()
def extract(config: Path = _CONFIG):
    stage_extract(load_config(config))


@app.command()
def normalize(config: Path = _CONFIG):
    stage_normalize(load_config(config))


@app.command()
def metrics(config: Path = _CONFIG):
    stage_metrics(load_config(config))


@app.command()
def viz_cmd(config: Path = _CONFIG):
    stage_viz(load_config(config))


# typer names commands after the function; we want `viz`, not `viz-cmd`
app.registered_commands[-1].name = "viz"


@app.command()
def all(config: Path = _CONFIG):  # noqa: A001 - it's the CLI verb we documented
    cfg = load_config(config)
    stage_extract(cfg)
    stage_normalize(cfg)
    stage_metrics(cfg)
    stage_viz(cfg)


@app.command()
def compare(
    runs: list[Path] = typer.Argument(..., help="run dirs, e.g. runs/pythia70m runs/random_init"),
    out: Path = typer.Option(Path("runs/compare.png"), "--out", "-o"),
):
    """Overlay the metric curves of several runs on the same axes."""
    p = compare_runs(runs, out)
    typer.echo(f"[compare] -> {p}")


@app.command()
def ksweep(
    config: list[Path] = typer.Option(..., "--config", "-c", help="one or more yaml configs"),
    ks: str = typer.Option("5,10,30,100", "--ks", help="comma-separated neighbourhood sizes"),
    out: Path = typer.Option(Path("runs/ksweep.png"), "--out", "-o"),
):
    """Q10: kNN overlap and purity at several k, one row per run. Needs metrics already run."""
    p = run_ksweep([load_config(c) for c in config], [int(k) for k in ks.split(",")], out)
    typer.echo(f"[ksweep] -> {p}")


if __name__ == "__main__":
    app()
