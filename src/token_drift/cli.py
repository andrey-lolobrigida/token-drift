"""`token-drift corpus|extract|normalize|metrics|viz|all --config configs/x.yaml`, plus `compare`, `ksweep`, `v0v1`.

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

from token_drift import corpus as cp
from token_drift import extract as ex
from token_drift import metrics as mt
from token_drift import viz
from token_drift.labels import categorize_all, corpus_freq_bins, freq_bins
from token_drift.normalize import normalize_all, normalize_layer

N_FREQ_BINS = 5  # quantiles of merge rank; bin 0 is reserved for base/byte tokens
N_CORPUS_BINS = 5  # equal-count bins of corpus count over eligible tokens; 0 = most frequent


def corpus_eligibility(counts: np.ndarray, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """(eligible mask, corpus freq bins). An average over 3 contexts mostly says which 3
    sentences they were, hence min_count; zero-count rows are never eligible."""
    eligible = counts >= max(min_count, 1)
    return eligible, corpus_freq_bins(counts, eligible, n_bins=N_CORPUS_BINS)

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

SHUFFLE_MODES = ("none", "window")


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    # we tokenize whole docs and window them ourselves; stop HF warning that docs are long
    tok.model_max_length = 10**9
    return tok


def stage_corpus(cfg: dict) -> Path:
    rd = _prepare_run_dir(cfg)
    c = cfg["corpus"]
    if c["shuffle"] not in SHUFFLE_MODES:
        raise ValueError(f"corpus.shuffle must be one of {SHUFFLE_MODES}, got {c['shuffle']!r}")
    tok = _load_tokenizer(cfg["model"])
    budget = c.get("max_tokens")
    docs: list[list[int]] = []
    sets: list[str | None] = []
    n_read = 0
    t0 = time.time()
    for path in cp.iter_parquet(c["source"]):
        texts, names = cp.read_parquet(path, c["text_field"])
        ids = tok(texts, add_special_tokens=False)["input_ids"]
        for i, d in enumerate(ids):
            docs.append(d)
            sets.append(names[i] if names is not None else None)
            n_read += len(d) + 1  # +1 for the EOS pack() appends
            if budget is not None and n_read >= budget:
                break
        if budget is not None and n_read >= budget:
            break  # don't even download the next shard
    windows = cp.pack(docs, eos_id=tok.eos_token_id, window=c["window"], max_tokens=budget)
    if windows.shape[0] == 0:
        raise ValueError(f"corpus has {n_read} tokens, less than one {c['window']}-token window")
    if c["shuffle"] == "window":
        windows = cp.shuffle_within_windows(windows, seed=cfg["seed"])
    out = stage_dir(rd, "corpus")
    np.save(out / "windows.npy", windows)
    meta = {
        "source": c["source"], "text_field": c["text_field"], "window": c["window"],
        "max_tokens": budget, "shuffle": c["shuffle"], "seed": cfg["seed"],
        "n_docs": len(docs), "n_tokens_read": min(n_read, budget or n_read),
        "n_tokens": int(windows.size), "n_windows": int(windows.shape[0]),
        "source_mix": cp.source_mix(sets),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    typer.echo(f"[corpus] {meta['n_docs']} docs -> {windows.shape} ({time.time() - t0:.0f}s) -> {out}")
    return rd


def stage_extract(cfg: dict) -> Path:
    rd = _prepare_run_dir(cfg)
    mode = cfg["extract"].get("mode", "vocab")  # v0 configs don't have the key
    if mode not in ("vocab", "corpus"):
        raise ValueError(f"extract.mode must be vocab or corpus, got {mode!r}")
    device = ex.pick_device(cfg["device"])
    typer.echo(f"[extract] {cfg['model']} random_init={cfg['random_init']} mode={mode} on {device}")
    t0 = time.time()
    model, tok = ex.build_model(
        cfg["model"], random_init=cfg["random_init"], seed=cfg["seed"], device=device
    )
    tokens = ex.vocab_tokens(tok)
    V = len(tokens)
    out = stage_dir(rd, "extract")
    if mode == "vocab":
        ids = np.arange(V)
        bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        acts = ex.extract_activations(
            model, ids, bos_id=bos_id, batch_size=cfg["extract"]["batch_size"], device=device
        )
    else:
        windows = np.load(rd / "corpus" / "windows.npy")
        unit_mean, raw_mean, counts, self_sim, baseline = ex.extract_corpus_means(
            model, windows, eos_id=tok.eos_token_id, min_context=cfg["extract"]["min_context"],
            batch_size=cfg["extract"]["batch_size"], device=device,
        )
        # the model's rows are padded past the tokenizer (50,304 vs 50,277); padding ids never occur
        acts = unit_mean[:, :V]
        np.save(out / "acts_rawmean.npy", raw_mean[:, :V])
        np.save(out / "counts.npy", counts[:V])
        np.save(out / "self_sim.npy", self_sim[:, :V])
        (out / "self_sim_baseline.json").write_text(json.dumps(baseline))
        typer.echo(f"[extract] {len(windows)} windows, {int(counts.sum())} occurrences, "
                   f"{int((counts[:V] > 0).sum())}/{V} tokens seen")
    embed, unembed = ex.get_embed_unembed(model)
    ln_gain, ln_bias = ex.get_final_ln(model)
    n_layers = model.config.num_hidden_layers
    del model  # free it: everything downstream is numpy
    # embed/unembed matrices are padded past the tokenizer's vocab; keep only real rows
    np.save(out / "acts.npy", acts)
    np.save(out / "embed.npy", embed[:V])
    np.save(out / "unembed.npy", unembed[:V])
    np.savez(out / "final_ln.npz", gain=ln_gain, bias=ln_bias)
    np.save(out / "labels.npy", categorize_all(tokens, special=set(tok.all_special_tokens)))
    ranks = ex.vocab_freq_ranks(tok)
    np.save(out / "freq_ranks.npy", ranks)
    np.save(out / "freq_bins.npy", freq_bins(ranks, n_bins=N_FREQ_BINS))
    (out / "tokens.json").write_text(json.dumps(tokens))
    (out / "layer_names.json").write_text(json.dumps(layer_names(n_layers)))
    typer.echo(f"[extract] acts {acts.shape} in {time.time() - t0:.0f}s -> {out}")
    return rd


NORMALIZE_SOURCES = {"unit_mean": "acts.npy", "raw_mean": "acts_rawmean.npy"}


def stage_normalize(cfg: dict) -> Path:
    rd = _prepare_run_dir(cfg)
    ex_dir = rd / "extract"
    n = cfg["normalize"]
    source = n.get("source", "unit_mean")  # ignored by vocab runs: they only have acts.npy
    if source not in NORMALIZE_SOURCES:
        raise ValueError(f"normalize.source must be one of {list(NORMALIZE_SOURCES)}, got {source!r}")
    acts_path = ex_dir / NORMALIZE_SOURCES[source]
    if not acts_path.exists():
        raise ValueError(f"normalize.source={source} needs extract/{acts_path.name}, which only "
                         "corpus-mode runs write (extract.mode: corpus)")
    acts = np.load(acts_path, mmap_mode="r")  # float16 on disk; upcast per layer
    counts_path = ex_dir / "counts.npy"
    # corpus runs: never-seen tokens are zero rows; keep them out of the mean (see normalize.py)
    fit = np.load(counts_path) > 0 if counts_path.exists() else None
    row_first = n.get("row_norm_first", False)  # configs from before 2026-09-24 don't have it
    kw = dict(center=n["center"], unit_norm=n["unit_norm"], drop_top_pcs=n["drop_top_pcs"],
              row_norm_first=row_first)
    norm_acts = normalize_all(acts, fit_rows=fit, **kw)
    # The unembedding matrix rides along as pseudo-layer L+1 from here on. Same rows
    # (tokens), and it's the cleanest "identity vs prediction" comparison we have. Always
    # centered on all rows, like v0, so v0 and v1 unembed frames are identical.
    norm_unembed = normalize_layer(np.load(ex_dir / "unembed.npy"), **kw).astype(np.float16)
    norm = np.concatenate([norm_acts, norm_unembed[None]], axis=0)
    np.save(stage_dir(rd, "normalize") / "acts_norm.npy", norm)
    typer.echo(f"[normalize] {norm.shape} source={source} center={n['center']} "
               f"drop_top_pcs={n['drop_top_pcs']} row_norm_first={row_first} "
               f"fit_rows={'all' if fit is None else int(fit.sum())}")
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
    counts_path = ex_dir / "counts.npy"
    if counts_path.exists():  # corpus-mode run
        eligible, cbins = corpus_eligibility(np.load(counts_path), m.get("min_count", 1))
        extra = dict(
            eligible=eligible, freq_bins=cbins, n_freq_bins=N_CORPUS_BINS, freq_bins_source="corpus",
            merge_rank_bins=fb, n_merge_rank_bins=N_FREQ_BINS + 1,
            self_sim=np.load(ex_dir / "self_sim.npy", mmap_mode="r"),
            self_sim_baseline=_load_json(ex_dir / "self_sim_baseline.json"),
        )
    else:
        extra = dict(freq_bins=fb, n_freq_bins=(int(fb.max()) + 1) if fb is not None else N_FREQ_BINS + 1)
    result = mt.compute_all(
        [norm[i] for i in range(norm.shape[0])], names, labels,
        knn_k=m["knn_k"], kmeans_k=m["kmeans_k"], seed=cfg["seed"], subsample=m["subsample"],
        raw_layers=raw, **extra,
    )
    out = stage_dir(rd, "metrics")
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    viz.plot_metrics({cfg["run_name"]: result}, out / "metrics.png")
    viz.plot_cka_heatmap(result, out / "cka.png")
    typer.echo(f"[metrics] knn_consecutive={np.round(result['knn_consecutive'], 3).tolist()}")
    typer.echo(f"[metrics] knn_purity={np.round(result['knn_purity'], 3).tolist()}")
    typer.echo(f"[metrics] anisotropy={np.round(result['anisotropy'], 3).tolist()}")
    typer.echo(f"[metrics] top_pc_share={np.round(result['top_pc_share'], 3).tolist()} ({time.time() - t0:.0f}s)")
    if result["self_sim"] is not None:
        typer.echo(f"[metrics] eligible={result['eligible_n']} self_sim_adjusted={np.round(result['self_sim_adjusted'], 3).tolist()}")
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


@app.command("corpus")
def corpus_cmd(config: Path = _CONFIG):
    """Download + pack the background corpus into windows (v1 runs only)."""
    stage_corpus(load_config(config))


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
