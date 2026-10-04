"""`token-drift corpus|probe_corpus|extract|normalize|metrics|viz|occ|all --config configs/x.yaml`, plus `compare`, `ksweep`, `v0v1`.

This is the only module that touches files from another stage. Everything else is
arrays in, arrays out.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer
import yaml

from token_drift import corpus as cp
from token_drift import extract as ex
from token_drift import metrics as mt
from token_drift import probe as pb
from token_drift import viz
from token_drift.labels import categorize_all, corpus_freq_bins, freq_bins
from token_drift.normalize import normalize_all, normalize_layer

N_FREQ_BINS = 5  # quantiles of merge rank; bin 0 is reserved for base/byte tokens
N_CORPUS_BINS = 5  # equal-count bins of corpus count over eligible tokens; 0 = most frequent
GROUPS = pb.GROUPS


def corpus_eligibility(counts: np.ndarray, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """(eligible mask, corpus freq bins). An average over 3 contexts mostly says which 3
    sentences they were, hence min_count; zero-count rows are never eligible."""
    eligible = counts >= max(min_count, 1)
    return eligible, corpus_freq_bins(counts, eligible, n_bins=N_CORPUS_BINS)

app = typer.Typer(add_completion=False, help="Watch a small LM's vocab geometry drift.")


# ---------- config / paths ----------

def _mode(cfg: dict) -> str:
    return (cfg.get("extract") or {}).get("mode", "vocab")  # v0 configs don't have the key


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    if "corpus" in cfg and "probe" in cfg:
        raise ValueError(f"{path}: has both corpus: (milestone A) and probe: (milestone B); pick one")
    if ("probe" in cfg) != (_mode(cfg) == "probe"):
        raise ValueError(f"{path}: a probe: block needs extract.mode: probe, and vice versa")
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


def _raw_layers(ex_dir: Path, acts_filename: str = "acts.npy") -> list[np.ndarray]:
    """Load raw (uncentered) frames for anisotropy; same order as normalized stack.

    acts_filename: which extract-stage array counts as "raw" here. Vocab runs (and v0)
    only ever have acts.npy; corpus runs can point this at acts_rawmean.npy instead, to
    match whatever normalize.source is actually feeding the rest of the pipeline.
    """
    raw_acts = np.load(ex_dir / acts_filename, mmap_mode="r")
    return [raw_acts[i] for i in range(raw_acts.shape[0])] + [np.load(ex_dir / "unembed.npy")]


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


def _model_max_context(name: str) -> int:
    from transformers import AutoConfig

    return int(AutoConfig.from_pretrained(name).max_position_embeddings)


def _tokenize_texts(tok, texts: list[str], chunk: int = 1000) -> list[np.ndarray]:
    """uint16 ids per text, `chunk` texts per tokenizer call. One call on a whole Pile shard
    hands back ~120M Python ints (~4 GB) before we get to shrink them; this box has 15 GB."""
    out: list[np.ndarray] = []
    for i in range(0, len(texts), chunk):
        out += [pb.to_uint16(d) for d in tok(texts[i : i + chunk], add_special_tokens=False)["input_ids"]]
    return out


def _load_books(p: dict, tok, out: Path) -> tuple[list[np.ndarray], list[str], dict]:
    """Each book = one document. Warns if a cached download differs from the run's last one.
    Returns the books meta; the caller writes books.json with the other outputs (a run that
    dies midway must not clobber the previous run's copy, or the next warning is lost)."""
    docs, names, meta = [], [], {}
    for b in p["books"]:
        raw = pb.fetch_gutenberg(b["gutenberg"], p["cache_dir"])
        try:
            body = pb.gutenberg_body(raw, b.get("start"), b.get("end"))
        except ValueError as e:
            raise ValueError(f"book {b['name']} (gutenberg {b['gutenberg']}): {e}") from e
        ids = _tokenize_texts(tok, [pb.unwrap(body)])[0]
        docs.append(ids)
        names.append(b["name"])
        meta[b["name"]] = {"gutenberg": b["gutenberg"], "start": b.get("start"), "end": b.get("end"),
                           "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(), "n_tokens": int(len(ids))}
    prev = out / "books.json"
    if prev.exists():
        old = _load_json(prev)
        for name, m in meta.items():
            if name in old and old[name]["sha256"] != m["sha256"]:
                typer.echo(f"[probe_corpus] WARNING: {name} (gutenberg {m['gutenberg']}) differs from "
                           "this run's previous download; counts will shift")
    return docs, names, meta


def _load_pile(p: dict, tok) -> tuple[list[np.ndarray], list[str]]:
    docs, sources = [], []
    shards = itertools.islice(cp.iter_parquet(p["pile"]["source"]), p["pile"]["shards"])
    for s, path in enumerate(shards):
        t0 = time.time()
        texts, _ = cp.read_parquet(path, p["pile"]["text_field"])
        docs += _tokenize_texts(tok, texts)
        sources += [f"pile:{s}"] * len(texts)
        typer.echo(f"[probe_corpus] pile shard {s}: {len(texts)} docs ({time.time() - t0:.0f}s)")
    return docs, sources


def stage_probe_corpus(cfg: dict) -> Path:
    """Milestone B: hits of every probe / polysemy / null word -> one window ending at each."""
    rd = _prepare_run_dir(cfg)
    p, m = cfg["probe"], cfg["metrics"]
    out = stage_dir(rd, "probe_corpus")
    t0 = time.time()
    words = pb.load_words(p["words"])
    tok = _load_tokenizer(cfg["model"])
    eos = tok.eos_token_id
    max_ctx = p.get("max_context") or _model_max_context(cfg["model"])
    probe_pos = pb.probe_word_pos(words)
    probe_list = list(probe_pos)
    poly = [w for w in words["polysemy"] if w not in probe_pos]
    # tokenize the words before any download, so a bad word fails in a second, not after the Pile
    wid = pb.word_token_ids(tok, probe_list + poly, eos_id=eos)
    cap_space = pb.word_token_ids(tok, [w.capitalize() for w in probe_list], eos_id=eos)
    cap_bare = pb.word_token_ids(tok, [w.capitalize() for w in probe_list], eos_id=eos, prefix="")

    book_docs, book_names, books_meta = _load_books(p, tok, out)
    pile_docs, pile_sources = _load_pile(p, tok)
    corpora = {"books": (*pb.concat_docs(book_docs), book_names),
               "pile": (*pb.concat_docs(pile_docs), pile_sources)}
    del book_docs, pile_docs
    ok_next = pb.boundary_table(tok)
    typer.echo(f"[probe_corpus] tokens: books {len(corpora['books'][0])}, pile {len(corpora['pile'][0])}")

    # pass 1: whole-word counts of every single-token word -> the null pool
    single = {g: pb.count_single_token_words(c[0], c[1], ok_next, len(tok)) for g, c in corpora.items()}
    book_hits = pb.find_hits(corpora["books"][0], corpora["books"][1], [wid[w] for w in probe_list], ok_next)[0]
    book_found = np.bincount(book_hits, minlength=len(probe_list))
    exclude = set(probe_list) | set(words["polysemy"]) | set(words["null_exclude"])
    cands = pb.null_candidates(ex.vocab_tokens(tok), single["books"], single["pile"],
                               min_count=m["min_count"], exclude=exclude)
    targets = {pos: [int(book_found[i]) for i, w in enumerate(probe_list) if probe_pos[w] == pos] for pos in pb.POS}
    pool = pb.pick_null_pool(cands, targets, n=m["null_pool"], seed=cfg["seed"])
    for pos, ws in pool.items():
        if len(ws) < m["null_k"]:
            bc = sorted(cands[pos].values())
            raise ValueError(f"only {len(ws)} {pos} null-pool candidates (>= {m['min_count']} hits in both "
                             f"groups; book counts {bc[:1]}..{bc[-1:]}), need null_k={m['null_k']}")
    null_words = [w for pos in pb.POS for w in pool[pos]]
    wid.update(pb.word_token_ids(tok, null_words, eos_id=eos))

    # pass 2: hits -> min_context -> cap -> windows
    all_words = probe_list + poly + null_words
    pats = [wid[w] for w in all_words]
    cap_pats = [cap_space[w.capitalize()] for w in probe_list] + [cap_bare[w.capitalize()] for w in probe_list]
    cap_owner = np.tile(np.arange(len(probe_list)), 2)  # cap pattern -> index into all_words
    tags = pb.role_tags(words, pool)
    counts: dict[str, dict] = {w: {} for w in all_words}
    windows: list[np.ndarray] = []
    meta: dict[str, list] = {k: [] for k in ("word", "role_tags", "group", "source", "doc", "position", "n_pieces")}
    for g_i, g in enumerate(GROUPS):
        ids, offs, srcs = corpora[g]
        pat, doc, end = pb.find_hits(ids, offs, pats, ok_next)
        ctx = end - offs[doc]  # tokens before the last piece, in its own document
        ok = ctx >= p["min_context"]
        sel = np.flatnonzero(ok)[pb.reservoir(pat[ok], p["cap"], seed=cfg["seed"] + g_i)]
        capn = np.bincount(cap_owner[pb.find_hits(ids, offs, cap_pats, ok_next)[0]], minlength=len(all_words))
        found, usable = np.bincount(pat, minlength=len(all_words)), np.bincount(pat[ok], minlength=len(all_words))
        kept = np.bincount(pat[sel], minlength=len(all_words))
        for i, w in enumerate(all_words):
            counts[w][g] = {"found": int(found[i]), "context_ok": int(usable[i]), "kept": int(kept[i]),
                            "capitalized": int(capn[i])}
        windows += pb.cut_windows(ids, offs, doc[sel], end[sel], max_context=max_ctx, eos_id=eos)
        meta["word"] += [all_words[i] for i in pat[sel]]
        meta["role_tags"] += [tags[all_words[i]] for i in pat[sel]]
        meta["group"] += [g] * len(sel)
        meta["source"] += [srcs[d] for d in doc[sel]]
        meta["doc"] += doc[sel].tolist()
        meta["position"] += ctx[sel].tolist()
        meta["n_pieces"] += [len(pats[i]) for i in pat[sel]]
    meta["snippet"] = pb.snippets(tok, windows)
    wt, wo = pb.concat_docs(windows)
    np.save(out / "windows_tokens.npy", wt)
    np.save(out / "windows_offsets.npy", wo)
    pq.write_table(pa.table(meta), out / "occ_meta.parquet")
    (out / "books.json").write_text(json.dumps(books_meta, indent=1))
    (out / "counts.json").write_text(json.dumps(counts, indent=1))
    (out / "null_pool.json").write_text(json.dumps(
        {pos: [{"word": w, "books": counts[w]["books"]["found"], "pile": counts[w]["pile"]["found"]}
               for w in pool[pos]] for pos in pb.POS}, indent=1))
    (out / "count_report.md").write_text(pb.count_report(words, counts, pool, min_count=m["min_count"]))
    # metrics reads this copy: the list these occurrences were cut for, whatever the yaml says later.
    # Copied last so a run that dies midway never pairs a new list with the old run's occurrences
    shutil.copy(p["words"], out / "probe_words.yaml")
    (out / "meta.json").write_text(json.dumps({
        "max_context": int(max_ctx), "n_windows": len(windows), "n_window_tokens": int(len(wt)),
        "n_tokens": {g: int(len(corpora[g][0])) for g in GROUPS}, "seconds": round(time.time() - t0),
    }, indent=1))
    typer.echo(f"[probe_corpus] {len(windows)} windows, {len(wt)} tokens ({time.time() - t0:.0f}s) -> {out}")
    return rd


def stage_extract(cfg: dict) -> Path:
    rd = _prepare_run_dir(cfg)
    mode = _mode(cfg)
    if mode not in ("vocab", "corpus", "probe"):
        raise ValueError(f"extract.mode must be vocab, corpus or probe, got {mode!r}")
    device = ex.pick_device(cfg["device"])
    typer.echo(f"[extract] {cfg['model']} random_init={cfg['random_init']} mode={mode} on {device}")
    t0 = time.time()
    model, tok = ex.build_model(
        cfg["model"], random_init=cfg["random_init"], seed=cfg["seed"], device=device
    )
    tokens = ex.vocab_tokens(tok)
    V = len(tokens)
    out = stage_dir(rd, "extract")
    acts = None
    if mode == "vocab":
        ids = np.arange(V)
        bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        acts = ex.extract_activations(
            model, ids, bos_id=bos_id, batch_size=cfg["extract"]["batch_size"], device=device
        )
    elif mode == "corpus":
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
    else:  # probe: one row per occurrence, not per token
        pc = rd / "probe_corpus"
        wt, wo = np.load(pc / "windows_tokens.npy"), np.load(pc / "windows_offsets.npy")
        n_frames = model.config.num_hidden_layers + 2
        # straight to disk: ~250k occurrences x 8 frames x 512 x 2 B is ~2 GB
        occ = np.lib.format.open_memmap(out / "occ.npy", mode="w+", dtype=np.float16,
                                        shape=(n_frames, len(wo) - 1, model.config.hidden_size))
        ex.extract_probe(model, wt, wo, pad_id=tok.eos_token_id, batch_size=cfg["extract"]["batch_size"],
                         device=device, out=occ)
        occ.flush()
        del occ
        typer.echo(f"[extract] {len(wo) - 1} probe windows, {len(wt)} tokens")
    embed, unembed = ex.get_embed_unembed(model)
    ln_gain, ln_bias = ex.get_final_ln(model)
    n_layers = model.config.num_hidden_layers
    del model  # free it: everything downstream is numpy
    # embed/unembed matrices are padded past the tokenizer's vocab; keep only real rows
    if acts is not None:
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
    typer.echo(f"[extract] done in {time.time() - t0:.0f}s -> {out}")
    return rd


NORMALIZE_SOURCES = {"unit_mean": "acts.npy", "raw_mean": "acts_rawmean.npy"}


def stage_normalize(cfg: dict) -> Path:
    rd = _prepare_run_dir(cfg)
    if _mode(cfg) == "probe":
        raise ValueError("probe runs skip normalize: t, d and seg are ratios of differences, so a "
                         "shared offset (the anisotropy cone) or a uniform scale can't move them")
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
    fb_path = ex_dir / "freq_bins.npy"
    fb = np.load(fb_path) if fb_path.exists() else None  # runs extracted before this existed
    m = cfg["metrics"]
    t0 = time.time()
    counts_path = ex_dir / "counts.npy"
    if counts_path.exists():  # corpus-mode run
        # anisotropy should describe whatever normalize actually centered (unit_mean or
        # raw_mean), not silently fall back to acts.npy - that's the v0v1/v1 mismatch bug.
        aniso_source = cfg["normalize"].get("source", "unit_mean")
        raw = _raw_layers(ex_dir, NORMALIZE_SOURCES[aniso_source])
        eligible, cbins = corpus_eligibility(np.load(counts_path), m.get("min_count", 1))
        extra = dict(
            eligible=eligible, freq_bins=cbins, n_freq_bins=N_CORPUS_BINS, freq_bins_source="corpus",
            merge_rank_bins=fb, n_merge_rank_bins=N_FREQ_BINS + 1,
            self_sim=np.load(ex_dir / "self_sim.npy", mmap_mode="r"),
            self_sim_baseline=_load_json(ex_dir / "self_sim_baseline.json"),
        )
    else:  # vocab-mode (v0) run: acts.npy already is the raw, occurrence-level thing
        aniso_source = None  # compute_all defaults None -> "acts"
        raw = _raw_layers(ex_dir)
        extra = dict(freq_bins=fb, n_freq_bins=(int(fb.max()) + 1) if fb is not None else N_FREQ_BINS + 1)
    result = mt.compute_all(
        [norm[i] for i in range(norm.shape[0])], names, labels,
        knn_k=m["knn_k"], kmeans_k=m["kmeans_k"], seed=cfg["seed"], subsample=m["subsample"],
        raw_layers=raw, anisotropy_source=aniso_source, **extra,
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


def stage_viz(cfg: dict, *, replot: bool = False) -> Path:
    """`replot` redraws every figure from the saved umap_coords.npy instead of re-running
    the projection: for cosmetic plot changes, which shouldn't cost another AlignedUMAP."""
    rd = _prepare_run_dir(cfg)
    norm = np.load(rd / "normalize" / "acts_norm.npy", mmap_mode="r")
    labels = np.load(rd / "extract" / "labels.npy")
    names = _load_json(rd / "extract" / "layer_names.json")
    tokens = _load_json(rd / "extract" / "tokens.json")
    v = cfg["viz"]
    out = stage_dir(rd, "viz")

    # AlignedUMAP on 50k x 9 frames takes a long time; reuse the metrics subsample and
    # force the hand-picked trajectory tokens into it so they're always drawn.
    metrics = _load_json(rd / "metrics" / "metrics.json")
    idx = set(metrics["subsample_idx"])
    counts_path = rd / "extract" / "counts.npy"
    counts = np.load(counts_path) if counts_path.exists() else None
    tok_to_row = {t: i for i, t in enumerate(tokens)}
    # v0 configs have a flat trajectory_tokens list -> one group written as trajectories.png
    groups = v.get("trajectory_groups") or {"": v.get("trajectory_tokens", [])}
    group_rows: dict[str, dict[str, int]] = {}
    for g, words in groups.items():
        rows: dict[str, int] = {}
        for t in words:
            if t not in tok_to_row:
                typer.echo(f"[viz] {t!r} is not a single token, skipping")
                continue
            r = tok_to_row[t]
            if counts is not None and counts[r] == 0:
                typer.echo(f"[viz] {t!r} never occurs in the corpus, skipping")
                continue
            # below min_count is still drawn: the n= label says how much to trust it
            rows[repr(t) if counts is None else f"{t!r} n={int(counts[r])}"] = r
            idx.add(r)
        group_rows[g] = rows
    idx = np.array(sorted(idx))
    pos = {row: k for k, row in enumerate(idx)}

    if replot:
        # saved coords are only valid for the exact token set they were fitted on
        if not np.array_equal(np.load(out / "viz_idx.npy"), idx):
            raise ValueError("replot: viz tokens changed since the last projection "
                             "(metrics subsample or trajectory words); run viz without --replot")
        coords = np.load(out / "umap_coords.npy")
        if coords.shape[0] != norm.shape[0]:  # e.g. fitted before the L6 pre-LN frame existed
            raise ValueError(f"replot: saved coords have {coords.shape[0]} frames, acts_norm has "
                             f"{norm.shape[0]}; run viz without --replot")
        typer.echo(f"[viz] replot: reusing {out / 'umap_coords.npy'}")
    else:
        np.save(out / "viz_idx.npy", idx)
        t0 = time.time()
        layers = [np.asarray(norm[i][idx], dtype=np.float32) for i in range(norm.shape[0])]
        coords = viz.project_layers(
            layers, method=v["method"], n_neighbors=v["n_neighbors"], min_dist=v["min_dist"],
            seed=cfg["seed"],
        )
        typer.echo(f"[viz] {v['method']} on {len(idx)} tokens x {len(layers)} frames in {time.time() - t0:.0f}s")
    viz.plot_flipbook(coords, labels[idx], names, out)
    for g, rows in group_rows.items():
        if not rows:
            continue
        fname = "trajectories.png" if g == "" else f"trajectories_{g}.png"
        group = {text: pos[r] for text, r in rows.items()}
        viz.plot_trajectories(coords, labels[idx], names, group, out / fname, title=g or None)
        viz.plot_group_frames(coords, names, group, out / fname.replace("trajectories", "group"),
                              title=g or None)
    typer.echo(f"[viz] wrote flipbook -> {out / 'flipbook.gif'}")
    return rd


def compare_runs(run_dirs: list[Path], out: Path) -> Path:
    runs = {Path(rd).name: _load_json(Path(rd) / "metrics" / "metrics.json") for rd in run_dirs}
    return viz.plot_metrics(runs, out)


def run_v0v1(v0: Path, v1: Path) -> Path:
    """v0 (token alone) vs v1 (corpus-averaged) on exactly v1's metrics tokens.

    Recomputes v0's curves on v1's subsample and corpus frequency bins (v0's own metrics/
    is left alone), plus the per-frame cross overlap. Writes <runs>/v0v1_<v0 name>__<v1 name>/.
    """
    v0, v1 = Path(v0), Path(v1)
    n0 = np.load(v0 / "normalize" / "acts_norm.npy", mmap_mode="r")
    n1 = np.load(v1 / "normalize" / "acts_norm.npy", mmap_mode="r")
    names = _load_json(v1 / "extract" / "layer_names.json")
    if n0.shape != n1.shape or _load_json(v0 / "extract" / "layer_names.json") != names:
        raise ValueError(f"can't pair {v0.name} {n0.shape} with {v1.name} {n1.shape}: "
                         "different model, vocab or frames")
    cfg1 = load_config(v1 / "config.yaml")
    mc = cfg1["metrics"]
    m1 = _load_json(v1 / "metrics" / "metrics.json")
    idx = np.array(m1["subsample_idx"])
    eligible, cbins = corpus_eligibility(np.load(v1 / "extract" / "counts.npy"), mc.get("min_count", 1))
    t0 = time.time()
    frames0 = [n0[i] for i in range(n0.shape[0])]
    raw0 = _raw_layers(v0 / "extract")
    m0 = mt.compute_all(
        frames0, names, np.load(v1 / "extract" / "labels.npy"),
        knn_k=mc["knn_k"], kmeans_k=mc["kmeans_k"], seed=cfg1["seed"], subsample=None,
        subsample_idx=idx, eligible=eligible, freq_bins=cbins, n_freq_bins=N_CORPUS_BINS,
        freq_bins_source="corpus", raw_layers=raw0,
    )
    cross = mt.cross_overlap(frames0, [n1[i] for i in range(n1.shape[0])], idx, mc["knn_k"])
    # v0-name-only collided when the same v0 was paired with a second v1 run; both names
    # make the dir unique per pairing.
    out = stage_dir(v0.parent, f"v0v1_{v0.name}__{v1.name}")
    result = {"v0_run": v0.name, "v1_run": v1.name, "layer_names": names, "n": int(len(idx)),
              "knn_k": mc["knn_k"], "cross_overlap": cross, "v0": m0, "v1": m1}
    (out / "v0v1.json").write_text(json.dumps(result, indent=1))
    viz.plot_metrics({f"{v0.name} (v0)": m0, f"{v1.name} (v1)": m1}, out / "v0v1.png")
    viz.plot_cross_overlap(cross, names, out / "cross_overlap.png",
                           title=f"kNN overlap, {v0.name} vs {v1.name}, same {len(idx)} tokens")
    typer.echo(f"[v0v1] cross_overlap={np.round(cross, 3).tolist()} ({time.time() - t0:.0f}s) -> {out}")
    return out


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


@app.command("probe_corpus")
def probe_corpus_cmd(config: Path = _CONFIG):
    """Find probe-word hits in the books + Pile and cut a window ending at each (milestone B)."""
    stage_probe_corpus(load_config(config))


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
def viz_cmd(
    config: Path = _CONFIG,
    replot: bool = typer.Option(False, "--replot", help="redraw from saved umap_coords.npy, skip UMAP"),
):
    stage_viz(load_config(config), replot=replot)


# typer names commands after the function; we want `viz`, not `viz-cmd`
app.registered_commands[-1].name = "viz"


@app.command()
def all(config: Path = _CONFIG):  # noqa: A001 - it's the CLI verb we documented
    cfg = load_config(config)
    probe = _mode(cfg) == "probe"
    if "corpus" in cfg:  # v1 runs start from text; v0 configs have no corpus block
        stage_corpus(cfg)
    if probe:
        stage_probe_corpus(cfg)
    stage_extract(cfg)
    if not probe:  # probe runs skip it: t, d and seg don't care about a shared offset or scale
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


@app.command()
def v0v1(
    v0_run: Path = typer.Argument(..., help="v0 run dir, e.g. runs/pythia70m"),
    v1_run: Path = typer.Argument(..., help="v1 run dir, e.g. runs/pythia70m_corpus"),
):
    """Compare a v0 run and a v1 run on exactly the same tokens. Needs both normalized + v1's metrics."""
    run_v0v1(v0_run, v1_run)


if __name__ == "__main__":
    app()
