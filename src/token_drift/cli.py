"""`token-drift corpus|probe_corpus|extract|normalize|metrics|viz|occ|all --config configs/x.yaml`, plus `compare`, `ksweep`, `v0v1`, `timeline`.

This is the only module that touches files from another stage. Everything else is
arrays in, arrays out.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import typer
import yaml

from token_drift import betweenness as bt
from token_drift import corpus as cp
from token_drift import extract as ex
from token_drift import metrics as mt
from token_drift import probe as pb
from token_drift import timeline as tl
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


DEFAULT_TIMELINE_FRAMES = [0, "unembed"]
_STEP = re.compile(r"step(\d+)")


def step_of(rev: str) -> int:
    m = _STEP.fullmatch(str(rev))
    if not m:
        raise ValueError(f"revision {rev!r} isn't step<N> (Pythia's checkpoint branches are step0 .. step143000)")
    return int(m.group(1))


def step_dir_name(rev: str) -> str:
    """step64 -> step0000064: zero-padded so the folders list in training order."""
    return f"step{step_of(rev):07d}"


def _check_revisions(cfg: dict, path) -> None:
    """v2 checkpoint configs: v0 mode, trained weights, step<N> names, step0 included."""
    if "corpus" in cfg or "probe" in cfg or _mode(cfg) != "vocab":
        raise ValueError(f"{path}: revisions: (checkpoint runs) is v0 mode only; drop corpus: / probe:")
    if cfg.get("random_init"):
        raise ValueError(f"{path}: revisions: loads trained checkpoints, but random_init: true never loads anything")
    revs = cfg["revisions"]
    steps = [step_of(r) for r in revs]
    dup = [r for r, s in zip(revs, steps) if steps.count(s) > 1]
    if dup:  # step8 and step08 would share a folder
        raise ValueError(f"{path}: revisions lists the same step twice: {dup}")
    if 0 not in steps:
        raise ValueError(f"{path}: revisions needs step0: row drift is measured from the init")
    frames = (cfg.get("timeline") or {}).get("frames", DEFAULT_TIMELINE_FRAMES)
    if len(frames) > len(viz.RUN_COLORS):  # plot_timeline gives each frame one of RUN_COLORS
        raise ValueError(f"{path}: timeline.frames has {len(frames)} entries but the plots only have "
                         f"{len(viz.RUN_COLORS)} colours; keep it to {len(viz.RUN_COLORS)} or fewer")
    for f in frames:
        if f != "unembed" and (isinstance(f, bool) or not isinstance(f, int)):
            raise ValueError(f"{path}: timeline.frames entries are frame indices or 'unembed', got {f!r}")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    if "corpus" in cfg and "probe" in cfg:
        raise ValueError(f"{path}: has both corpus: (milestone A) and probe: (milestone B); pick one")
    if ("probe" in cfg) != (_mode(cfg) == "probe"):
        raise ValueError(f"{path}: a probe: block needs extract.mode: probe, and vice versa")
    if "revisions" in cfg:
        _check_revisions(cfg, path)
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
    dst = rd / "config.yaml"
    if "_config_yaml" in cfg:
        # a checkpoint run's per-revision sub-config has no yaml file of its own; write it out so
        # `token-drift metrics --config <step folder>/config.yaml` re-runs one revision by itself
        dst.write_text(cfg["_config_yaml"])
        return rd
    # Every run is reproducible from its yaml + seed, so the yaml travels with the outputs.
    src = Path(cfg["_config_path"])
    # skip when the config *is* the run dir's copy: shutil.copy onto itself raises SameFileError
    if src.exists() and not (dst.exists() and src.samefile(dst)):
        shutil.copy(src, dst)
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
    want = p["pile"]["shards"]
    shards = list(itertools.islice(cp.iter_parquet(p["pile"]["source"]), want))
    if len(shards) < want:  # else a half-downloaded source quietly gives a smaller Pile group
        raise ValueError(f"pile source {p['pile']['source']}: found {len(shards)} shard files, wanted {want}")
    for s, path in enumerate(shards):
        t0 = time.time()
        texts, _ = cp.read_parquet(path, p["pile"]["text_field"])
        docs += _tokenize_texts(tok, texts)
        sources += [f"pile:{s}"] * len(texts)
        typer.echo(f"[probe_corpus] pile shard {s}: {len(texts)} docs ({time.time() - t0:.0f}s)")
    return docs, sources


def _shared_piece_section(shared: list[dict], wid: dict, tok) -> str:
    """count_report.md section: triples where two words are the same point at frame 0."""
    lines = ["", "## Triples sharing a last token piece", "",
             "Vectors are read at the last piece, so at frame 0 these pairs are one point: that frame gets no verdict.", ""]
    if not shared:
        return "\n".join(lines + ["None.", ""])
    lines += ["| triple | pairs | shared piece |", "|---|---|---|"]
    for r in shared:
        pieces = ", ".join(repr(tok.decode([wid[a][-1]])) for a, _ in r["pairs"])
        lines.append(f"| {r['id']} | {'; '.join(' / '.join(pr) for pr in r['pairs'])} | {pieces} |")
    return "\n".join(lines) + "\n"


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
    shared = pb.shared_last_piece(words, wid)
    (out / "shared_last_piece.json").write_text(json.dumps(shared, indent=1))
    (out / "count_report.md").write_text(pb.count_report(words, counts, pool, min_count=m["min_count"])
                                         + _shared_piece_section(shared, wid, tok))
    # metrics reads this copy: the list these occurrences were cut for, whatever the yaml says later.
    # Copied last so a run that dies midway never pairs a new list with the old run's occurrences
    shutil.copy(p["words"], out / "probe_words.yaml")
    (out / "meta.json").write_text(json.dumps({
        "max_context": int(max_ctx), "n_windows": len(windows), "n_window_tokens": int(len(wt)),
        "n_tokens": {g: int(len(corpora[g][0])) for g in GROUPS}, "seconds": round(time.time() - t0),
    }, indent=1))
    typer.echo(f"[probe_corpus] {len(windows)} windows, {len(wt)} tokens ({time.time() - t0:.0f}s) -> {out}")
    return rd


def _windows_sha256(pc: Path, chunk: int = 16 << 20) -> str:
    """sha256 over windows_offsets.npy + windows_tokens.npy bytes, streamed (~0.8 GB, ~2 s).
    Ties an occ.npy to the exact windows it came from; a count check misses a same-size change."""
    h = hashlib.sha256()
    for name in ("windows_offsets.npy", "windows_tokens.npy"):
        with open(pc / name, "rb") as f:
            while block := f.read(chunk):
                h.update(block)
    return h.hexdigest()


def stage_extract(cfg: dict) -> Path:
    rd = _prepare_run_dir(cfg)
    mode = _mode(cfg)
    if mode not in ("vocab", "corpus", "probe"):
        raise ValueError(f"extract.mode must be vocab, corpus or probe, got {mode!r}")
    device = ex.pick_device(cfg["device"])
    typer.echo(f"[extract] {cfg['model']} random_init={cfg['random_init']} revision={cfg.get('revision')} "
               f"mode={mode} on {device}")
    t0 = time.time()
    model, tok = ex.build_model(
        cfg["model"], random_init=cfg["random_init"], seed=cfg["seed"], device=device,
        revision=cfg.get("revision"),
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
        # straight to disk: ~250k occurrences x 8 frames x 512 x 2 B is ~2 GB. Into a .part first:
        # open_memmap makes the full-size file up front, so a crash would leave a right-shaped,
        # partly zero occ.npy that looks valid
        part = out / "occ.npy.part"
        part.unlink(missing_ok=True)
        occ = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16,
                                        shape=(n_frames, len(wo) - 1, model.config.hidden_size))
        ex.extract_probe(model, wt, wo, pad_id=tok.eos_token_id, batch_size=cfg["extract"]["batch_size"],
                         device=device, out=occ)
        occ.flush()
        del occ
        part.replace(out / "occ.npy")
        (out / "probe_source.json").write_text(json.dumps(
            {"windows_sha256": _windows_sha256(pc), "n_windows": int(len(wo) - 1)}, indent=1))
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


@dataclass
class _Probe:
    meta: dict
    vocab: list[str]
    w_index: dict[str, int]
    widx: np.ndarray
    gidx: np.ndarray
    occ: np.ndarray  # memmap (frames, n_occ, d) float16


def _load_probe(rd: Path, columns=("word", "group")) -> _Probe:
    """Occurrence rows from probe_corpus + their vectors from extract, checked to line up."""
    meta = pq.read_table(rd / "probe_corpus" / "occ_meta.parquet", columns=list(columns)).to_pydict()
    vocab = sorted(set(meta["word"]))
    w_index = {w: i for i, w in enumerate(vocab)}
    widx = np.array([w_index[w] for w in meta["word"]], dtype=np.int64)
    gidx = np.array([GROUPS.index(g) for g in meta["group"]], dtype=np.int64)
    occ = np.load(rd / "extract" / "occ.npy", mmap_mode="r")
    if occ.shape[1] != len(widx):
        raise ValueError(f"extract/occ.npy has {occ.shape[1]} occurrences but probe_corpus/occ_meta.parquet "
                         f"has {len(widx)}: probe_corpus changed since; re-run extract")
    src = rd / "extract" / "probe_source.json"
    if not src.exists() or _load_json(src)["windows_sha256"] != _windows_sha256(rd / "probe_corpus"):
        raise ValueError("extract/occ.npy doesn't match probe_corpus's windows (probe_source.json missing "
                         "or a different hash): re-run extract")
    return _Probe(meta, vocab, w_index, widx, gidx, occ)


def _nan_to_none(x):
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_nan_to_none(v) for v in x]
    return x


def _stage_metrics_probe(cfg: dict) -> Path:
    rd = _prepare_run_dir(cfg)
    pc = rd / "probe_corpus"
    m = cfg["metrics"]
    t0 = time.time()
    words = pb.load_words(pc / "probe_words.yaml")  # the list these occurrences were cut for
    data = _load_probe(rd)
    found = {w: {g: c[g]["found"] for g in GROUPS} for w, c in _load_json(pc / "counts.json").items()}
    pool = {pos: [r["word"] for r in rows] for pos, rows in _load_json(pc / "null_pool.json").items()}
    # no unembed frame here: that's one row per token, not per occurrence
    names = _load_json(rd / "extract" / "layer_names.json")[:-1]
    points, self_sim = [], []
    for f in range(data.occ.shape[0]):
        pts, counts, ss = bt.unit_mean(np.asarray(data.occ[f], dtype=np.float32), data.widx, data.gidx,
                                       len(data.vocab), len(GROUPS))
        points.append(pts)
        self_sim.append(ss)
    entries, missing = bt.run_q16(points, counts, data.w_index, words["triples"], pool, found,
                                  groups=GROUPS, min_count=m["min_count"], null_k=m["null_k"],
                                  between_pct=m["between_pct"])
    shared = {r["id"]: r["pairs"] for r in _load_json(pc / "shared_last_piece.json")}
    for e in entries:
        e["shared_last_piece"] = shared.get(e["id"], [])
    ss = np.stack(self_sim)  # (frames, words, groups)
    result = {
        "layer_names": names, "groups": list(GROUPS), "min_count": m["min_count"], "null_k": m["null_k"],
        "between_pct": m["between_pct"], "triples": entries,
        "summary": bt.summarize(entries, GROUPS, len(names)), "missing": missing,
        "self_sim": {w: {g: [None if np.isnan(x) else float(x) for x in ss[:, i, g_i]]
                         for g_i, g in enumerate(GROUPS)} for w, i in data.w_index.items()},
    }
    out = stage_dir(rd, "metrics")
    # allow_nan=False: a stray NaN would make q16.json invalid JSON for anything but Python
    (out / "q16.json").write_text(json.dumps(_nan_to_none(result), indent=1, allow_nan=False))
    for s, by_g in result["summary"].items():
        for g, per_frame in by_g.items():
            best = [x["best_of_three"] for x in per_frame]
            typer.echo(f"[metrics] {s}/{g}: n={per_frame[0]['n']} best_of_three per frame={best}")
    typer.echo(f"[metrics] {len(missing)} triple x group missing ({time.time() - t0:.0f}s) -> {out / 'q16.json'}")
    return rd


def stage_metrics(cfg: dict) -> Path:
    if _mode(cfg) == "probe":
        return _stage_metrics_probe(cfg)
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


# ---------- v2: training checkpoints ----------

_LOOP_KEYS = ("revisions", "timeline")


def _revision_cfg(cfg: dict, rev: str) -> dict:
    """The plain v0 config for one revision. Its run dir is the step folder inside the parent run."""
    sub = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_") and k not in _LOOP_KEYS})
    sub.update(run_name=step_dir_name(rev), out_dir=str(run_dir(cfg)), revision=rev)
    sub["_config_yaml"] = yaml.safe_dump(sub, sort_keys=False)
    sub["_config_path"] = cfg["_config_path"]
    return sub


def resolve_frames(spec: list, names: list[str]) -> list[int]:
    """timeline.frames -> indices into the normalized stack. 'unembed' = the last frame."""
    out = []
    for f in spec:
        i = len(names) - 1 if f == "unembed" else f
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(names):
            raise ValueError(f"timeline.frames: {f!r} isn't a frame index 0..{len(names) - 1} or 'unembed'")
        out.append(i)
    return out


def _timeline_rows(subsample_idx, tokens: list[str], trajectory_tokens: list[str]) -> np.ndarray:
    """Rows kept per revision: the metrics subsample plus the hand-picked trajectory tokens
    (kept in the frame rows so later plots can use them; the flipbooks don't highlight them yet)."""
    tok_to_row = {t: i for i, t in enumerate(tokens)}
    rows = {int(i) for i in subsample_idx}
    for t in trajectory_tokens:
        if t in tok_to_row:
            rows.add(tok_to_row[t])
        else:
            typer.echo(f"[checkpoints] {t!r} is not a single token, skipping")
    return np.array(sorted(rows), dtype=np.int64)


def _revision_done(sd: Path) -> bool:
    # idx.npy is written last, so this is False for a revision that crashed after metrics
    return (sd / "metrics" / "metrics.json").exists() and (sd / "frames" / "idx.npy").exists()


def _save_revision_extras(sub: dict, frames_spec: list) -> None:
    """Copy what timeline needs out of extract/ + normalize/, before those get deleted."""
    sd = run_dir(sub)
    ex_dir = sd / "extract"
    names = _load_json(ex_dir / "layer_names.json")
    tokens = _load_json(ex_dir / "tokens.json")
    fids = resolve_frames(frames_spec, names)
    sub_idx = _load_json(sd / "metrics" / "metrics.json")["subsample_idx"]
    idx = _timeline_rows(sub_idx, tokens, sub["viz"].get("trajectory_tokens", []))
    norm = np.load(sd / "normalize" / "acts_norm.npy", mmap_mode="r")
    wd = stage_dir(sd, "weights")
    for w in ("embed", "unembed"):  # raw matrices, full vocab: row drift needs every row
        shutil.copy(ex_dir / f"{w}.npy", wd / f"{w}.npy")
    fr = stage_dir(sd, "frames")
    np.save(fr / "frames.npy", np.stack([np.asarray(norm[i][idx], dtype=np.float16) for i in fids]))
    for f in ("labels.npy", "freq_bins.npy", "tokens.json", "layer_names.json"):
        shutil.copy(ex_dir / f, fr / f)
    (fr / "frames.json").write_text(json.dumps({"ids": fids, "names": [names[i] for i in fids]}))
    np.save(fr / "idx.npy", idx)  # last: its presence marks the revision finished (_revision_done)


def stage_checkpoints(cfg: dict) -> Path:
    """v2: extract -> normalize -> metrics once per revision, each in its own step folder."""
    rd = _prepare_run_dir(cfg)
    tl_cfg = cfg.get("timeline") or {}
    frames_spec = tl_cfg.get("frames", DEFAULT_TIMELINE_FRAMES)
    revs = sorted(cfg["revisions"], key=step_of)
    todo = [r for r in revs if not _revision_done(rd / step_dir_name(r))]
    # only the ones still to run: a finished run re-checks fine offline
    missing = ex.missing_revisions(cfg["model"], todo) if todo else []
    if missing:
        raise ValueError(f"{cfg['model']} has no revision(s) {missing} on the Hub; nothing downloaded")
    for rev in revs:
        sub = _revision_cfg(cfg, rev)
        sd = run_dir(sub)
        if rev not in todo:
            typer.echo(f"[checkpoints] {rev}: already done, skipping")
            if not tl_cfg.get("keep_acts", False):
                for s in ("extract", "normalize"):  # crash between idx.npy and the rmtree leaves ~1 GB each
                    if (sd / s).exists():
                        shutil.rmtree(sd / s)
            continue
        t0 = time.time()
        stage_extract(sub)
        stage_normalize(sub)
        stage_metrics(sub)
        _save_revision_extras(sub, frames_spec)
        if not tl_cfg.get("keep_acts", False):
            for s in ("extract", "normalize"):  # ~1 GB per revision; frames/ + weights/ have what timeline needs
                shutil.rmtree(sd / s)
        typer.echo(f"[checkpoints] {rev} done in {time.time() - t0:.0f}s -> {sd}")
    return rd


TIMELINE_CURVES = ("knn_purity", "knn_purity_shuffled", "silhouette", "silhouette_shuffled",
                   "knn_consecutive", "anisotropy", "top_pc_share")


@dataclass
class _Ckpt:
    revs: list[str]  # step order
    dirs: dict[str, Path]
    idx: np.ndarray  # frames.npy rows = metrics subsample + trajectory tokens
    sub_idx: list[int]  # the metrics subsample
    frame_ids: list[int]
    frame_names: list[str]
    frames: dict[str, np.ndarray]  # rev -> (n timeline frames, len(idx), d) float16
    metrics: dict[str, dict]


def _load_checkpoints(cfg: dict) -> _Ckpt:
    """Every revision's frames + metrics, checked to be finished and on the same tokens and frames."""
    rd = run_dir(cfg)
    revs = sorted(cfg["revisions"], key=step_of)
    dirs = {r: rd / step_dir_name(r) for r in revs}
    todo = [r for r in revs if not _revision_done(dirs[r])]
    if todo:
        raise ValueError(f"revisions not finished yet: {todo}; run `token-drift all --config {cfg['_config_path']}`")
    final = revs[-1]
    mets = {r: _load_json(dirs[r] / "metrics" / "metrics.json") for r in revs}
    idx = np.load(dirs[final] / "frames" / "idx.npy")
    sub_idx = mets[final]["subsample_idx"]
    info = _load_json(dirs[final] / "frames" / "frames.json")
    for r in revs:
        # same seed + config -> same draw; a mismatch means this revision ran under another config
        if not np.array_equal(np.load(dirs[r] / "frames" / "idx.npy"), idx) or mets[r]["subsample_idx"] != sub_idx:
            raise ValueError(f"{r}: its tokens (frames/idx.npy or metrics subsample_idx) differ from {final}'s; "
                             f"delete {dirs[r]} and re-run all")
        if _load_json(dirs[r] / "frames" / "frames.json") != info:
            raise ValueError(f"{r}: stores different frames than {final}; delete {dirs[r]} and re-run all")
    names = _load_json(dirs[final] / "frames" / "layer_names.json")
    want = resolve_frames((cfg.get("timeline") or {}).get("frames", DEFAULT_TIMELINE_FRAMES), names)
    if want != info["ids"]:
        # the step folders only kept the frames asked for at run time; extract/ is gone
        raise ValueError(f"timeline.frames is now {want} but the revisions were run with {info['ids']}; "
                         "put it back, or delete the step folders and re-run all")
    frames = {r: np.load(dirs[r] / "frames" / "frames.npy") for r in revs}  # ~30 MB each for Pythia
    return _Ckpt(revs, dirs, idx, sub_idx, info["ids"], info["names"], frames, mets)


def stage_timeline(cfg: dict) -> Path:
    """v2 across checkpoints: A (per-checkpoint curves), B (geometry vs final), C (row drift)."""
    rd = run_dir(cfg)
    ck = _load_checkpoints(cfg)
    t0 = time.time()
    final, first = ck.revs[-1], ck.revs[0]  # first is step0: load_config insists
    steps = [step_of(r) for r in ck.revs]
    fin_dir = ck.dirs[final] / "frames"
    names = ck.metrics[final]["layer_names"]
    in_sub = np.isin(ck.idx, ck.sub_idx)  # B on the metrics subsample, like every other curve
    k = cfg["metrics"]["knn_k"]
    with_final = {}
    for j, name in enumerate(ck.frame_names):
        fin = np.asarray(ck.frames[final][j][in_sub], dtype=np.float32)
        xs = [np.asarray(ck.frames[r][j][in_sub], dtype=np.float32) for r in ck.revs]
        with_final[name] = {"knn": [tl.overlap_with_final(x, fin, k) for x in xs],
                            "cka": [tl.cka_with_final(x, fin) for x in xs]}
    fb = np.load(fin_dir / "freq_bins.npy")  # merge-rank bins: 0 = base/byte, 1 = most frequent ...
    n_bins = int(fb.max()) + 1
    drift = {}
    for w in ("embed", "unembed"):
        w0 = np.load(ck.dirs[first] / "weights" / f"{w}.npy")
        per_row = {r: tl.row_drift(np.load(ck.dirs[r] / "weights" / f"{w}.npy"), w0) for r in ck.revs}
        drift[w] = [tl.median_by_bin(per_row[r], fb, n_bins) for r in ck.revs]
        if w == "embed":
            last = per_row[final]
    tokens = _load_json(fin_dir / "tokens.json")
    # sanity check 3: the embed rows that barely moved should be tokens the Pile (nearly) never has
    order = np.argsort(np.where(np.isnan(last), np.inf, last), kind="stable")[:20]
    lowest = [{"id": int(i), "token": tokens[i], "drift": float(last[i]), "freq_bin": int(fb[i])} for i in order]

    curves = {key: [ck.metrics[r][key] for r in ck.revs] for key in TIMELINE_CURVES}
    pur = np.asarray(curves["knn_purity"], dtype=float) - np.asarray(curves["knn_purity_shuffled"], dtype=float)
    half = {f"purity_minus_shuffled {names[f]}": tl.half_way_step(pur[:, f], steps) for f in ck.frame_ids}
    n_layers = len(names) - 3
    if n_layers > 2:  # L1->L2 .. L(n-2)->L(n-1); a 2-layer toy has no middle block
        cons = np.asarray(curves["knn_consecutive"], dtype=float)[:, 1 : n_layers - 1]
        half["middle_block_overlap"] = tl.half_way_step(cons.mean(1), steps)
    for name, wf in with_final.items():
        half[f"knn_with_final {name}"] = tl.half_way_step(wf["knn"], steps)
        half[f"cka_with_final {name}"] = tl.half_way_step(wf["cka"], steps)

    result = {
        "revisions": ck.revs, "steps": steps, "final": final, "layer_names": names,
        "timeline_frames": ck.frame_names, "timeline_frame_ids": ck.frame_ids,
        "knn_k": k, "n_subsample": int(in_sub.sum()), "curves": curves, "with_final": with_final,
        "drift": drift, "n_freq_bins": n_bins, "freq_bins_source": "merge_rank",
        "half_way": half, "lowest_embed_drift": lowest,
    }
    out = stage_dir(rd, "timeline")
    # allow_nan=False: NaN (empty bins, NaN silhouettes) goes out as null, so it's valid JSON
    (out / "timeline.json").write_text(json.dumps(_nan_to_none(result), indent=1, allow_nan=False))
    viz.plot_timeline(result, out / "timeline.png")
    viz.plot_drift(result, out / "drift.png")
    for name, wf in with_final.items():
        typer.echo(f"[timeline] {name}: knn_with_final={np.round(wf['knn'], 3).tolist()} "
                   f"cka_with_final={np.round(wf['cka'], 3).tolist()}")
    typer.echo(f"[timeline] half_way={half}")
    typer.echo(f"[timeline] {len(ck.revs)} revisions ({time.time() - t0:.0f}s) -> {out}")
    return rd


def frame_tag(name: str) -> str:
    """Frame name -> file-name tag: 'L0 (embed)' -> 'L0', 'L6 (pre-LN)' -> 'L6_preLN'."""
    return name.replace(" (embed)", "").replace(" (pre-LN)", "_preLN").replace(" (post-LN)", "_postLN")


def stage_flipbooks(cfg: dict) -> Path:
    """v2 training-time flipbooks: one projection per timeline frame, pages = checkpoints.

    Same AlignedUMAP as v0's viz, but the "same token, next frame" link now runs across
    checkpoints instead of layers, so a dot moving between pages is training moving that token.
    ~20-50 min per frame on the real run (same cost as one v0 flipbook).
    """
    rd = run_dir(cfg)
    ck = _load_checkpoints(cfg)
    labels = np.load(ck.dirs[ck.revs[-1]] / "frames" / "labels.npy")[ck.idx]
    v = cfg["viz"]
    out = stage_dir(rd, "timeline")
    for j, name in enumerate(ck.frame_names):
        tag = frame_tag(name)
        t0 = time.time()
        coords = viz.project_layers([ck.frames[r][j] for r in ck.revs], method=v["method"],
                                    n_neighbors=v["n_neighbors"], min_dist=v["min_dist"], seed=cfg["seed"])
        viz.plot_flipbook(coords, labels, [f"{name}, {r}" for r in ck.revs], out,
                          png_names=[f"umap_{tag}_{step_dir_name(r)}.png" for r in ck.revs],
                          gif_name=f"flipbook_{tag}.gif", coords_name=f"umap_coords_{tag}.npy")
        typer.echo(f"[flipbooks] {name}: {v['method']} on {len(ck.idx)} tokens x {len(ck.revs)} checkpoints "
                   f"({time.time() - t0:.0f}s) -> {out / f'flipbook_{tag}.gif'}")
    return rd


def _stage_viz_probe(cfg: dict) -> Path:
    rd = _prepare_run_dir(cfg)
    out = stage_dir(rd, "viz")
    q16 = _load_json(rd / "metrics" / "q16.json")
    names = q16["layer_names"]
    occ_frames = cfg["viz"]["occ_frames"]
    bad = [f for f in occ_frames if not 0 <= f < len(names)]
    if bad:
        raise ValueError(f"viz.occ_frames {bad} out of range: this run has frames 0..{len(names) - 1}")
    viz.plot_q16_summary(q16, out / "q16_summary.png")
    data = _load_probe(rd)
    pts = {f: bt.unit_mean(np.asarray(data.occ[f], dtype=np.float32), data.widx, data.gidx,
                           len(data.vocab), len(GROUPS))[0] for f in occ_frames}
    by_slot: dict[tuple, list] = {}
    for t in q16["triples"]:
        by_slot.setdefault((t["set"], t["concept"], t["pos"]), []).append(t)
    for (s, c, p), entries in by_slot.items():
        stem = f"{s}_{c}_{p}"
        viz.plot_q16_triples(entries, names, out / f"q16_{stem}.png", title=f"{s} / {c} / {p}")
        rows = []
        for t in entries:
            di, mi, ei = (data.w_index.get(w) for w in t["words"])
            per_group = {}
            for g_i, g in enumerate(GROUPS):
                if "missing" in t["groups"][g]:
                    continue
                sel = np.flatnonzero((data.widx == mi) & (data.gidx == g_i))
                per_group[g] = [bt.occurrence_stats(np.asarray(data.occ[f][sel], dtype=np.float32),
                                                    pts[f][di, g_i], pts[f][ei, g_i])[:2] for f in occ_frames]
            if per_group:
                rows.append((" / ".join(t["words"]), per_group))
        if rows:
            viz.plot_q16_occ(rows, [names[f] for f in occ_frames], out / f"q16_occ_{stem}.png",
                             title=f"{s} / {c} / {p}")
    typer.echo(f"[viz] q16 figures for {len(by_slot)} set/concept/pos slots -> {out}")
    return rd


def occ_report(cfg: dict, words: list[str], frame: int, group: str, n: int) -> str:
    """The n most and least 'between' occurrences of the middle word, with their text.
    Reading the senses is half the point: is the between-ness the ethical sense or not?"""
    if len(words) != 3:
        raise ValueError(f"--triple needs three words (deficiency,mean,excess), got {words}")
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
    rd = run_dir(cfg)
    data = _load_probe(rd, columns=("word", "group", "source", "snippet"))
    if not 0 <= frame < data.occ.shape[0]:  # -1 would silently index the last frame
        raise ValueError(f"frame {frame} out of range: this run has frames 0..{data.occ.shape[0] - 1}")
    g_i = GROUPS.index(group)
    gcount = np.bincount(data.widx[data.gidx == g_i], minlength=len(data.vocab))
    for w in words:
        if w not in data.w_index or gcount[data.w_index[w]] == 0:
            raise ValueError(f"{w!r} has no occurrences in {group} in this run")
    occ_f = np.asarray(data.occ[frame], dtype=np.float32)
    pts = bt.unit_mean(occ_f, data.widx, data.gidx, len(data.vocab), len(GROUPS))[0]
    d_i, m_i, e_i = (data.w_index[w] for w in words)
    sel = np.flatnonzero((data.widx == m_i) & (data.gidx == g_i))
    t, d, seg = bt.occurrence_stats(occ_f[sel], pts[d_i, g_i], pts[e_i, g_i])
    order = np.argsort(seg, kind="stable")
    names = _load_json(rd / "extract" / "layer_names.json")
    lines = [f"{' / '.join(words)} in {group}, frame {names[frame]}: {len(sel)} occurrences of {words[1]!r} "
             f"(n = {', '.join(f'{w}:{gcount[data.w_index[w]]}' for w in words)})"]
    for label, picks in (("most between", order[:n]), ("least between", order[::-1][:n])):
        lines.append(f"\n{label}:")
        for k in picks:
            row = sel[k]
            lines.append(f"  seg={seg[k]:.3f} t={t[k]:.2f} d={d[k]:.2f} [{data.meta['source'][row]}] "
                         f"...{data.meta['snippet'][row]}")
    return "\n".join(lines)


def stage_viz(cfg: dict, *, replot: bool = False) -> Path:
    """`replot` redraws every figure from the saved umap_coords.npy instead of re-running
    the projection: for cosmetic plot changes, which shouldn't cost another AlignedUMAP."""
    if _mode(cfg) == "probe":
        return _stage_viz_probe(cfg)
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
def occ(
    config: Path = _CONFIG,
    triple: str = typer.Option(..., "--triple", help="deficiency,mean,excess e.g. cowardice,courage,rashness"),
    frame: int = typer.Option(3, "--frame", help="frame index (0 = embed)"),
    group: str = typer.Option("books", "--group", help="books | pile"),
    n: int = typer.Option(5, "-n", help="how many occurrences at each end"),
):
    """Print the most and least 'between' occurrences of a triple's virtue, with snippets."""
    typer.echo(occ_report(load_config(config), triple.split(","), frame, group, n))


@app.command("timeline")
def timeline_cmd(
    config: Path = _CONFIG,
    skip_flipbooks: bool = typer.Option(False, "--skip-flipbooks",
                                        help="numbers + timeline.png + drift.png only; the flipbooks take ~25 min"),
):
    """v2: compare a checkpoint run's revisions (every revision must have finished `all`)."""
    cfg = load_config(config)
    stage_timeline(cfg)
    if not skip_flipbooks:
        stage_flipbooks(cfg)


@app.command()
def all(config: Path = _CONFIG):  # noqa: A001 - it's the CLI verb we documented
    cfg = load_config(config)
    if "revisions" in cfg:  # v2: the v0 pipeline once per training checkpoint, then across them
        stage_checkpoints(cfg)
        stage_timeline(cfg)
        stage_flipbooks(cfg)
        typer.echo(f"[all] the HF cache now holds {len(cfg['revisions'])} revisions of {cfg['model']} "
                   "(~160 MB each for pythia-70m); `uv run hf cache ls` / `uv run hf cache rm` to reclaim it")
        return
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
