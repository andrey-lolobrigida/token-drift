"""Stage wiring. Extract needs a real model so it's covered by test_extract; here we
fake its outputs on disk and run the downstream stages through the same code the CLI uses."""
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from token_drift import cli
from token_drift.labels import categorize_all
from conftest import tiny_docs, tiny_model, tiny_tokenizer

V, D, L = 120, 8, 2  # tiny "model": 2 layers -> embed, L1, L2 pre-LN, L2 post-LN + unembed = 5 frames


@pytest.fixture
def cfg(tmp_path):
    c = {
        "run_name": "fake",
        "model": "not-loaded",
        "random_init": False,
        "seed": 0,
        "device": "cpu",
        "extract": {"batch_size": 64, "dtype_on_disk": "float16"},
        "normalize": {"center": True, "unit_norm": True, "drop_top_pcs": 0},
        "metrics": {"subsample": 100, "knn_k": 5, "kmeans_k": 4, "intrinsic_dim": False},
        "viz": {
            "method": "stacked_umap", "n_neighbors": 5, "min_dist": 0.1,
            "color_by": "category", "trajectory_tokens": [" the", "7"],
        },
        "out_dir": str(tmp_path / "runs"),
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(c))
    return p


@pytest.fixture
def fake_extract(cfg):
    """Write what stage_extract would have written."""
    c = cli.load_config(cfg)
    rd = cli.run_dir(c)
    ex = rd / "extract"
    ex.mkdir(parents=True)
    rng = np.random.default_rng(0)
    tokens = [" the", "7", "The", "ing", " bank"] + [f"tok{i}" for i in range(V - 5)]
    acts = rng.normal(size=(L + 2, V, D)).astype(np.float16)
    np.save(ex / "acts.npy", acts)
    np.save(ex / "embed.npy", acts[0])
    np.save(ex / "unembed.npy", rng.normal(size=(V, D)).astype(np.float16))
    np.save(ex / "labels.npy", categorize_all(tokens))
    np.save(ex / "freq_bins.npy", rng.integers(0, 6, size=V).astype(np.int8))
    (ex / "tokens.json").write_text(json.dumps(tokens))
    (ex / "layer_names.json").write_text(json.dumps(cli.layer_names(L)))
    return c, rd


@pytest.fixture
def corpus_run(tmp_path, fake_extract):
    """A fake corpus-mode run next to the v0 one: same tokens, same frame 0 and unembed,
    perturbed middle frames, plus the files corpus-mode extract adds."""
    c0, rd0 = fake_extract
    c = yaml.safe_load(open(c0["_config_path"]))
    c["run_name"] = "fake_corpus"
    c["extract"].update(mode="corpus", min_context=4)
    c["normalize"]["source"] = "unit_mean"
    c["metrics"].update(min_count=20, subsample=50)
    p = tmp_path / "corpus_run.yaml"
    p.write_text(yaml.safe_dump(c))
    c = cli.load_config(p)
    rd = cli.run_dir(c)
    shutil.copytree(rd0 / "extract", rd / "extract")
    ex = rd / "extract"
    rng = np.random.default_rng(1)
    acts = np.load(ex / "acts.npy").astype(np.float32)
    acts[1:] += rng.normal(size=acts[1:].shape)  # context moved the blocks; frame 0 is still the embedding
    np.save(ex / "acts.npy", acts.astype(np.float16))
    np.save(ex / "acts_rawmean.npy", (acts * 3).astype(np.float16))
    counts = rng.integers(1, 60, size=V).astype(np.int64)
    counts[:2] = 100  # " the" and "7" are common
    np.save(ex / "counts.npy", counts)
    np.save(ex / "self_sim.npy", rng.uniform(0.2, 0.9, size=(L + 2, V)).astype(np.float32))
    (ex / "self_sim_baseline.json").write_text(json.dumps([0.1] * (L + 2)))
    return c, rd


def test_load_config_and_run_dir(cfg, tmp_path):
    c = cli.load_config(cfg)
    assert c["run_name"] == "fake"
    assert cli.run_dir(c) == tmp_path / "runs" / "fake"


def test_layer_names_marks_bookends_and_final_ln():
    names = cli.layer_names(6)
    assert len(names) == 9  # embed, 5 blocks, block 6 pre- and post-LN, unembed
    assert names[0].startswith("L0")
    # HF's last hidden state is post final LayerNorm; the hook frame before it isn't. Say so.
    assert names[6] == "L6 (pre-LN)" and names[7] == "L6 (post-LN)"
    assert names[-1] == "unembed"


def test_stage_normalize_appends_unembed_as_last_layer(fake_extract):
    c, rd = fake_extract
    cli.stage_normalize(c)
    norm = np.load(rd / "normalize" / "acts_norm.npy")
    assert norm.shape == (L + 3, V, D)
    assert norm.dtype == np.float16
    np.testing.assert_allclose(np.linalg.norm(norm[-1].astype(np.float32), axis=1), 1, atol=2e-3)


def test_stage_metrics_writes_json_and_plots(fake_extract):
    c, rd = fake_extract
    cli.stage_normalize(c)
    cli.stage_metrics(c)
    md = rd / "metrics"
    m = json.loads((md / "metrics.json").read_text())
    assert m["layer_names"] == cli.layer_names(L)
    assert len(m["knn_consecutive"]) == L + 2
    assert (md / "metrics.png").exists() and (md / "cka.png").exists()
    # literature checks ride along: anisotropy on raw acts, change by frequency bin
    assert len(m["anisotropy"]) == L + 3 and len(m["top_pc_share"]) == L + 3
    assert np.asarray(m["knn_change_by_freq"]).shape == (L + 2, 6)


def test_stage_metrics_without_freq_bins_still_runs(fake_extract):
    c, rd = fake_extract
    (rd / "extract" / "freq_bins.npy").unlink()
    cli.stage_normalize(c)
    cli.stage_metrics(c)
    m = json.loads((rd / "metrics" / "metrics.json").read_text())
    assert m["knn_change_by_freq"] is None and len(m["anisotropy"]) == L + 3


def test_stage_viz_uses_metric_subsample_plus_trajectory_tokens(fake_extract):
    c, rd = fake_extract
    cli.stage_normalize(c)
    cli.stage_metrics(c)
    cli.stage_viz(c)
    vd = rd / "viz"
    assert (vd / "flipbook.gif").exists()
    coords = np.load(vd / "umap_coords.npy")
    idx = np.load(vd / "viz_idx.npy")
    assert coords.shape[0] == L + 3
    assert coords.shape[1] == len(idx)
    # both trajectory tokens (" the" = row 0, "7" = row 1) are guaranteed in the subsample
    assert 0 in idx and 1 in idx
    assert (vd / "trajectories.png").exists()
    # nothing but config.yaml and the four stage dirs at the top level
    assert sorted(p.name for p in rd.iterdir()) == ["config.yaml", "extract", "metrics", "normalize", "viz"]


def test_config_is_copied_into_run_dir(fake_extract, cfg):
    c, rd = fake_extract
    cli.stage_normalize(c)
    assert (rd / "config.yaml").exists()


def test_compare_overlays_two_runs(fake_extract, tmp_path):
    c, rd = fake_extract
    cli.stage_normalize(c)
    cli.stage_metrics(c)
    out = tmp_path / "compare.png"
    cli.compare_runs([rd, rd], out)
    assert out.exists()


def test_ksweep_writes_json_per_run_and_one_plot(fake_extract, tmp_path):
    c, rd = fake_extract
    cli.stage_normalize(c)
    cli.stage_metrics(c)
    out = tmp_path / "ksweep.png"
    cli.run_ksweep([c], [3, 5, 10], out)
    s = json.loads((rd / "metrics" / "ksweep.json").read_text())
    assert s["ks"] == [3, 5, 10]
    # the config's k (5) reproduces metrics.json exactly: same subsample, same neighbours
    m = json.loads((rd / "metrics" / "metrics.json").read_text())
    assert s["by_k"]["5"]["knn_consecutive"] == m["knn_consecutive"]
    assert out.exists()


# ---------- corpus stage (v1) ----------

class _CharTok:
    """Stands in for the HF tokenizer in stage_corpus: one id per character."""
    eos_token_id = 0

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[1 + ord(ch) % 50 for ch in t] for t in texts]}


def _corpus_cfg(tmp_path, monkeypatch, **corpus_overrides):
    monkeypatch.setattr(cli, "_load_tokenizer", lambda name: _CharTok())
    src = tmp_path / "docs"
    src.mkdir(exist_ok=True)
    # part0: 5 docs of 7 chars + one null doc; part1 has no meta column
    pq.write_table(pa.table({
        "text": ["abcdefg"] * 5 + [None],
        "meta": [{"pile_set_name": "A"}] * 4 + [{"pile_set_name": "B"}] * 2,
    }), src / "part0.parquet")
    pq.write_table(pa.table({"text": ["hijklmn"] * 5}), src / "part1.parquet")
    corpus = {"source": str(src), "text_field": "text", "max_tokens": None, "window": 8,
              "shuffle": "none", **corpus_overrides}
    c = {"run_name": "fake_corpus", "model": "not-loaded", "random_init": False, "seed": 0,
         "device": "cpu", "corpus": corpus, "out_dir": str(tmp_path / "runs")}
    p = tmp_path / "corpus_cfg.yaml"
    p.write_text(yaml.safe_dump(c))
    return cli.load_config(p)


def test_stage_corpus_writes_windows_and_meta(tmp_path, monkeypatch):
    c = _corpus_cfg(tmp_path, monkeypatch)
    rd = cli.stage_corpus(c)
    w = np.load(rd / "corpus" / "windows.npy")
    meta = json.loads((rd / "corpus" / "meta.json").read_text())
    # part0: 5*(7+1) + (0+1) = 41 tokens; part1: 5*8 = 40 -> 81 read, 10 full windows of 8
    assert w.shape == (10, 8) and w.dtype == np.int32
    assert meta["n_docs"] == 11 and meta["n_tokens_read"] == 81 and meta["n_tokens"] == 80
    assert meta["n_windows"] == 10 and meta["source_mix"] == {"A": 4, "B": 2}
    assert w[0].tolist() == [1 + ord(ch) % 50 for ch in "abcdefg"] + [0]


def test_stage_corpus_max_tokens_stops_before_the_second_file(tmp_path, monkeypatch):
    c = _corpus_cfg(tmp_path, monkeypatch, max_tokens=16)
    rd = cli.stage_corpus(c)
    meta = json.loads((rd / "corpus" / "meta.json").read_text())
    assert np.load(rd / "corpus" / "windows.npy").shape == (2, 8)
    assert meta["n_docs"] == 2


def test_stage_corpus_window_shuffle_keeps_each_windows_tokens(tmp_path, monkeypatch):
    plain = np.load(cli.stage_corpus(_corpus_cfg(tmp_path, monkeypatch)) / "corpus" / "windows.npy")
    shuf = np.load(cli.stage_corpus(_corpus_cfg(tmp_path, monkeypatch, shuffle="window")) / "corpus" / "windows.npy")
    assert np.array_equal(np.sort(plain, axis=1), np.sort(shuf, axis=1))
    assert not np.array_equal(plain, shuf)


def test_stage_corpus_smaller_than_one_window_fails_loudly(tmp_path, monkeypatch):
    c = _corpus_cfg(tmp_path, monkeypatch, window=1000)
    with pytest.raises(ValueError, match="less than one"):
        cli.stage_corpus(c)


def test_stage_corpus_rejects_unknown_shuffle(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="shuffle"):
        cli.stage_corpus(_corpus_cfg(tmp_path, monkeypatch, shuffle="global"))


def test_stage_extract_corpus_mode_writes_means_counts_and_self_sim(tmp_path, monkeypatch):
    tok = tiny_tokenizer()
    model = tiny_model(tok, window=16)
    monkeypatch.setattr(cli.ex, "build_model", lambda name, **kw: (model, tok))
    c = {"run_name": "tiny", "model": "tiny", "random_init": False, "seed": 0, "device": "cpu",
         "extract": {"mode": "corpus", "batch_size": 4, "min_context": 4, "dtype_on_disk": "float16"},
         "out_dir": str(tmp_path / "runs")}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(c))
    c = cli.load_config(p)
    rd = cli.run_dir(c)
    (rd / "corpus").mkdir(parents=True)
    ids = tok(" the cat sat in the inner dog then ran far." * 20, add_special_tokens=False)["input_ids"]
    w = np.array(ids[: (len(ids) // 16) * 16], dtype=np.int32).reshape(-1, 16)
    np.save(rd / "corpus" / "windows.npy", w)

    cli.stage_extract(c)
    ex = rd / "extract"
    V = len(tok)
    acts = np.load(ex / "acts.npy")
    assert acts.shape == (4, V, 16) and np.load(ex / "acts_rawmean.npy").shape == acts.shape
    counts = np.load(ex / "counts.npy")
    assert counts.shape == (V,) and counts.sum() == (w[:, 4:] != tok.eos_token_id).sum()
    assert np.load(ex / "self_sim.npy").shape == (4, V)
    assert len(json.loads((ex / "self_sim_baseline.json").read_text())) == 4
    # the v0 extras are all still there
    for f in ("embed.npy", "unembed.npy", "final_ln.npz", "labels.npy", "freq_ranks.npy",
              "freq_bins.npy", "tokens.json", "layer_names.json"):
        assert (ex / f).exists(), f


def test_stage_normalize_corpus_run_centers_acts_on_seen_rows_and_unembed_on_all(corpus_run):
    from token_drift.normalize import normalize_layer

    c, rd = corpus_run
    ex = rd / "extract"
    counts = np.load(ex / "counts.npy")
    counts[10:20] = 0
    np.save(ex / "counts.npy", counts)
    cli.stage_normalize(c)
    norm = np.load(rd / "normalize" / "acts_norm.npy").astype(np.float32)
    kw = dict(center=True, unit_norm=True, drop_top_pcs=0)
    acts, unembed = np.load(ex / "acts.npy"), np.load(ex / "unembed.npy")
    np.testing.assert_allclose(norm[1], normalize_layer(acts[1], fit_rows=counts > 0, **kw), atol=2e-3)
    np.testing.assert_allclose(norm[-1], normalize_layer(unembed, **kw), atol=2e-3)


def test_stage_normalize_raw_mean_source_reads_acts_rawmean(corpus_run):
    from token_drift.normalize import normalize_layer

    c, rd = corpus_run
    c["normalize"]["source"] = "raw_mean"
    cli.stage_normalize(c)
    norm = np.load(rd / "normalize" / "acts_norm.npy").astype(np.float32)
    raw = np.load(rd / "extract" / "acts_rawmean.npy")
    ref = normalize_layer(raw[2], center=True, unit_norm=True, drop_top_pcs=0)
    np.testing.assert_allclose(norm[2], ref, atol=2e-3)


def test_stage_normalize_raw_mean_on_a_vocab_run_fails_fast(fake_extract):
    c, _ = fake_extract
    c["normalize"]["source"] = "raw_mean"
    with pytest.raises(ValueError, match="corpus"):
        cli.stage_normalize(c)


def test_stage_metrics_on_a_corpus_run(corpus_run):
    c, rd = corpus_run
    cli.stage_normalize(c)
    cli.stage_metrics(c)
    m = json.loads((rd / "metrics" / "metrics.json").read_text())
    counts = np.load(rd / "extract" / "counts.npy")
    assert all(counts[i] >= 20 for i in m["subsample_idx"])
    assert m["eligible_n"] == int((counts >= 20).sum())
    assert m["freq_bins_source"] == "corpus"
    assert np.asarray(m["knn_change_by_freq"]).shape == (L + 2, cli.N_CORPUS_BINS)
    assert np.asarray(m["knn_change_by_merge_rank"]).shape == (L + 2, cli.N_FREQ_BINS + 1)
    assert len(m["self_sim"]) == L + 2 and len(m["self_sim_adjusted"]) == L + 2


def test_stage_viz_trajectory_groups_one_png_each_with_counts(corpus_run):
    c, rd = corpus_run
    c["viz"]["trajectory_groups"] = {"common": [" the", "7"], "mixed": ["The", "not-a-token"]}
    counts = np.load(rd / "extract" / "counts.npy")
    counts[2] = 0  # "The" never occurs in this fake corpus
    counts[3] = 3  # "ing" is below min_count but still drawn if asked for
    np.save(rd / "extract" / "counts.npy", counts)
    c["viz"]["trajectory_groups"]["rare"] = ["ing"]
    cli.stage_normalize(c)
    cli.stage_metrics(c)
    cli.stage_viz(c)
    vd = rd / "viz"
    assert (vd / "trajectories_common.png").exists() and (vd / "trajectories_rare.png").exists()
    assert not (vd / "trajectories_mixed.png").exists()  # nothing left to draw in that group
    idx = np.load(vd / "viz_idx.npy")
    assert 3 in idx and 2 not in idx


# ---------- v1 config tests ----------

CONFIGS = Path(__file__).parents[1] / "configs"


def test_v1_configs_differ_only_where_they_should():
    load = lambda n: yaml.safe_load((CONFIGS / n).read_text())  # noqa: E731
    base, shuf, rand = load("pythia70m_corpus.yaml"), load("pythia70m_corpus_shuf.yaml"), load("random_init_corpus.yaml")
    assert base["corpus"]["shuffle"] == "none" and base["extract"]["mode"] == "corpus"
    s = copy.deepcopy(shuf)
    s["run_name"], s["corpus"]["shuffle"] = base["run_name"], "none"
    assert s == base and shuf["corpus"]["shuffle"] == "window"
    r = copy.deepcopy(rand)
    r["run_name"], r["random_init"] = base["run_name"], False
    assert r == base and rand["random_init"] is True


def test_all_runs_the_whole_v1_pipeline_on_a_tiny_corpus(tmp_path, monkeypatch):
    tok = tiny_tokenizer()
    model = tiny_model(tok, window=16)
    monkeypatch.setattr(cli, "_load_tokenizer", lambda name: tok)
    monkeypatch.setattr(cli.ex, "build_model", lambda name, **kw: (model, tok))
    pq.write_table(pa.table({"text": tiny_docs()}), tmp_path / "docs.parquet")
    c = {
        "run_name": "tiny_v1", "model": "tiny", "random_init": False, "seed": 0, "device": "cpu",
        "corpus": {"source": str(tmp_path / "docs.parquet"), "text_field": "text", "max_tokens": None,
                   "window": 16, "shuffle": "none"},
        "extract": {"mode": "corpus", "batch_size": 8, "min_context": 4, "dtype_on_disk": "float16"},
        "normalize": {"source": "unit_mean", "center": True, "unit_norm": True, "drop_top_pcs": 0},
        "metrics": {"min_count": 5, "subsample": 1000, "knn_k": 3, "kmeans_k": 3, "intrinsic_dim": False},
        "viz": {"method": "stacked_umap", "n_neighbors": 5, "min_dist": 0.1, "color_by": "category",
                "trajectory_tokens": [" the", " in"]},
        "out_dir": str(tmp_path / "runs"),
    }
    p = tmp_path / "tiny_v1.yaml"
    p.write_text(yaml.safe_dump(c))
    cli.all(p)
    rd = tmp_path / "runs" / "tiny_v1"
    assert sorted(x.name for x in rd.iterdir()) == ["config.yaml", "corpus", "extract", "metrics", "normalize", "viz"]
    m = json.loads((rd / "metrics" / "metrics.json").read_text())
    counts = np.load(rd / "extract" / "counts.npy")
    assert m["eligible_n"] == int((counts >= 5).sum())
    assert m["self_sim"][0] == pytest.approx(1.0, abs=1e-3)  # frame 0 = the embedding, every time
    assert (rd / "viz" / "flipbook.gif").exists()


def test_all_on_a_v0_config_skips_the_corpus_stage(cfg, monkeypatch):
    calls = []
    for s in ("corpus", "extract", "normalize", "metrics", "viz"):
        monkeypatch.setattr(cli, f"stage_{s}", lambda c, s=s: calls.append(s))
    cli.all(cfg)
    assert calls == ["extract", "normalize", "metrics", "viz"]


def test_v0v1_compares_on_v1s_tokens_and_leaves_v0_alone(fake_extract, corpus_run):
    c0, rd0 = fake_extract
    c1, rd1 = corpus_run
    cli.stage_normalize(c0)
    cli.stage_normalize(c1)
    cli.stage_metrics(c1)
    out = cli.run_v0v1(rd0, rd1)
    assert out == rd0.parent / "v0v1_fake"
    r = json.loads((out / "v0v1.json").read_text())
    m1 = json.loads((rd1 / "metrics" / "metrics.json").read_text())
    # frame 0 (embedding) and the unembed are identical in both runs; the blocks were perturbed
    assert r["cross_overlap"][0] == pytest.approx(1.0) and r["cross_overlap"][-1] == pytest.approx(1.0)
    assert r["cross_overlap"][1] < 0.9
    assert r["v0"]["subsample_idx"] == m1["subsample_idx"]
    assert r["v0"]["freq_bins_source"] == "corpus"  # same bins as v1, so the Voita panels compare
    assert (out / "v0v1.png").exists() and (out / "cross_overlap.png").exists()
    assert not (rd0 / "metrics").exists()  # v0's own metrics are never touched
    # both runs use the same eligible mask and frequency bins
    assert r["v0"]["eligible_n"] == r["v1"]["eligible_n"]
    assert r["v0"]["anisotropy"] is not None and len(r["v0"]["anisotropy"]) == len(r["v1"]["anisotropy"])


def test_v0v1_refuses_runs_that_dont_line_up(fake_extract, corpus_run):
    c0, rd0 = fake_extract
    c1, rd1 = corpus_run
    cli.stage_normalize(c0)
    cli.stage_normalize(c1)
    cli.stage_metrics(c1)
    np.save(rd0 / "normalize" / "acts_norm.npy", np.load(rd0 / "normalize" / "acts_norm.npy")[:, :-1])
    with pytest.raises(ValueError, match="can't pair"):
        cli.run_v0v1(rd0, rd1)
