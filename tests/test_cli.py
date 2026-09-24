"""Stage wiring. Extract needs a real model so it's covered by test_extract; here we
fake its outputs on disk and run the downstream stages through the same code the CLI uses."""
import json

import numpy as np
import pytest
import yaml

from token_drift import cli
from token_drift.labels import categorize_all

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
