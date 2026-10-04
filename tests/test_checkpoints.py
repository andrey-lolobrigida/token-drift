"""v2 checkpoint runs. A tiny GPT-NeoX stands in for every revision, nudged further from its
init the later the step, so drift and with-final metrics have something to measure. No network."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from token_drift import cli
from conftest import tiny_model, tiny_tokenizer

REVS = ["step0", "step1", "step8"]


def fake_build(tok):
    def build(name, *, random_init, seed, device, revision=None):
        model = tiny_model(tok)  # tiny_model seeds 0, so this is the same "init" every call
        step = cli.step_of(revision)
        g = torch.Generator().manual_seed(step)
        with torch.no_grad():  # "training": later steps sit further from the init
            for p in model.parameters():
                p.add_(0.05 * float(np.log1p(step)) * torch.randn(p.shape, generator=g))
        return model.eval(), tok
    return build


def ckpt_cfg(tmp_path, monkeypatch, revisions=REVS, **timeline):
    tok = tiny_tokenizer()
    build = fake_build(tok)
    calls = []

    def counted(name, **kw):
        calls.append(kw.get("revision"))
        return build(name, **kw)

    monkeypatch.setattr(cli.ex, "build_model", counted)
    monkeypatch.setattr(cli.ex, "missing_revisions", lambda name, revs: [])
    c = {
        "run_name": "tiny_ckpt", "model": "tiny", "random_init": False, "seed": 0, "device": "cpu",
        "extract": {"batch_size": 16, "dtype_on_disk": "float16"},
        "normalize": {"center": True, "unit_norm": True, "drop_top_pcs": 0},
        "metrics": {"subsample": 1000, "knn_k": 3, "kmeans_k": 3, "intrinsic_dim": False},
        "viz": {"method": "stacked_umap", "n_neighbors": 5, "min_dist": 0.1, "color_by": "category",
                "trajectory_tokens": [" the", " in", " notatoken"]},  # the last one isn't a token: skipped
        "out_dir": str(tmp_path / "runs"),
        "revisions": list(revisions),
        "timeline": {"frames": [0, 1, "unembed"], "keep_acts": False, **timeline},
    }
    p = tmp_path / "tiny_ckpt.yaml"
    p.write_text(yaml.safe_dump(c))
    return cli.load_config(p), calls


def test_revision_cfg_points_at_the_step_folder_and_drops_the_loop_keys(tmp_path, monkeypatch):
    c, _ = ckpt_cfg(tmp_path, monkeypatch)
    sub = cli._revision_cfg(c, "step8")
    assert cli.run_dir(sub) == cli.run_dir(c) / "step0000008"
    assert sub["revision"] == "step8" and "revisions" not in sub and "timeline" not in sub
    assert c["revisions"] == REVS and "revision" not in c  # the parent config is left alone


def test_resolve_frames_maps_unembed_to_the_last_frame_and_refuses_junk():
    names = cli.layer_names(6)
    assert cli.resolve_frames([0, 3, "unembed"], names) == [0, 3, 8]
    for bad in ([9], [-1], ["L3"], [True]):
        with pytest.raises(ValueError, match="timeline.frames"):
            cli.resolve_frames(bad, names)


def test_timeline_rows_add_trajectory_tokens_and_skip_non_tokens():
    tokens = ["a", " the", "b", " in", "c"]
    assert cli._timeline_rows([0, 4], tokens, [" the", " nope"]).tolist() == [0, 1, 4]


def test_checkpoints_keep_only_the_small_outputs_per_revision(tmp_path, monkeypatch):
    c, calls = ckpt_cfg(tmp_path, monkeypatch)
    rd = cli.stage_checkpoints(c)
    assert calls == REVS
    assert sorted(p.name for p in rd.iterdir()) == ["config.yaml", "step0000000", "step0000001", "step0000008"]
    for r in REVS:
        sd = rd / cli.step_dir_name(r)
        assert sorted(p.name for p in sd.iterdir()) == ["config.yaml", "frames", "metrics", "weights"]
        V = len(json.loads((sd / "frames" / "tokens.json").read_text()))
        idx = np.load(sd / "frames" / "idx.npy")
        fr = np.load(sd / "frames" / "frames.npy")
        assert fr.shape == (3, len(idx), 16) and fr.dtype == np.float16
        assert np.load(sd / "weights" / "embed.npy").shape == (V, 16)
        assert np.load(sd / "weights" / "unembed.npy").shape == (V, 16)
        assert json.loads((sd / "frames" / "frames.json").read_text()) == {
            "ids": [0, 1, 4], "names": ["L0 (embed)", "L1", "unembed"]}


def test_frames_are_the_normalized_rows_of_the_chosen_frames(tmp_path, monkeypatch):
    c, _ = ckpt_cfg(tmp_path, monkeypatch, keep_acts=True)
    sd = cli.stage_checkpoints(c) / "step0000008"
    assert (sd / "extract").exists() and (sd / "normalize").exists()  # keep_acts: nothing deleted
    norm = np.load(sd / "normalize" / "acts_norm.npy")
    idx = np.load(sd / "frames" / "idx.npy")
    fr = np.load(sd / "frames" / "frames.npy")
    np.testing.assert_array_equal(fr[0], norm[0][idx])
    np.testing.assert_array_equal(fr[2], norm[-1][idx])  # "unembed" = the last frame
    np.testing.assert_array_equal(np.load(sd / "weights" / "embed.npy"), np.load(sd / "extract" / "embed.npy"))


def test_rerun_skips_finished_revisions_and_redoes_a_half_finished_one(tmp_path, monkeypatch):
    c, calls = ckpt_cfg(tmp_path, monkeypatch)
    rd = cli.stage_checkpoints(c)
    # what a crash after metrics but before the frames copy leaves behind
    (rd / "step0000001" / "frames" / "idx.npy").unlink()
    calls.clear()
    cli.stage_checkpoints(c)
    assert calls == ["step1"]


def test_adding_a_revision_later_runs_only_the_new_one(tmp_path, monkeypatch):
    c, _ = ckpt_cfg(tmp_path, monkeypatch)
    cli.stage_checkpoints(c)
    c2, calls2 = ckpt_cfg(tmp_path, monkeypatch, revisions=["step0", "step1", "step4", "step8"])
    cli.stage_checkpoints(c2)
    assert calls2 == ["step4"]


def test_missing_revision_stops_before_any_download(tmp_path, monkeypatch):
    c, calls = ckpt_cfg(tmp_path, monkeypatch)
    monkeypatch.setattr(cli.ex, "missing_revisions", lambda name, revs: ["step8"])
    with pytest.raises(ValueError, match="step8"):
        cli.stage_checkpoints(c)
    assert calls == []


def test_a_step_folders_config_reloads_as_a_plain_run(tmp_path, monkeypatch):
    c, _ = ckpt_cfg(tmp_path, monkeypatch, keep_acts=True)
    rd = cli.stage_checkpoints(c)
    sc = cli.load_config(rd / "step0000001" / "config.yaml")
    assert cli.run_dir(sc) == rd / "step0000001" and sc["revision"] == "step1"
    assert "revisions" not in sc
    cli.stage_metrics(sc)  # copies its own config.yaml onto itself: must not crash


def test_all_on_a_checkpoint_config_runs_the_checkpoint_loop(tmp_path, monkeypatch):
    c, _ = ckpt_cfg(tmp_path, monkeypatch)
    seen = []
    for s in ("corpus", "extract", "normalize", "metrics", "viz", "checkpoints"):
        monkeypatch.setattr(cli, f"stage_{s}", lambda c, s=s: seen.append(s))
    cli.all(Path(c["_config_path"]))
    assert seen == ["checkpoints"]
