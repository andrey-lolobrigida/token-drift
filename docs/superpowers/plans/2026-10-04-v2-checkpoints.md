# v2: Pythia-70m training checkpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the v0 pipeline on 10 log-spaced Pythia-70m training checkpoints and measure when
layer-0 surface-form clustering and the stable middle block show up, how "finished" each frame's
geometry is at each step, and how far embed vs unembed rows have moved from init.

**Architecture:** A config with `revisions:` makes `all` loop over checkpoints. Each revision runs
the existing extract -> normalize -> metrics in its own step folder
(`runs/pythia70m_ckpt/step0000064/`), keeps a small `frames/` + `weights/` copy, and deletes the
big `extract/` + `normalize/`. A new `timeline` stage reads every step folder and writes
`timeline.json`, `timeline.png` (per-checkpoint curves + similarity to the final model) and
`drift.png` (row drift from init by frequency bin). A `flipbooks` stage fits one AlignedUMAP per
timeline frame, where the flipbook pages are checkpoints instead of layers.

**Tech Stack:** Python 3.12, uv, torch + transformers (GPT-NeoX), huggingface_hub
(`list_repo_refs`, already installed as a transformers dependency), numpy, scikit-learn, umap-learn,
matplotlib, typer, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-04-v2-checkpoints-design.md`. Read it first; this plan
argues from it. Also `CLAUDE.md` (how we work).

**Where this plan departs from the spec (on purpose):**
- The spec says `load_model(..., revision=None)`. The real function is `extract.build_model`; it gets
  the `revision` keyword.
- `step0` must be in `revisions:`, because row drift is measured from it. `load_config` refuses
  otherwise.
- A revision counts as finished when `metrics/metrics.json` **and** `frames/idx.npy` exist. The spec
  only checks metrics.json, but then a crash between metrics and the frames copy would be skipped
  forever. `idx.npy` is written last.
- The Hub check only covers revisions that aren't finished yet, so a rerun works offline.
- `frames/` also gets small copies of `labels.npy`, `freq_bins.npy`, `tokens.json`,
  `layer_names.json` and a `frames.json` (which frames are stored). Without them, `timeline` would
  need an `extract/` that has been deleted by then.
- Each step folder gets its own `config.yaml` (the plain v0 config for that one revision), so
  `token-drift metrics --config runs/pythia70m_ckpt/step0000064/config.yaml` works by itself.
- Flipbooks are their own stage (`stage_flipbooks`). `token-drift timeline --skip-flipbooks` skips
  the 1-2.5 h of AlignedUMAP.
- `timeline.json` gets a `half_way` table: the first step where a curve has covered half the
  distance from its step-0 value to its final value. It's FINDINGS 13's "when" number, fixed
  before anyone looks at a curve.
- `drift.png` draws a faint line at float16 resolution (2^-11 ~ 4.9e-4). Pythia's checkpoints are
  stored in float16 on the Hub (`"torch_dtype": "float16"`), so a relative row change below that
  is at the edge of what the checkpoint can even record. Saving `weights/` in float16 loses
  nothing beyond that.

## Global Constraints

- Run everything through uv: `uv run pytest -q`, `uv run token-drift ...`.
- v0 mode only: `revisions:` together with `corpus:`, a probe block, `extract.mode` other than
  vocab, or `random_init: true` is refused by `load_config`.
- Revision names are `step<N>` (Pythia's Hub branches). Step folders are `step%07d`
  (`step64` -> `step0000064`) so `ls` sorts them in training order.
- Revisions are always processed and plotted in step order, whatever order the yaml lists them in.
  "Final" means the largest step, and drift is measured from `step0`.
- Tokenizer always comes from the model's main branch (no `revision=`). It's the same at every step.
- float16 on disk, upcast to float32 per frame / per matrix when computing.
- Seed everything from `cfg["seed"]`. Every revision must use the same metrics subsample and the
  same frame rows; `timeline` refuses otherwise and names the bad revision.
- Functions take arrays and return arrays (`timeline.py`). Only `cli.py` reads another stage's files.
- Tests are tiny and fast, with no network: `build_model` and `missing_revisions` are monkeypatched.
- `keep_acts: false` deletes only the step folder's own `extract/` and `normalize/`, and only after
  `frames/` + `weights/` are written.
- Comments and docs: casual tone, plus a one-line *why* on every non-obvious choice (CLAUDE.md).
- One commit per task, with `uv run pytest -q` green before each. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **Crash between metrics and the frames copy.** `metrics.json` exists but `frames/idx.npy`
   doesn't. A rerun must redo that revision, not skip it. Pinned by
   `test_rerun_skips_finished_revisions_and_redoes_a_half_finished_one` (Task 2).
2. **Andrey fills a gap later** (adds `step4` between `step1` and `step8`). A rerun must only run
   the new revision, and timeline must accept it (same seed -> same tokens). Pinned by
   `test_adding_a_revision_later_runs_only_the_new_one` (Task 2) and the timeline tests (Task 4).
3. **Revisions listed out of step order in the yaml.** Folders, curves and "final" must still go by
   step. Pinned by `test_timeline_orders_revisions_by_step_whatever_the_yaml_order` (Task 4).
4. **A trajectory token that isn't a single token** (`" notatoken"`). It must be skipped with a
   message, never crash, and leave the frame rows identical across revisions. Pinned by
   `test_timeline_rows_add_trajectory_tokens_and_skip_non_tokens` (Task 2) and by every
   checkpoint test (the test config includes one).
5. **`timeline.frames` edited after the revisions ran** (e.g. Andrey wants L5 too). Old step
   folders hold the old frames. Timeline must refuse loudly, not plot old frames under new names.
   Pinned by `test_timeline_refuses_when_timeline_frames_changed_since_the_run` (Task 4).

---

### Task 1: `revision` in `build_model`, Hub check, config validation, checkpoint config

**Files:**
- Modify: `src/token_drift/extract.py` (`build_model`, new `missing_revisions`)
- Modify: `src/token_drift/cli.py` (`load_config`, new `step_of`, `step_dir_name`, `_check_revisions`; `stage_extract` passes `revision`)
- Create: `configs/pythia70m_ckpt.yaml`
- Test: `tests/test_extract.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `extract.build_model(name, *, random_init: bool, seed: int, device: str, revision: str | None = None) -> (model, tokenizer)`
  - `extract.missing_revisions(name: str, revisions: list[str]) -> list[str]` (the ones not on the Hub, input order)
  - `cli.step_of(rev: str) -> int` (`"step64"` -> 64; ValueError if not `step<N>`)
  - `cli.step_dir_name(rev: str) -> str` (`"step64"` -> `"step0000064"`)
  - `cli.load_config` refuses bad checkpoint configs (see tests)
  - `cli.DEFAULT_TIMELINE_FRAMES = [0, "unembed"]`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_extract.py`. Put the import next to the existing ones at the top:

```python
from token_drift import extract as ex
```

and the tests at the end:

```python
def test_build_model_passes_revision_to_the_model_not_the_tokenizer(monkeypatch, model):
    seen = {}

    def fake_tok(name, **kw):
        seen["tok"] = kw
        return "tok"

    def fake_model(name, **kw):
        seen["model"] = kw
        return model

    monkeypatch.setattr(ex.AutoTokenizer, "from_pretrained", fake_tok)
    monkeypatch.setattr(ex.AutoModelForCausalLM, "from_pretrained", fake_model)
    ex.build_model("EleutherAI/pythia-70m", random_init=False, seed=0, device="cpu", revision="step64")
    assert seen["model"] == {"revision": "step64"}
    assert "revision" not in seen["tok"]  # one tokenizer for every step: the final one


def test_build_model_refuses_a_revision_with_random_init():
    with pytest.raises(ValueError, match="random_init"):
        ex.build_model("x", random_init=True, seed=0, device="cpu", revision="step64")


def test_missing_revisions_asks_the_hub_once_and_lists_absent_branches(monkeypatch):
    from types import SimpleNamespace as NS

    import huggingface_hub

    calls = []

    def fake_refs(name):
        calls.append(name)
        return NS(branches=[NS(name="main"), NS(name="step0"), NS(name="step1")])

    monkeypatch.setattr(huggingface_hub, "list_repo_refs", fake_refs)
    assert ex.missing_revisions("EleutherAI/pythia-70m", ["step0", "step7", "step1", "step9"]) == ["step7", "step9"]
    assert calls == ["EleutherAI/pythia-70m"]
```

Add to the end of `tests/test_cli.py` (`CONFIGS` is already defined further up the file):

```python
def test_step_dir_name_zero_pads_so_folders_sort_in_step_order():
    revs = ["step143000", "step8", "step0", "step1000", "step64"]
    assert cli.step_dir_name("step64") == "step0000064"
    assert cli.step_of("step143000") == 143000
    # plain string sort of the folder names == numeric sort of the steps
    assert sorted(cli.step_dir_name(r) for r in revs) == [cli.step_dir_name(r) for r in sorted(revs, key=cli.step_of)]


@pytest.mark.parametrize("change, msg", [
    ({"corpus": {"source": "x"}}, "v0 mode"),
    ({"extract": {"mode": "probe"}, "probe": {}}, "v0 mode"),
    ({"random_init": True}, "random_init"),
    ({"revisions": ["step0", "final"]}, "step<N>"),
    ({"revisions": ["step0", "step8", "step08"]}, "twice"),
    ({"revisions": ["step1", "step8"]}, "step0"),
    ({"timeline": {"frames": [0, "L3"]}}, "timeline.frames"),
])
def test_load_config_refuses_bad_checkpoint_configs(cfg, tmp_path, change, msg):
    c = yaml.safe_load(cfg.read_text())
    c["revisions"] = ["step0", "step8"]
    c.update(change)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match=msg):
        cli.load_config(p)


def test_checkpoint_config_is_the_v0_config_plus_revisions():
    base = yaml.safe_load((CONFIGS / "pythia70m.yaml").read_text())
    ck = yaml.safe_load((CONFIGS / "pythia70m_ckpt.yaml").read_text())
    assert ck["revisions"][0] == "step0" and ck["revisions"][-1] == "step143000" and len(ck["revisions"]) == 10
    assert ck["timeline"] == {"frames": [0, 3, "unembed"], "keep_acts": False}
    rest = {k: v for k, v in ck.items() if k not in ("revisions", "timeline")}
    assert rest == {**base, "run_name": "pythia70m_ckpt"}
    cli.load_config(CONFIGS / "pythia70m_ckpt.yaml")  # passes its own validation
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_extract.py tests/test_cli.py -q -k "revision or step_dir or checkpoint"`
Expected: FAIL. `build_model() got an unexpected keyword argument 'revision'`, `module 'token_drift.extract' has no attribute 'missing_revisions'`, `module 'token_drift.cli' has no attribute 'step_dir_name'`, `pythia70m_ckpt.yaml` not found.

- [ ] **Step 3: Implement**

In `src/token_drift/extract.py`, replace `build_model` and add `missing_revisions` right after it:

```python
def build_model(name: str, *, random_init: bool, seed: int, device: str, revision: str | None = None):
    """Load the trained model, or the same architecture with fresh random weights.

    `revision` = a Hub branch, i.e. a training checkpoint (Pythia: step0 .. step143000).
    None = main = the finished model.
    """
    if revision is not None and random_init:
        raise ValueError(f"revision={revision!r} with random_init: true: random init never loads any weights")
    torch.manual_seed(seed)
    # always main's tokenizer: Pythia's vocab is the same at every checkpoint
    tokenizer = AutoTokenizer.from_pretrained(name)
    if random_init:
        # from_config runs HF's init (normal, std=initializer_range) instead of loading
        # weights. Seeded above, so the control is reproducible.
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name))
    elif revision is not None:
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision)
    else:
        model = AutoModelForCausalLM.from_pretrained(name)
    return model.to(device).eval(), tokenizer


def missing_revisions(name: str, revisions: list[str]) -> list[str]:
    """The revisions (Hub branches) `name` doesn't have. One API call, no downloads, so a typo
    in the config fails in a second instead of after three checkpoints' worth of work."""
    from huggingface_hub import list_repo_refs  # only checkpoint runs need the network for this

    have = {b.name for b in list_repo_refs(name).branches}
    return [r for r in revisions if r not in have]
```

(The `elif`/`else` split keeps the no-revision call exactly as before, so nothing that
monkeypatches `from_pretrained(name)` elsewhere sees a new keyword.)

In `src/token_drift/cli.py`, add `import re` to the stdlib imports. Then put this block after
`_mode`, before `load_config`:

```python
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
    for f in (cfg.get("timeline") or {}).get("frames", DEFAULT_TIMELINE_FRAMES):
        if f != "unembed" and (isinstance(f, bool) or not isinstance(f, int)):
            raise ValueError(f"{path}: timeline.frames entries are frame indices or 'unembed', got {f!r}")
```

In `load_config`, just before `return cfg`:

```python
    if "revisions" in cfg:
        _check_revisions(cfg, path)
```

In `stage_extract`, pass the revision and say it in the log line:

```python
    typer.echo(f"[extract] {cfg['model']} random_init={cfg['random_init']} revision={cfg.get('revision')} "
               f"mode={mode} on {device}")
    t0 = time.time()
    model, tok = ex.build_model(
        cfg["model"], random_init=cfg["random_init"], seed=cfg["seed"], device=device,
        revision=cfg.get("revision"),
    )
```

Create `configs/pythia70m_ckpt.yaml` (a copy of `pythia70m.yaml` with a new run_name, plus two
blocks at the end):

```yaml
# v2: the v0 pipeline on Pythia-70m training checkpoints. Same settings as pythia70m.yaml,
# so step143000 should reproduce runs/pythia70m (sanity check 1 in the v2 spec).
run_name: pythia70m_ckpt
model: EleutherAI/pythia-70m
random_init: false
seed: 0
device: auto            # cpu | cuda | mps | auto

extract:
  batch_size: 512
  dtype_on_disk: float16

normalize:
  center: true
  unit_norm: true
  drop_top_pcs: 0       # try 0 and 2

metrics:
  subsample: 10000      # tokens used for kNN / silhouette / k-means
  knn_k: 10
  kmeans_k: 20
  intrinsic_dim: false

viz:
  method: aligned_umap  # aligned_umap | stacked_umap
  n_neighbors: 15
  min_dist: 0.1
  color_by: category    # category | freq_bin
  trajectory_tokens: [" bank", " river", " money", " the", " a", " an", "1", "2", " 1", ".", ",", "ing", " ing"]

out_dir: runs

# log-spaced: most of the change is expected early, and even spacing would mostly sample the slow tail
revisions: [step0, step1, step8, step64, step512, step1000, step4000, step16000, step64000, step143000]
timeline:
  frames: [0, 3, unembed]   # embed, middle of the stable block, unembed: training-time flipbooks + with-final curves
  keep_acts: false          # delete each revision's extract/ + normalize/ once frames/ + weights/ are copied
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_extract.py tests/test_cli.py -q`
Expected: PASS (all old tests too).

- [ ] **Step 5: Commit**

```bash
git add src/token_drift/extract.py src/token_drift/cli.py configs/pythia70m_ckpt.yaml tests/test_extract.py tests/test_cli.py
git commit -m "v2: revision in build_model, Hub check, checkpoint config validation

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The checkpoint loop (sub-config, frames/weights copy, cleanup, resume)

**Files:**
- Modify: `src/token_drift/cli.py` (`_prepare_run_dir`; new `_revision_cfg`, `resolve_frames`, `_timeline_rows`, `_revision_done`, `_save_revision_extras`, `stage_checkpoints`; `all`)
- Create: `tests/test_checkpoints.py`

**Interfaces:**
- Consumes (Task 1): `step_of`, `step_dir_name`, `DEFAULT_TIMELINE_FRAMES`, `ex.build_model(..., revision=)`, `ex.missing_revisions`.
- Produces:
  - `cli._revision_cfg(cfg: dict, rev: str) -> dict` (plain v0 config; `run_dir` = the step folder; has `revision`, no `revisions`/`timeline`)
  - `cli.resolve_frames(spec: list, names: list[str]) -> list[int]` (`"unembed"` -> `len(names)-1`)
  - `cli._timeline_rows(subsample_idx, tokens: list[str], trajectory_tokens: list[str]) -> np.ndarray` (sorted int64)
  - `cli._revision_done(sd: Path) -> bool`
  - `cli.stage_checkpoints(cfg: dict) -> Path` (the parent run dir)
  - Per step folder: `config.yaml`, `metrics/`, `weights/{embed,unembed}.npy`, `frames/{frames.npy, idx.npy, frames.json, labels.npy, freq_bins.npy, tokens.json, layer_names.json}`.
    `frames.json` = `{"ids": [frame indices], "names": [frame names]}`. `frames.npy` = `(len(ids), len(idx), d)` float16.
  - Test helpers in `tests/test_checkpoints.py`: `REVS`, `fake_build(tok)`, `ckpt_cfg(tmp_path, monkeypatch, revisions=REVS, **timeline) -> (cfg, calls)` where `calls` is the list of revisions `build_model` was called with.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_checkpoints.py`:

```python
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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_checkpoints.py -q`
Expected: FAIL with `module 'token_drift.cli' has no attribute '_revision_cfg'` (and similar for the rest).

- [ ] **Step 3: Implement**

In `src/token_drift/cli.py`, add `import copy` to the stdlib imports.

Replace `_prepare_run_dir`:

```python
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
```

Add a new section after `stage_metrics` (before `_stage_viz_probe`):

```python
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
    (so the flipbooks can show them, like v0's viz does)."""
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
```

In `all`, right after `cfg = load_config(config)`:

```python
    if "revisions" in cfg:  # v2: the v0 pipeline once per training checkpoint
        stage_checkpoints(cfg)
        return
```

Also update the module docstring's first line to list the new verbs:
`"""`token-drift corpus|probe_corpus|extract|normalize|metrics|viz|occ|all --config configs/x.yaml`, plus `compare`, `ksweep`, `v0v1`, `timeline`.`

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest -q`
Expected: PASS, the whole suite (the `_prepare_run_dir` change touches every stage).

- [ ] **Step 5: Commit**

```bash
git add src/token_drift/cli.py tests/test_checkpoints.py
git commit -m "v2: checkpoint loop in all (step folders, frames/weights copy, resume)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `timeline.py` math

**Files:**
- Create: `src/token_drift/timeline.py`
- Test: `tests/test_timeline.py`

**Interfaces:**
- Consumes: `metrics.knn_indices`, `metrics.knn_overlap`, `metrics.linear_cka` (existing).
- Produces:
  - `timeline.overlap_with_final(x: np.ndarray, x_final: np.ndarray, k: int) -> float`
  - `timeline.cka_with_final(x: np.ndarray, x_final: np.ndarray) -> float`
  - `timeline.row_drift(w_t: np.ndarray, w_0: np.ndarray) -> np.ndarray` (float32, one value per row, NaN where the init row is zero)
  - `timeline.median_by_bin(values: np.ndarray, bins: np.ndarray, n_bins: int) -> list[float]` (NaN for empty bins; NaN values and bins outside 0..n_bins-1 ignored)
  - `timeline.half_way_step(values, steps: list[int], frac: float = 0.5) -> int | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_timeline.py`:

```python
"""v2 timeline maths on toy arrays."""
import numpy as np
import pytest

from token_drift import timeline as tl


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_identical_geometry_is_fully_finished():
    x = _unit(np.random.default_rng(0).normal(size=(60, 8)))
    assert tl.overlap_with_final(x, x, k=5) == pytest.approx(1.0)
    assert tl.cka_with_final(x, x) == pytest.approx(1.0)


def test_a_rotation_counts_as_finished_too():
    # both metrics only care about the shape of the cloud, not which way it faces
    rng = np.random.default_rng(1)
    x = _unit(rng.normal(size=(60, 8)))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert tl.overlap_with_final(x @ q, x, k=5) == pytest.approx(1.0)
    assert tl.cka_with_final(x @ q, x) == pytest.approx(1.0)


def test_unrelated_geometry_is_far_from_finished():
    rng = np.random.default_rng(0)
    a, b = _unit(rng.normal(size=(300, 16))), _unit(rng.normal(size=(300, 16)))
    assert tl.overlap_with_final(a, b, k=5) < 0.1  # chance is ~0.01
    assert tl.cka_with_final(a, b) < 0.3


def test_row_drift_is_zero_for_untouched_rows_and_relative_for_moved_ones():
    rng = np.random.default_rng(0)
    w0 = rng.normal(size=(10, 4)).astype(np.float32)
    wt = w0.copy()
    wt[3] = w0[3] * 1.5  # moved by half its own length
    wt[7] = 0            # shrunk to nothing: drift 1
    d = tl.row_drift(wt, w0)
    assert d.shape == (10,) and d.dtype == np.float32
    assert d[3] == pytest.approx(0.5, rel=1e-5) and d[7] == pytest.approx(1.0)
    assert (d[[0, 1, 2, 4, 5, 6, 8, 9]] == 0).all()


def test_row_drift_takes_float16_like_weights_on_disk():
    w0 = np.array([[3.0, 4.0]], dtype=np.float16)
    wt = np.array([[3.0, 4.5]], dtype=np.float16)
    assert tl.row_drift(wt, w0)[0] == pytest.approx(0.1, rel=1e-3)  # 0.5 / 5


def test_row_drift_of_a_zero_init_row_is_nan_not_inf():
    w0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    wt = np.array([[1.0, 0.0], [3.0, 4.0]])
    d = tl.row_drift(wt, w0)
    assert np.isnan(d[0]) and d[1] == 0


def test_median_by_bin_hand_checked():
    values = np.array([1.0, 5.0, 3.0, 10.0, 20.0, np.nan, 7.0])
    bins = np.array([0, 0, 0, 1, 1, 2, -1])  # -1 = no bin, ignored
    out = tl.median_by_bin(values, bins, n_bins=4)
    assert out[:2] == [3.0, 15.0]
    assert np.isnan(out[2]) and np.isnan(out[3])  # bin 2 holds only a NaN, bin 3 is empty


def test_half_way_step_rising_falling_and_flat():
    steps = [0, 1, 8, 64]
    assert tl.half_way_step([0.0, 0.1, 0.6, 1.0], steps) == 8
    assert tl.half_way_step([1.0, 0.8, 0.4, 0.0], steps) == 8  # falling curves count the same way
    assert tl.half_way_step([0.0, 0.5, 0.6, 1.0], steps) == 1  # exactly half counts as reached
    assert tl.half_way_step([0.3, 0.3, 0.3, 0.3], steps) is None  # nothing happened
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_timeline.py -q`
Expected: FAIL with `cannot import name 'timeline' from 'token_drift'`.

- [ ] **Step 3: Implement**

Create `src/token_drift/timeline.py`:

```python
"""v2: how training moves the vocab geometry. Arrays in, numbers out.

"with final" = how close a checkpoint's frame already is to the finished model's same frame.
kNN overlap asks it locally (does each token have its final neighbours yet?), CKA globally
(is the whole cloud in its final shape?). They can disagree, and that split is interesting
(cf. FINDINGS 11.2).
"""
from __future__ import annotations

import numpy as np

from token_drift.metrics import knn_indices, knn_overlap, linear_cka


def overlap_with_final(x: np.ndarray, x_final: np.ndarray, k: int) -> float:
    """Mean Jaccard of each row's k-NN set in `x` vs in `x_final`. Rows = the same tokens."""
    return knn_overlap(knn_indices(x, k), knn_indices(x_final, k))


def cka_with_final(x: np.ndarray, x_final: np.ndarray) -> float:
    return linear_cka(x, x_final)


def row_drift(w_t: np.ndarray, w_0: np.ndarray) -> np.ndarray:
    """||W_t[i] - W_0[i]|| / ||W_0[i]|| per row: how far each row moved, in units of its own
    starting length. Upcast first: the matrices sit on disk in float16."""
    a = np.asarray(w_t, dtype=np.float32)
    b = np.asarray(w_0, dtype=np.float32)
    num = np.linalg.norm(a - b, axis=1)
    den = np.linalg.norm(b, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan).astype(np.float32)  # a zero init row has no scale


def median_by_bin(values: np.ndarray, bins: np.ndarray, n_bins: int) -> list[float]:
    """Median of `values` inside each bin 0..n_bins-1. Median, not mean: a handful of rows that
    blow up (an attention-sink token, say) shouldn't drag a whole frequency bin."""
    values, bins = np.asarray(values, dtype=float), np.asarray(bins)
    out = []
    for b in range(n_bins):
        v = values[(bins == b) & ~np.isnan(values)]
        out.append(float(np.median(v)) if len(v) else float("nan"))
    return out


def half_way_step(values, steps: list[int], frac: float = 0.5) -> int | None:
    """First step where `values` has covered `frac` of the way from its first to its last value.

    The 'when did X appear' number for FINDINGS 13, fixed before looking at any curve so the
    step isn't picked by eye. None when the curve ends where it started (nothing to cover).
    """
    v = np.asarray(values, dtype=float)
    span = v[-1] - v[0]
    if not np.isfinite(span) or span == 0:
        return None
    reached = (v - v[0]) / span >= frac
    return int(steps[int(np.argmax(reached))])  # the last point always reaches, so argmax finds one
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest tests/test_timeline.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/token_drift/timeline.py tests/test_timeline.py
git commit -m "v2: timeline maths (overlap/CKA with final, row drift, median by bin, half-way step)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `timeline` stage + `plot_timeline` / `plot_drift`

**Files:**
- Modify: `src/token_drift/cli.py` (import `timeline as tl`; new `_Ckpt`, `_load_checkpoints`, `TIMELINE_CURVES`, `stage_timeline`, `timeline` command; `all`)
- Modify: `src/token_drift/viz.py` (new `_as_float`, `_step_axis`, `plot_timeline`, `plot_drift`)
- Test: `tests/test_checkpoints.py`, `tests/test_viz.py`

**Interfaces:**
- Consumes: Task 2's step-folder layout, `_revision_done`, `resolve_frames`, `step_of`, `step_dir_name`, `DEFAULT_TIMELINE_FRAMES`, and the test helpers `ckpt_cfg`, `REVS`. Task 3's `tl.*`. Existing `_nan_to_none`, `_load_json`, `stage_dir`.
- Produces:
  - `cli._load_checkpoints(cfg: dict) -> _Ckpt` with fields `revs: list[str]` (step order), `dirs: dict[str, Path]`, `idx: np.ndarray`, `sub_idx: list[int]`, `frame_ids: list[int]`, `frame_names: list[str]`, `frames: dict[str, np.ndarray]` (rev -> `(n_tl_frames, len(idx), d)` float16), `metrics: dict[str, dict]`
  - `cli.stage_timeline(cfg: dict) -> Path`, writing `<run>/timeline/{timeline.json, timeline.png, drift.png}`
  - `timeline.json` keys: `revisions, steps, final, layer_names, timeline_frames, timeline_frame_ids, knn_k, n_subsample, curves{TIMELINE_CURVES -> (steps x frames)}, with_final{frame name -> {knn: [...], cka: [...]}}, drift{embed|unembed -> (steps x bins)}, n_freq_bins, freq_bins_source, half_way{label -> step|None}, lowest_embed_drift[{id, token, drift, freq_bin}]`
  - `viz.plot_timeline(tl: dict, out_path, *, return_fig=False)`, `viz.plot_drift(tl: dict, out_path, *, return_fig=False)`, `viz._step_axis(ax, steps)`, `viz._as_float(rows) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_checkpoints.py`:

```python
@pytest.fixture
def finished(tmp_path, monkeypatch):
    c, _ = ckpt_cfg(tmp_path, monkeypatch)
    cli.stage_checkpoints(c)
    return c, cli.run_dir(c)


def test_timeline_writes_json_and_both_plots(finished):
    c, rd = finished
    cli.stage_timeline(c)
    out = rd / "timeline"
    t = json.loads((out / "timeline.json").read_text())
    assert t["steps"] == [0, 1, 8] and t["final"] == "step8"
    assert t["timeline_frames"] == ["L0 (embed)", "L1", "unembed"]
    for name in t["timeline_frames"]:
        wf = t["with_final"][name]
        assert len(wf["knn"]) == 3
        assert wf["knn"][-1] == pytest.approx(1.0) and wf["cka"][-1] == pytest.approx(1.0)  # final vs itself
    for w in ("embed", "unembed"):
        d = t["drift"][w]
        assert len(d) == 3 and len(d[0]) == t["n_freq_bins"]
        assert all(v in (0, None) for v in d[0])  # step0 vs itself; None = empty bin
        # our fake training nudges step8 further than step1, so every non-empty bin agrees
        assert all(b > a for a, b in zip(d[1], d[2]) if a is not None)
    assert len(t["curves"]["knn_purity"]) == 3 and len(t["curves"]["knn_consecutive"][0]) == 4
    assert "middle_block_overlap" not in t["half_way"]  # 2-layer toy: no middle block to speak of
    assert len(t["lowest_embed_drift"]) == 20
    assert (out / "timeline.png").exists() and (out / "drift.png").exists()


def test_timeline_orders_revisions_by_step_whatever_the_yaml_order(tmp_path, monkeypatch):
    c, _ = ckpt_cfg(tmp_path, monkeypatch, revisions=["step8", "step0", "step1"])
    cli.stage_checkpoints(c)
    cli.stage_timeline(c)
    t = json.loads((cli.run_dir(c) / "timeline" / "timeline.json").read_text())
    assert t["revisions"] == ["step0", "step1", "step8"] and t["final"] == "step8"


def test_timeline_refuses_while_a_revision_is_unfinished(finished):
    c, rd = finished
    shutil.rmtree(rd / "step0000001")
    with pytest.raises(ValueError, match="step1"):
        cli.stage_timeline(c)


def test_timeline_refuses_revisions_with_different_tokens(finished):
    c, rd = finished
    p = rd / "step0000001" / "frames" / "idx.npy"
    np.save(p, np.load(p)[:-1])
    with pytest.raises(ValueError, match="step1"):
        cli.stage_timeline(c)


def test_timeline_refuses_when_timeline_frames_changed_since_the_run(finished, tmp_path):
    c, rd = finished
    c2 = yaml.safe_load(Path(c["_config_path"]).read_text())
    c2["timeline"]["frames"] = [0, 2, "unembed"]
    p = tmp_path / "edited.yaml"
    p.write_text(yaml.safe_dump(c2))
    with pytest.raises(ValueError, match="timeline.frames"):
        cli.stage_timeline(cli.load_config(p))


def test_all_on_a_checkpoint_config_runs_checkpoints_then_timeline(tmp_path, monkeypatch):
    c, _ = ckpt_cfg(tmp_path, monkeypatch)
    seen = []
    for s in ("corpus", "extract", "normalize", "metrics", "viz", "checkpoints", "timeline"):
        monkeypatch.setattr(cli, f"stage_{s}", lambda c, s=s: seen.append(s))
    cli.all(Path(c["_config_path"]))
    assert seen == ["checkpoints", "timeline"]
```

Delete `test_all_on_a_checkpoint_config_runs_the_checkpoint_loop` from Task 2. The new test replaces it.

Add to the end of `tests/test_viz.py`:

```python
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
```

(`rng` is the fixture already defined in `tests/test_viz.py`.)

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_checkpoints.py tests/test_viz.py -q`
Expected: FAIL with `module 'token_drift.cli' has no attribute 'stage_timeline'` and `module 'token_drift.viz' has no attribute 'plot_timeline'`.

- [ ] **Step 3: Implement the plots**

Add to the end of `src/token_drift/viz.py`:

```python
# ---------- v2 timeline (training checkpoints) ----------

F16_RESOLUTION = 2.0**-11  # float16 relative spacing: Pythia's checkpoints are stored in fp16


def _as_float(rows) -> np.ndarray:
    """JSON lists (None where a NaN was) -> float array with NaN; matplotlib skips NaN points."""
    return np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)


def _step_axis(ax, steps: list[int]):
    # symlog = linear in [-1, 1] and log beyond, so step 0 gets a spot at the left edge
    # (plain log can't show 0) and 1 .. 143000 still spread out evenly per decade
    ax.set_xscale("symlog", linthresh=1)
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
    ax.set_ylabel("median ||W_t[i] - W_0[i]|| / ||W_0[i]||", fontsize=8, color=_MUTED)
    ax.set_title("C. row drift from init by merge-rank bin (embed solid, unembed dashed)", loc="left", fontsize=10)
    _style_axes(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7)
    return _save(fig, out_path, return_fig)
```

- [ ] **Step 4: Implement the stage**

In `src/token_drift/cli.py`, add to the imports (aliased, because `timeline` becomes a command name too):

```python
from token_drift import timeline as tl
```

Add after `stage_checkpoints`:

```python
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
```

Change the `all` branch from Task 2 to:

```python
    if "revisions" in cfg:  # v2: the v0 pipeline once per training checkpoint, then across them
        stage_checkpoints(cfg)
        stage_timeline(cfg)
        typer.echo(f"[all] the HF cache now holds {len(cfg['revisions'])} revisions of {cfg['model']} "
                   "(~160 MB each for pythia-70m); `uv run hf cache ls` / `uv run hf cache rm` to reclaim it")
        return
```

Add the command next to the other typer commands:

```python
@app.command("timeline")
def timeline_cmd(config: Path = _CONFIG):
    """v2: compare a checkpoint run's revisions (every revision must have finished `all`)."""
    stage_timeline(load_config(config))
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/token_drift/cli.py src/token_drift/viz.py tests/test_checkpoints.py tests/test_viz.py
git commit -m "v2: timeline stage (curves, with-final, row drift, half-way steps) + plots

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Training-time flipbooks

**Files:**
- Modify: `src/token_drift/viz.py` (`plot_flipbook` gets `png_names`, `gif_name`, `coords_name`)
- Modify: `src/token_drift/cli.py` (new `frame_tag`, `stage_flipbooks`; `all`; `timeline` command gets `--skip-flipbooks`)
- Test: `tests/test_viz.py`, `tests/test_checkpoints.py`

**Interfaces:**
- Consumes: Task 4's `_load_checkpoints` / `_Ckpt`, the `finished` fixture; existing `viz.project_layers`.
- Produces:
  - `viz.plot_flipbook(coords, labels, layer_names, out_dir, *, trajectories=None, frame_seconds=0.8, png_names: list[str] | None = None, gif_name: str = "flipbook.gif", coords_name: str = "umap_coords.npy") -> list[Path]` (defaults unchanged)
  - `cli.frame_tag(name: str) -> str` (`"L0 (embed)"` -> `"L0"`, `"L6 (pre-LN)"` -> `"L6_preLN"`, `"L6 (post-LN)"` -> `"L6_postLN"`, `"unembed"` -> `"unembed"`)
  - `cli.stage_flipbooks(cfg: dict) -> Path`, writing `timeline/flipbook_<tag>.gif`, `timeline/umap_<tag>_<step folder>.png`, `timeline/umap_coords_<tag>.npy`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_viz.py`:

```python
def test_plot_flipbook_custom_file_names(tmp_path, rng):
    coords = rng.normal(size=(2, 30, 2))
    labels = rng.integers(0, 3, 30)
    written = plot_flipbook(coords, labels, ["a @ step0", "a @ step1"], tmp_path,
                            png_names=["umap_a_s0.png", "umap_a_s1.png"], gif_name="flipbook_a.gif",
                            coords_name="umap_coords_a.npy")
    assert [p.name for p in written] == ["umap_a_s0.png", "umap_a_s1.png", "flipbook_a.gif"]
    assert (tmp_path / "umap_coords_a.npy").exists()
    assert not (tmp_path / "umap_coords.npy").exists() and not (tmp_path / "flipbook.gif").exists()


def test_plot_flipbook_refuses_a_png_name_count_mismatch(tmp_path, rng):
    with pytest.raises(ValueError, match="png_names"):
        plot_flipbook(rng.normal(size=(2, 30, 2)), rng.integers(0, 3, 30), ["a", "b"], tmp_path,
                      png_names=["only_one.png"])
```

Add to the end of `tests/test_checkpoints.py`:

```python
def test_frame_tag_is_short_and_unique():
    tags = [cli.frame_tag(n) for n in cli.layer_names(6)]
    assert tags[0] == "L0" and tags[3] == "L3" and tags[-1] == "unembed"
    assert tags[6] == "L6_preLN" and tags[7] == "L6_postLN"
    assert len(set(tags)) == len(tags)


def test_flipbooks_one_gif_per_timeline_frame_paged_by_step(finished):
    c, rd = finished
    cli.stage_flipbooks(c)
    out = rd / "timeline"
    idx = np.load(rd / "step0000008" / "frames" / "idx.npy")
    for tag in ("L0", "L1", "unembed"):
        assert (out / f"flipbook_{tag}.gif").exists()
        for r in REVS:
            assert (out / f"umap_{tag}_{cli.step_dir_name(r)}.png").exists()
        assert np.load(out / f"umap_coords_{tag}.npy").shape == (len(REVS), len(idx), 2)
```

In the same file, replace `test_all_on_a_checkpoint_config_runs_checkpoints_then_timeline` with:

```python
def test_all_on_a_checkpoint_config_runs_checkpoints_timeline_flipbooks(tmp_path, monkeypatch):
    c, _ = ckpt_cfg(tmp_path, monkeypatch)
    seen = []
    for s in ("corpus", "extract", "normalize", "metrics", "viz", "checkpoints", "timeline", "flipbooks"):
        monkeypatch.setattr(cli, f"stage_{s}", lambda c, s=s: seen.append(s))
    cli.all(Path(c["_config_path"]))
    assert seen == ["checkpoints", "timeline", "flipbooks"]


def test_timeline_command_can_skip_the_flipbooks(tmp_path, monkeypatch):
    c, _ = ckpt_cfg(tmp_path, monkeypatch)
    seen = []
    for s in ("timeline", "flipbooks"):
        monkeypatch.setattr(cli, f"stage_{s}", lambda c, s=s: seen.append(s))
    cli.timeline_cmd(Path(c["_config_path"]), skip_flipbooks=True)
    assert seen == ["timeline"]
    cli.timeline_cmd(Path(c["_config_path"]), skip_flipbooks=False)
    assert seen == ["timeline", "timeline", "flipbooks"]
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_viz.py tests/test_checkpoints.py -q`
Expected: FAIL with `plot_flipbook() got an unexpected keyword argument 'png_names'`, `module 'token_drift.cli' has no attribute 'frame_tag'`, and `timeline_cmd() got an unexpected keyword argument 'skip_flipbooks'`.

- [ ] **Step 3: Implement**

In `src/token_drift/viz.py`, change `plot_flipbook`'s signature and the three spots that use
the fixed file names:

```python
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
```

Inside the loop, `p = out_dir / f"umap_layer_{i}.png"` becomes `p = out_dir / png_names[i]`,
and after the loop `gif = out_dir / "flipbook.gif"` becomes `gif = out_dir / gif_name`.
Nothing else changes.

In `src/token_drift/cli.py`, add after `stage_timeline`:

```python
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
```

In `all`'s revisions branch, add `stage_flipbooks(cfg)` right after `stage_timeline(cfg)`.

Replace the `timeline` command:

```python
@app.command("timeline")
def timeline_cmd(
    config: Path = _CONFIG,
    skip_flipbooks: bool = typer.Option(False, "--skip-flipbooks",
                                        help="numbers + timeline.png + drift.png only; the flipbooks take 1-2.5 h"),
):
    """v2: compare a checkpoint run's revisions (every revision must have finished `all`)."""
    cfg = load_config(config)
    stage_timeline(cfg)
    if not skip_flipbooks:
        stage_flipbooks(cfg)
```

- [ ] **Step 4: Run the tests to see them pass**

Run: `uv run pytest -q`
Expected: PASS (the old flipbook tests too: default names are unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/token_drift/cli.py src/token_drift/viz.py tests/test_checkpoints.py tests/test_viz.py
git commit -m "v2: training-time flipbooks (one AlignedUMAP per timeline frame, pages = checkpoints)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Real run + sanity checks (with Andrey, not a subagent)

**Files:**
- Create: `scripts/v2_sanity.py`

**Interfaces:**
- Consumes: `runs/pythia70m_ckpt/` from `all`; `runs/pythia70m/metrics/metrics.json` and `runs/random_init/metrics/metrics.json` (both still on disk).
- Produces: printed sanity-check numbers for FINDINGS 13.

- [ ] **Step 1: Check disk and GPU**

Run: `df -h . && nvidia-smi --query-gpu=memory.used,memory.total --format=csv`
Expected: at least ~5 GB free (19 GB on 2026-10-04). The peak is ~1 GB for the revision in flight, plus
~1.3 GB of `frames/` + `weights/` in total, plus ~1.6 GB of HF cache.

- [ ] **Step 2: Write the sanity script**

Create `scripts/v2_sanity.py`:

```python
"""v2 sanity checks (spec 'Sanity checks on the real run'). Reads runs/ directly, like the other scripts.

1. step143000 should reproduce runs/pythia70m (same weights as main), within ~1e-3.
2. step0 vs runs/random_init: similar, not identical (HF's init may not be EleutherAI's).
3. The embed rows that moved least by the final step should be tokens the Pile (nearly) never has.
"""
import json
from pathlib import Path

import numpy as np

KEYS = ("knn_consecutive", "knn_purity", "knn_purity_shuffled", "anisotropy", "top_pc_share", "cka")
CK = Path("runs/pythia70m_ckpt")


def load(p):
    return json.loads(Path(p).read_text())


def gaps(a: dict, b: dict) -> dict:
    assert a["layer_names"] == b["layer_names"], (a["layer_names"], b["layer_names"])
    assert a["subsample_idx"] == b["subsample_idx"], "different metrics subsample: not comparable"
    return {k: round(float(np.nanmax(np.abs(np.asarray(a[k], float) - np.asarray(b[k], float)))), 4)
            for k in KEYS if a.get(k) is not None and b.get(k) is not None}


print("1. step143000 vs runs/pythia70m, max |diff| per curve (expect < ~1e-3):")
print("  ", gaps(load(CK / "step0143000/metrics/metrics.json"), load("runs/pythia70m/metrics/metrics.json")))
print("2. step0 vs runs/random_init, max |diff| per curve (expect similar, not identical):")
print("  ", gaps(load(CK / "step0000000/metrics/metrics.json"), load("runs/random_init/metrics/metrics.json")))
tl = load(CK / "timeline/timeline.json")
print("3. lowest embed drift at", tl["final"], "(expect never-seen tokens: odd bytes, junk merges):")
for r in tl["lowest_embed_drift"]:
    print(f"   {r['drift']:.4f}  bin {r['freq_bin']}  id {r['id']:>5}  {r['token']!r}")
print("half-way steps:", json.dumps(tl["half_way"], indent=1))
```

- [ ] **Step 3: Run the pipeline**

Run (in the background; it's long):
`uv run token-drift all --config configs/pythia70m_ckpt.yaml 2>&1 | tee runs/pythia70m_ckpt_all.log`
Expected: one `[checkpoints] stepN done` line per revision (a few minutes each: download, extract,
normalize, metrics), then the `[timeline]` lines, then three `[flipbooks]` lines at ~20-50 min each.
If an early revision fails to load (old branches may ship `pytorch_model.bin` instead of
safetensors), stop, use superpowers:systematic-debugging, and tell Andrey before changing anything.
A rerun resumes at the first unfinished revision.

- [ ] **Step 4: Run the sanity checks**

Run: `uv run python scripts/v2_sanity.py`
Expected: check 1 under ~1e-3 everywhere. If not, **stop**: revision loading or the config differs
from `runs/pythia70m`, and every curve after that is suspect. Checks 2 and 3 get reported whatever
they show.

- [ ] **Step 5: Look at the plots together**

Open `runs/pythia70m_ckpt/timeline/timeline.png` and `drift.png` (Read tool). The flipbooks are GIFs
and PyCharm can't play them, so show Andrey the per-step PNGs (`umap_L0_step*.png`) side by side,
or have him open the GIF in a browser. Write down, with numbers from `timeline.json`: the half-way
step for L0 purity, for the middle block, and for with-final kNN/CKA per frame; whether the embed's
rare bins lag the unembed in `drift.png`; anything surprising.

- [ ] **Step 6: Commit**

```bash
git add scripts/v2_sanity.py
git commit -m "scripts: v2 sanity checks (step143000 vs pythia70m, step0 vs random_init, lowest drift)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Write-up

**Files:**
- Modify: `docs/FINDINGS.md` (new section 13), `README.md` (v2 paragraph), `docs/OPEN_QUESTIONS.md` (v2 entry), `CLAUDE.md` (layout + pipeline + status)
- Create: `docs/results/v2_timeline.png`, `docs/results/v2_drift.png` (copies of the run's plots)

**Interfaces:**
- Consumes: Task 6's numbers and plots.
- Produces: docs only.

- [ ] **Step 1: Copy the figures**

```bash
cp runs/pythia70m_ckpt/timeline/timeline.png docs/results/v2_timeline.png
cp runs/pythia70m_ckpt/timeline/drift.png docs/results/v2_drift.png
```

- [ ] **Step 2: FINDINGS 13**

Append `## 13. v2: Pythia-70m training checkpoints (<date of the run>)` to `docs/FINDINGS.md`, in
the style of sections 11-12 (numbers in tables, casual tone). Subsections:
- 13.1 Setup: the 10 revisions, v0 mode, what "half-way step" means, float16 resolution of the checkpoints.
- 13.2 Sanity checks 1-3, numbers from `scripts/v2_sanity.py`, gaps reported whatever they are.
- 13.3 When L0 clusters by surface form: half-way step plus the purity-minus-shuffled values at each step for L0 and unembed.
- 13.4 When the stable middle block forms: half-way step plus the mean consecutive overlap L1->L2..L4->L5 per step.
- 13.5 How finished is each frame: with-final kNN vs CKA per step for L0 / L3 / unembed, and where local and global disagree.
- 13.6 The drift prediction: held or not, per bin, embed vs unembed, and what the lowest-drift tokens are.
- 13.7 Surprises, flagged and not smoothed over (CLAUDE.md "Working with me").

- [ ] **Step 3: README, OPEN_QUESTIONS, CLAUDE.md**

- README: add `## Results (v2, training checkpoints, <date>)` after the milestone B section: the two
  figures, three sentences of what we saw, and a quickstart line
  `uv run token-drift all --config configs/pythia70m_ckpt.yaml` (+ `timeline --skip-flipbooks`).
- OPEN_QUESTIONS: add `## Surprises from the v2 checkpoint runs (<date>)` with whatever 13.7 raised.
  Mark the two v0 questions the spec answers (L0 clustering timing, middle block timing) as answered,
  with a pointer to FINDINGS 13.
- CLAUDE.md: add `timeline.py  # v2: with-final overlap/CKA, row drift, half-way steps` to the repo
  layout; add `timeline` to the cli line; add a short "Checkpoint mode (v2)" paragraph to the
  pipeline section (config `revisions:` + `timeline:`, step folders `step%07d` with `config.yaml`,
  `metrics/`, `frames/`, `weights/`, the `timeline/` folder, `--skip-flipbooks`, resume rule = metrics.json + frames/idx.npy);
  update the Status paragraph.

- [ ] **Step 4: Run the tests and commit**

Run: `uv run pytest -q`
Expected: PASS.

```bash
git add docs/FINDINGS.md docs/OPEN_QUESTIONS.md README.md CLAUDE.md docs/results/v2_timeline.png docs/results/v2_drift.png
git commit -m "docs: v2 checkpoints write-up (FINDINGS 13, README, OPEN_QUESTIONS, CLAUDE.md)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
