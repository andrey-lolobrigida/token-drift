# v1 milestone A (corpus-averaged vocab) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v0's "token alone" vocab matrix with "each token's residual averaged over every
time it appears in real text", rerun the v0 metrics + flipbook on it, and put v0 and v1 side
by side on the same tokens.

**Architecture:** A new `corpus` stage downloads a Pile slice, glues docs together with
`<|endoftext|>` and cuts them into 2048-token windows (optionally shuffled inside each window).
`extract` gets a `corpus` mode that runs those windows through the model and keeps per-token
running sums on the GPU (unit-normed and plain), from which it gets the means, the counts and
Ethayarajh self-similarity without ever storing single occurrences. `normalize`, `metrics` and
`viz` learn about counts (eligibility, corpus frequency bins, self-sim panel, trajectory
groups), and a new `v0v1` command compares a v0 run and a v1 run on identical tokens.

**Tech Stack:** Python 3.11+ (venv is 3.12), uv, torch, transformers, numpy, scikit-learn,
umap-learn, matplotlib, pyyaml, typer, **pyarrow (new)**, huggingface_hub (already pulled in by
transformers, now used directly).

**Spec:** `docs/superpowers/specs/2026-09-24-v1-milestone-a-design.md`

## Global Constraints

- Every run is reproducible from its yaml + seed; the yaml is copied into the run dir.
- Seed everything (torch, numpy, UMAP `random_state`, k-means `random_state`).
- float16 on disk, upcast per layer when computing.
- Functions take arrays and return arrays; only `cli.py` reads another stage's files.
- Existing v0 configs keep working untouched: no `corpus:` block, `extract.mode` defaults to
  `vocab`, `normalize.source` is ignored, no `min_count` means all tokens are eligible, a flat
  `trajectory_tokens` list is still accepted.
- Frames are the v0 frames: embed, blocks 1..n-1, block n pre-final-LN, block n post-final-LN;
  unembed appended as a pseudo-layer from `normalize` on.
- Positions counted: `position >= min_context (32)` and `token != <|endoftext|>`.
- Saved arrays are trimmed to the tokenizer's rows (50,277 for Pythia); zero-count rows are zero
  vectors and are never eligible.
- Comments say *why*, in a casual tone (CLAUDE.md "Working with me").
- `uv run pytest -q` passes before every commit (baseline: 86 passed).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Deviations from the spec (small, flagged for Andrey)

1. `corpus.py` API: `iter_parquet(source)` (lazy, local path *or* HF repo) + `read_parquet(path,
   text_field) -> (texts, set_names)` instead of `download_parquet` + `read_texts`. Lazy so a
   `max_tokens` budget on a sharded source stops downloading early; a local path makes the
   stage testable offline.
2. **`normalize` centers activation frames on counted rows only** (`counts > 0`). The spec says
   normalize is "otherwise unchanged", but the ~few-thousand never-seen tokens are zero rows,
   and averaging them into the mean shrinks it: real rows keep a leftover shared offset, which is
   exactly the anisotropy centering is there to remove. The unembed is still centered on all rows
   (as in v0), so the v0/v1 unembed frames are identical and cross overlap there is exactly 1.
3. Self-sim baseline is drawn with a faded triangle-down marker, not a dashed line: in
   `metrics.png`, `--` already means "second run".
4. Corpus frequency bins are `0 = most frequent ... 4 = rarest` (no "base/byte" bin 0 like the
   merge-rank bins). `metrics.json` gets `freq_bins_source: "corpus" | "merge_rank"` so the plot
   labels them right.
5. `v0v1` writes two PNGs: `v0v1.png` (the usual metrics overlay, v0 vs v1) and
   `cross_overlap.png`. Recomputed v0 metrics use v1's *corpus* frequency bins, so the Voita
   panel compares v0 and v1 on the same bins.
6. Known limitation, not fixed: `ksweep` on a corpus run reproduces the kNN curves but not the
   shuffled-label purity exactly (its rng replay assumes the v0 draw). Out of scope for A.

## Review Focus

1. A corpus shorter than one window (tiny `max_tokens`, wrong source) → clear `ValueError`, not
   an empty `windows.npy` that crashes three stages later. Test in Task 1.
2. Null or empty `text` values in a parquet file → treated as empty docs, not a crash. Test in Task 1.
3. `window <= min_context` → nothing would ever be counted → fail fast in extract. Test in Task 2.
4. Fewer eligible tokens than `metrics.subsample` → warn and use all of them. Test in Task 4.
5. `v0v1` given two runs that don't line up (different model / vocab / frames) → clear error
   instead of a numpy broadcast error. Test in Task 6.

## File map

| file | change |
|---|---|
| `src/token_drift/corpus.py` | **new**: iter_parquet, read_parquet, source_mix, pack, shuffle_within_windows |
| `src/token_drift/extract.py` | + `self_similarity`, `extract_corpus_means` |
| `src/token_drift/normalize.py` | + `fit_rows` param on `normalize_layer` / `normalize_all` |
| `src/token_drift/labels.py` | + `corpus_freq_bins` |
| `src/token_drift/metrics.py` | `compute_all` gets eligible / subsample_idx / merge-rank / self-sim; + `cross_overlap` |
| `src/token_drift/viz.py` | self-sim panel, freq-bin labels by source, public `plot_trajectories`, `plot_cross_overlap` |
| `src/token_drift/cli.py` | `stage_corpus`, extract branch, normalize source + mask, metrics eligibility, `v0v1`, trajectory groups, `all` |
| `configs/pythia70m_corpus.yaml`, `pythia70m_corpus_shuf.yaml`, `random_init_corpus.yaml` | **new** |
| `tests/test_corpus.py`, `tests/conftest.py` | **new** |
| `tests/test_extract.py`, `test_normalize.py`, `test_labels.py`, `test_metrics.py`, `test_viz.py`, `test_cli.py` | extended |
| `pyproject.toml`, `uv.lock` | + pyarrow |

---

### Task 1: `corpus` stage

**Files:**
- Create: `src/token_drift/corpus.py`, `tests/test_corpus.py`
- Modify: `src/token_drift/cli.py` (imports, new `_load_tokenizer`, `stage_corpus`, `corpus` command), `tests/test_cli.py`, `pyproject.toml`

**Interfaces:**
- Produces:
  - `corpus.iter_parquet(source: str) -> Iterator[Path]`
  - `corpus.read_parquet(path, text_field: str) -> tuple[list[str], list[str | None] | None]`
  - `corpus.source_mix(set_names: list[str | None] | None) -> dict[str, int]`
  - `corpus.pack(docs: Iterable[Sequence[int]], *, eos_id: int, window: int, max_tokens: int | None = None) -> np.ndarray` (int32, `(n_windows, window)`)
  - `corpus.shuffle_within_windows(windows: np.ndarray, seed: int) -> np.ndarray`
  - `cli._load_tokenizer(name: str)` (tests monkeypatch this)
  - `cli.stage_corpus(cfg: dict) -> Path` writing `runs/<run>/corpus/windows.npy` and `corpus/meta.json`

- [ ] **Step 1: Add the dependency**

Run: `uv add "pyarrow>=15"`
Expected: `pyproject.toml` dependencies gain `"pyarrow>=15"`, `uv.lock` updates, `uv run python -c "import pyarrow"` works.

- [ ] **Step 2: Write the failing corpus tests** (`tests/test_corpus.py`)

```python
"""Background corpus: parquet in, packed (n_windows, window) token ids out. No downloads."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from token_drift.corpus import iter_parquet, pack, read_parquet, shuffle_within_windows, source_mix


def _write_parquet(path, texts, sets=None):
    cols = {"text": texts}
    if sets is not None:
        cols["meta"] = [{"pile_set_name": s} for s in sets]
    pq.write_table(pa.table(cols), path)


def test_pack_puts_one_eos_after_each_doc_and_drops_the_tail():
    w = pack([[1, 2, 3], [4], [5, 6]], eos_id=0, window=4)
    # stream is 1 2 3 0 4 0 5 6 0: two full windows, the lone trailing 0 is dropped
    assert w.tolist() == [[1, 2, 3, 0], [4, 0, 5, 6]]
    assert w.dtype == np.int32


def test_pack_respects_max_tokens():
    w = pack([[1] * 10 for _ in range(10)], eos_id=0, window=4, max_tokens=13)
    assert w.shape == (3, 4)  # 13-token budget -> 3 full windows


def test_pack_empty_docs_are_just_an_eos():
    assert pack([[], [7], []], eos_id=0, window=3).tolist() == [[0, 7, 0]]


def test_pack_too_short_gives_zero_windows():
    assert pack([[1, 2]], eos_id=0, window=8).shape == (0, 8)


def test_shuffle_keeps_each_windows_tokens_and_is_seeded():
    w = np.arange(40, dtype=np.int32).reshape(4, 10)
    a = shuffle_within_windows(w, seed=0)
    assert np.array_equal(a, shuffle_within_windows(w, seed=0))
    assert not np.array_equal(a, shuffle_within_windows(w, seed=1))
    assert np.array_equal(np.sort(a, axis=1), w)  # same bag of tokens per window
    assert not np.array_equal(a, w)
    # each row gets its own permutation, not one shared permutation
    assert not all(np.array_equal(np.argsort(a[i]), np.argsort(a[0])) for i in range(1, 4))


def test_read_parquet_texts_and_source_mix(tmp_path):
    p = tmp_path / "a.parquet"
    _write_parquet(p, ["hi", None, "yo"], ["Pile-CC", "Github", "Pile-CC"])
    texts, sets = read_parquet(p, "text")
    assert texts == ["hi", "", "yo"]  # a null text is an empty doc, not a crash
    assert source_mix(sets) == {"Pile-CC": 2, "Github": 1}


def test_read_parquet_without_meta_column(tmp_path):
    p = tmp_path / "a.parquet"
    _write_parquet(p, ["hi"])
    texts, sets = read_parquet(p, "text")
    assert texts == ["hi"] and sets is None
    assert source_mix(None) == {}


def test_iter_parquet_local_dir_is_sorted_and_local_file_is_itself(tmp_path):
    _write_parquet(tmp_path / "b.parquet", ["x"])
    _write_parquet(tmp_path / "a.parquet", ["y"])
    assert [p.name for p in iter_parquet(str(tmp_path))] == ["a.parquet", "b.parquet"]
    assert list(iter_parquet(str(tmp_path / "b.parquet"))) == [tmp_path / "b.parquet"]
```

- [ ] **Step 3: Run to see them fail**

Run: `uv run pytest tests/test_corpus.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'token_drift.corpus'`

- [ ] **Step 4: Write `src/token_drift/corpus.py`**

```python
"""Background corpus for v1: fetch text, pack it into model-sized windows, maybe shuffle.

v1 swaps v0's "[BOS, tok] alone" for "the token as the model usually sees it", so we
need real text in the shape Pythia trained on: docs glued with <|endoftext|> and cut
into 2048-token windows. See docs/superpowers/specs/2026-09-24-v1-milestone-a-design.md.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def iter_parquet(source: str) -> Iterator[Path]:
    """Parquet files of a dataset in a stable order, fetched one at a time.

    `source` is a local file or dir (tests, offline) or an HF dataset repo id. Lazy on
    purpose: with a `max_tokens` budget on a big sharded source (the_pile_deduplicated)
    we stop downloading as soon as the budget is met.
    """
    local = Path(source)
    if local.is_file():
        yield local
        return
    if local.is_dir():
        yield from sorted(local.glob("*.parquet"))
        return
    from huggingface_hub import hf_hub_download, list_repo_files

    names = sorted(f for f in list_repo_files(source, repo_type="dataset") if f.endswith(".parquet"))
    if not names:
        raise ValueError(f"no parquet files in HF dataset {source!r}")
    for name in names:
        yield Path(hf_hub_download(source, name, repo_type="dataset"))


def read_parquet(path, text_field: str) -> tuple[list[str], list[str | None] | None]:
    """(texts, pile_set_name per doc). Set names are None if there's no `meta` column."""
    table = pq.read_table(path)
    texts = [t or "" for t in table.column(text_field).to_pylist()]  # null text = empty doc
    sets = None
    if "meta" in table.column_names:
        sets = [m.get("pile_set_name") if isinstance(m, dict) else None
                for m in table.column("meta").to_pylist()]
    return texts, sets


def source_mix(set_names: list[str | None] | None) -> dict[str, int]:
    """How many docs came from each Pile subset, biggest first. Just for meta.json."""
    if set_names is None:
        return {}
    return dict(Counter(s for s in set_names if s is not None).most_common())


def pack(
    docs: Iterable[Sequence[int]], *, eos_id: int, window: int, max_tokens: int | None = None
) -> np.ndarray:
    """Token-id docs -> (n_windows, window) int32.

    One EOS after every doc (that's how Pythia's training data was packed, so windows
    that span two docs are in-distribution). The partial tail is dropped: it's less than
    one window of data and would be the only window with a different length.
    """
    parts, total = [], 0
    for d in docs:
        a = np.append(np.asarray(d, dtype=np.int32), np.int32(eos_id))
        parts.append(a)
        total += len(a)
        if max_tokens is not None and total >= max_tokens:
            break
    stream = np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
    if max_tokens is not None:
        stream = stream[:max_tokens]
    n = len(stream) // window
    return stream[: n * window].reshape(n, window)


def shuffle_within_windows(windows: np.ndarray, seed: int) -> np.ndarray:
    """Independent seeded permutation of each row.

    Keeps each window's bag of tokens (so its topic) and destroys word order: the
    "is the model reading, or just soaking up co-occurrence?" control.
    """
    rng = np.random.default_rng(seed)
    # argsort of iid uniforms = a uniform random permutation, one per row, vectorized
    perm = np.argsort(rng.random(windows.shape), axis=1)
    return np.take_along_axis(windows, perm, axis=1)
```

- [ ] **Step 5: Run the corpus tests**

Run: `uv run pytest tests/test_corpus.py -q`
Expected: 8 passed

- [ ] **Step 6: Write the failing cli tests** (append to `tests/test_cli.py`; add `import pyarrow as pa`, `import pyarrow.parquet as pq` at the top)

```python
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
```

- [ ] **Step 7: Run to see them fail**

Run: `uv run pytest tests/test_cli.py -q -k corpus`
Expected: FAIL, `AttributeError: module 'token_drift.cli' has no attribute '_load_tokenizer'`

- [ ] **Step 8: Implement `stage_corpus` in `cli.py`**

Imports: add `from token_drift import corpus as cp`. Update the module docstring's first line to
`` `token-drift corpus|extract|normalize|metrics|viz|all --config configs/x.yaml`, plus `compare`, `ksweep`, `v0v1`. ``
Add under `# ---------- stages ----------`, before `stage_extract`:

```python
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
```

Add the command next to the others under `# ---------- typer commands ----------`:

```python
@app.command("corpus")
def corpus_cmd(config: Path = _CONFIG):
    """Download + pack the background corpus into windows (v1 runs only)."""
    stage_corpus(load_config(config))
```

- [ ] **Step 9: Run the whole suite**

Run: `uv run pytest -q`
Expected: 99 passed (86 + 8 + 5)

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock src/token_drift/corpus.py src/token_drift/cli.py tests/test_corpus.py tests/test_cli.py
git commit -m "corpus stage: pack a Pile slice into 2048-token windows, optional in-window shuffle

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `extract` corpus mode

**Files:**
- Create: `tests/conftest.py`
- Modify: `src/token_drift/extract.py` (append two functions), `src/token_drift/cli.py:stage_extract`, `tests/test_extract.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `corpus/windows.npy` from Task 1 (int32 `(n_windows, window)`)
- Produces:
  - `extract.self_similarity(sq_norm: np.ndarray, n: np.ndarray) -> np.ndarray` (float64, NaN where n < 2; broadcasts `(F, V)` with `(V,)`)
  - `extract.extract_corpus_means(model, windows, *, eos_id, min_context, batch_size, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]` = `(unit_mean f16 (F,Vm,d), raw_mean f16 (F,Vm,d), counts int64 (Vm,), self_sim f32 (F,Vm), baseline list of F floats)`, `F = n_layers+2`, `Vm` = model embedding rows
  - extract dir files for corpus runs: `acts.npy` (unit mean), `acts_rawmean.npy`, `counts.npy`, `self_sim.npy`, `self_sim_baseline.json`, plus every v0 file
  - `tests/conftest.py`: `tiny_tokenizer()`, `tiny_model(tok, window=16)`, `tiny_docs(n_docs=12, seed=0)`

- [ ] **Step 1: Write `tests/conftest.py`** (shared by this task's cli test and Task 5's end-to-end test)

```python
"""A real-but-tiny BPE tokenizer and GPT-NeoX, so corpus-mode extract runs for real offline."""
import numpy as np
import torch
from tokenizers import Tokenizer, models
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast

LETTERS = "abcdefghijklmnopqrstuvwxyz"
MERGES = [("t", "h"), ("th", "e"), (" ", "the"), ("i", "n"), (" ", "in"), ("a", "n"), ("o", "n"), ("e", "r")]
WORDS = ["the", "then", "in", "inner", "an", "on", "under", "a", "cat", "sat", "mat", "dog", "ran",
         "far", "quiz", "jump", "box", "vex", "kept", "why", "go", "lol"]


def tiny_tokenizer():
    base = ["<|endoftext|>", " ", "."] + list(LETTERS)
    vocab = {t: i for i, t in enumerate(base)}
    for a, b in MERGES:
        vocab.setdefault(a + b, len(vocab))
    bpe = Tokenizer(models.BPE(vocab=vocab, merges=MERGES))
    return PreTrainedTokenizerFast(
        tokenizer_object=bpe, eos_token="<|endoftext|>", bos_token="<|endoftext|>"
    )


def tiny_model(tok, window: int = 16):
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(
        vocab_size=len(tok), hidden_size=16, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=32, max_position_embeddings=window,
    )
    return GPTNeoXForCausalLM(cfg).eval()


def tiny_docs(n_docs: int = 12, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, size=60)) + "." for _ in range(n_docs)]
```

- [ ] **Step 2: Write the failing extract tests** (append to `tests/test_extract.py`)

```python
# ---------- corpus mode (v1): running sums over real windows ----------
from collections import defaultdict

from token_drift.extract import extract_corpus_means, self_similarity

EOS, MIN_CTX = 0, 4


@pytest.fixture(scope="module")
def windows():
    rng = np.random.default_rng(0)
    w = rng.integers(1, 20, size=(6, 16))  # few distinct ids, so every token repeats
    w[:, 5] = EOS
    w[2, 10] = EOS
    return w


@torch.no_grad()
def _occurrences(model, windows):
    """Every counted occurrence, the slow obvious way: {(frame, tok): [vectors]}."""
    grabbed = {}
    hook = final_norm(model).register_forward_pre_hook(lambda m, a: grabbed.__setitem__("pre", a[0]))
    occ = defaultdict(list)
    try:
        for w in windows:
            hs = model(input_ids=torch.as_tensor(w[None], dtype=torch.long), output_hidden_states=True).hidden_states
            frames = [*hs[:-1], grabbed.pop("pre"), hs[-1]]
            for p, t in enumerate(w):
                if p < MIN_CTX or t == EOS:
                    continue
                for f, h in enumerate(frames):
                    occ[f, int(t)].append(h[0, p].numpy().astype(np.float64))
    finally:
        hook.remove()
    return occ


def _means(model, windows, batch_size=4):
    return extract_corpus_means(model, windows, eos_id=EOS, min_context=MIN_CTX,
                                batch_size=batch_size, device="cpu")


def test_corpus_means_match_a_plain_loop(model, windows):
    unit, raw, counts, self_sim, _ = _means(model, windows)
    assert unit.shape == raw.shape == (LAYERS + 2, VOCAB, D) and unit.dtype == np.float16
    occ = _occurrences(model, windows)
    for (f, t), vs in occ.items():
        vs = np.stack(vs)
        assert counts[t] == len(vs)
        np.testing.assert_allclose(raw[f, t], vs.mean(0), atol=2e-3)
        units = vs / np.linalg.norm(vs, axis=1, keepdims=True)
        np.testing.assert_allclose(unit[f, t], units.mean(0), atol=2e-3)
        if len(vs) >= 2:  # self-sim = mean cosine over ordered pairs i != j
            cos = units @ units.T
            brute = (cos.sum() - len(vs)) / (len(vs) * (len(vs) - 1))
            assert self_sim[f, t] == pytest.approx(brute, abs=1e-4)


def test_zero_count_rows_are_zero_and_self_sim_nan(model, windows):
    unit, raw, counts, self_sim, _ = _means(model, windows)
    unseen = counts == 0
    assert unseen.any()  # ids >= 20 never appear
    assert not unit[:, unseen].any() and not raw[:, unseen].any()
    assert np.isnan(self_sim[:, unseen]).all()


def test_frame0_is_the_embedding_so_unit_mean_is_its_direction_and_self_sim_is_one(model, windows):
    unit, raw, counts, self_sim, _ = _means(model, windows)
    seen = counts > 0
    embed, _ = get_embed_unembed(model)
    e = embed[seen].astype(np.float32)
    np.testing.assert_allclose(raw[0, seen].astype(np.float32), e, atol=1e-3)
    np.testing.assert_allclose(unit[0, seen].astype(np.float32), e / np.linalg.norm(e, axis=1, keepdims=True), atol=2e-3)
    np.testing.assert_allclose(self_sim[0, counts >= 2], 1.0, atol=1e-4)


def test_early_positions_and_eos_are_never_counted(model, windows):
    _, _, counts, _, _ = _means(model, windows)
    pos = np.arange(windows.shape[1])
    keep = (pos >= MIN_CTX)[None, :] & (windows != EOS)
    assert counts.sum() == keep.sum()
    assert counts[EOS] == 0
    only_early = np.setdiff1d(np.unique(windows[:, :MIN_CTX]), np.unique(windows[keep]))
    assert (counts[only_early] == 0).all()


def test_corpus_batch_size_does_not_matter(model, windows):
    a = _means(model, windows, batch_size=1)
    b = _means(model, windows, batch_size=6)
    np.testing.assert_allclose(a[0].astype(np.float32), b[0].astype(np.float32), atol=2e-3)
    np.testing.assert_array_equal(a[2], b[2])


def test_baseline_is_mean_cosine_over_all_counted_occurrence_pairs(model, windows):
    *_, baseline = _means(model, windows)
    occ = _occurrences(model, windows)
    assert len(baseline) == LAYERS + 2
    for f in (0, LAYERS + 1):
        vs = np.concatenate([np.stack(v) for (ff, _), v in occ.items() if ff == f])
        u = vs / np.linalg.norm(vs, axis=1, keepdims=True)
        brute = ((u @ u.T).sum() - len(u)) / (len(u) * (len(u) - 1))
        assert baseline[f] == pytest.approx(brute, abs=1e-4)


def test_self_similarity_formula_on_known_vectors():
    u = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])  # cos pairs: 0, 1, 0 -> mean 1/3
    s = u.sum(0)
    assert self_similarity(np.array([s @ s]), np.array([3]))[0] == pytest.approx(1 / 3)
    assert np.isnan(self_similarity(np.array([1.0]), np.array([1]))[0])


def test_window_no_longer_than_min_context_fails_fast(model):
    w = np.ones((2, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="min_context"):
        extract_corpus_means(model, w, eos_id=EOS, min_context=4, batch_size=2, device="cpu")
```

- [ ] **Step 3: Run to see them fail**

Run: `uv run pytest tests/test_extract.py -q`
Expected: collection error, `ImportError: cannot import name 'extract_corpus_means'`

- [ ] **Step 4: Implement** (append to `src/token_drift/extract.py`; also update the module docstring's last sentence to "`extract_corpus_means` is the v1 corpus-averaged version.")

```python
def self_similarity(sq_norm: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Mean pairwise cosine of a token's occurrences, from the norm of their unit-vector sum.

    ||u_1 + ... + u_n||^2 = n (each vector with itself) + sum over ordered pairs i != j of
    cos_ij, so the mean over the n(n-1) pairs falls out without storing any occurrence.
    Ethayarajh (2019) self-similarity. NaN for n < 2 (no pairs). Broadcasts (F, V) with (V,).
    """
    sq_norm = np.asarray(sq_norm, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (sq_norm - n) / (n * (n - 1))
    return np.where(n >= 2, s, np.nan)


@torch.no_grad()
def extract_corpus_means(
    model, windows: np.ndarray, *, eos_id: int, min_context: int, batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Per-token mean residual over real text, same frames as `extract_activations`.

    -> (unit_mean, raw_mean, counts, self_sim, self_sim_baseline). Means are
    (n_layers+2, model vocab rows, d) float16; zero-count rows are zero.

    Two running sums per frame live on the device in float32 (2 x 8 x 50,304 x 512 x 4 B =
    1.65 GB for Pythia). unit_mean averages unit-normed occurrences: an occasional ordinary
    token turns into an attention sink (norm ~170 vs ~12) and would dominate a plain mean.
    float32 is plenty: " the" gets ~430k additions, relative error ~sqrt(n) x 1e-7 = 1e-4.
    """
    windows = np.asarray(windows)
    if windows.shape[1] <= min_context:
        raise ValueError(
            f"window ({windows.shape[1]}) must be longer than min_context ({min_context}), "
            "or no position ever gets counted"
        )
    n_frames = model.config.num_hidden_layers + 2
    vocab = model.get_input_embeddings().weight.shape[0]
    d = model.config.hidden_size
    sum_unit = torch.zeros(n_frames, vocab, d, device=device)
    sum_raw = torch.zeros(n_frames, vocab, d, device=device)
    counts = torch.zeros(vocab, dtype=torch.int64, device=device)
    pos = torch.arange(windows.shape[1], device=device)
    grabbed: dict[str, torch.Tensor] = {}
    hook = final_norm(model).register_forward_pre_hook(
        lambda mod, args: grabbed.__setitem__("pre_ln", args[0])
    )
    try:
        for start in tqdm(range(0, len(windows), batch_size), desc="extract", unit="batch"):
            ids = torch.as_tensor(windows[start : start + batch_size], dtype=torch.long, device=device)
            hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            frames = [*hs[:-1], grabbed.pop("pre_ln"), hs[-1]]
            # early positions and every EOS are attention sinks (L3 norm ~120 vs ~12); skip them
            keep = (pos >= min_context)[None, :] & (ids != eos_id)
            kept = ids[keep]
            counts += torch.bincount(kept, minlength=vocab)
            for f, h in enumerate(frames):
                h = h[keep].float()
                sum_raw[f].index_add_(0, kept, h)
                sum_unit[f].index_add_(0, kept, h / h.norm(dim=1, keepdim=True).clamp_min(1e-8))
    finally:
        hook.remove()

    counts_np = counts.cpu().numpy()
    denom = counts.clamp_min(1).float()[:, None]  # zero-count rows: 0 / 1 = 0, not NaN
    unit_mean = np.empty((n_frames, vocab, d), dtype=np.float16)
    raw_mean = np.empty((n_frames, vocab, d), dtype=np.float16)
    sq = np.empty((n_frames, vocab))
    total_sq = np.empty(n_frames)
    for f in range(n_frames):
        unit_mean[f] = (sum_unit[f] / denom).half().cpu().numpy()
        raw_mean[f] = (sum_raw[f] / denom).half().cpu().numpy()
        s = sum_unit[f].cpu().double()  # float64 on CPU: MPS has none, and ||S||^2 - n cancels a lot
        sq[f] = (s * s).sum(1).numpy()
        tot = s.sum(0)
        total_sq[f] = float(tot @ tot)
    self_sim = self_similarity(sq, counts_np).astype(np.float32)
    # anisotropy baseline: same formula on the grand total, i.e. mean cosine between any two
    # counted occurrences of any tokens. Uncentered, like Ethayarajh's.
    n_all = np.full(n_frames, counts_np.sum())
    baseline = [float(b) for b in self_similarity(total_sq, n_all)]
    return unit_mean, raw_mean, counts_np, self_sim, baseline
```

- [ ] **Step 5: Run the extract tests**

Run: `uv run pytest tests/test_extract.py -q`
Expected: 18 passed

- [ ] **Step 6: Write the failing cli test** (append to `tests/test_cli.py`; add `from conftest import tiny_model, tiny_tokenizer` near the top — pytest puts `tests/` on `sys.path` for conftest imports)

```python
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
```

- [ ] **Step 7: Run to see it fail**

Run: `uv run pytest tests/test_cli.py -q -k corpus_mode`
Expected: FAIL — v0 extract runs `extract_activations` and no `counts.npy` is written (`FileNotFoundError` on `acts_rawmean.npy`)

- [ ] **Step 8: Rewrite `stage_extract` in `cli.py`**

```python
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
        bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        acts = ex.extract_activations(
            model, np.arange(V), bos_id=bos_id, batch_size=cfg["extract"]["batch_size"], device=device
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
```

- [ ] **Step 9: Run the whole suite**

Run: `uv run pytest -q`
Expected: 108 passed

- [ ] **Step 10: Commit**

```bash
git add src/token_drift/extract.py src/token_drift/cli.py tests/conftest.py tests/test_extract.py tests/test_cli.py
git commit -m "extract: corpus mode (running unit/raw means, counts, Ethayarajh self-sim)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `normalize` picks the source and centers on counted rows

**Files:**
- Modify: `src/token_drift/normalize.py`, `src/token_drift/cli.py:stage_normalize`, `tests/test_normalize.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `extract/acts.npy`, `extract/acts_rawmean.npy`, `extract/counts.npy` from Task 2
- Produces:
  - `normalize_layer(x, *, center, unit_norm, drop_top_pcs, row_norm_first=False, fit_rows: np.ndarray | None = None)`
  - `normalize_all(acts, *, ..., fit_rows: np.ndarray | None = None)` (same mask for every frame passed in)
  - `tests/test_cli.py::corpus_run` fixture (a fake corpus-mode run dir), reused by Tasks 4, 6, 7

- [ ] **Step 1: Write the failing normalize tests** (append to `tests/test_normalize.py`; it already imports `numpy as np` and `normalize_layer`; add `normalize_all` to the import if missing)

```python
def test_fit_rows_centers_on_those_rows_only():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 8)) + 3
    x[:10] = 0  # ten never-seen tokens: zero rows
    fit = np.ones(50, bool)
    fit[:10] = False
    out = normalize_layer(x, center=True, unit_norm=False, drop_top_pcs=0, fit_rows=fit)
    np.testing.assert_allclose(out[fit].mean(0), 0, atol=1e-5)
    # without the mask the zero rows drag the mean toward 0 and real rows keep an offset
    out_all = normalize_layer(x, center=True, unit_norm=False, drop_top_pcs=0)
    assert np.abs(out_all[fit].mean(0)).max() > 0.3


def test_fit_rows_pcs_come_from_those_rows():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 8))
    x[:20, 0] *= 50  # junk rows stretched along axis 0; real rows aren't
    fit = np.ones(60, bool)
    fit[:20] = False
    out = normalize_layer(x, center=True, unit_norm=False, drop_top_pcs=1, fit_rows=fit)
    ref = normalize_layer(x[fit], center=True, unit_norm=False, drop_top_pcs=1)
    np.testing.assert_allclose(out[fit], ref, atol=1e-5)


def test_normalize_all_passes_fit_rows_to_every_frame():
    rng = np.random.default_rng(0)
    acts = rng.normal(size=(3, 40, 8)).astype(np.float16)
    fit = np.arange(40) >= 5
    out = normalize_all(acts, center=True, unit_norm=True, drop_top_pcs=0, fit_rows=fit)
    for i in range(3):
        ref = normalize_layer(acts[i], center=True, unit_norm=True, drop_top_pcs=0, fit_rows=fit)
        np.testing.assert_allclose(out[i].astype(np.float32), ref, atol=2e-3)
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/test_normalize.py -q`
Expected: FAIL, `TypeError: normalize_layer() got an unexpected keyword argument 'fit_rows'`

- [ ] **Step 3: Implement** — in `normalize.py`, add `fit_rows: np.ndarray | None = None` to both signatures, pass it through in `normalize_all`, and change the centering / PC block of `normalize_layer` to:

```python
    if center or drop_top_pcs > 0:
        # all-but-the-top (Mu & Viswanath 2018) is defined on centered data, so dropping
        # PCs implies centering even if the flag says otherwise.
        # fit_rows: compute the mean (and PCs) from these rows only, apply to all. Corpus
        # runs pass "tokens that occurred": never-seen tokens are zero rows, and averaging
        # them in shrinks the mean, leaving real rows with a shared offset.
        fit = out if fit_rows is None else out[fit_rows]
        out -= fit.mean(axis=0, keepdims=True)
    if drop_top_pcs > 0:
        # SVD of the centered matrix; right singular vectors are the PCs.
        # full_matrices=False keeps this cheap for (50k, 512).
        fit = out if fit_rows is None else out[fit_rows]
        _, _, vt = np.linalg.svd(fit, full_matrices=False)
        top = vt[:drop_top_pcs]  # (k, d)
        out -= (out @ top.T) @ top
```

Add one line to the `normalize_layer` docstring: `fit_rows: boolean mask of rows the mean and PCs are computed from (default: all).`

- [ ] **Step 4: Run the normalize tests**

Run: `uv run pytest tests/test_normalize.py -q`
Expected: all pass

- [ ] **Step 5: Write the failing cli tests** (append to `tests/test_cli.py`; add `import shutil` at the top)

```python
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
```

- [ ] **Step 6: Run to see them fail**

Run: `uv run pytest tests/test_cli.py -q -k normalize`
Expected: the three new tests FAIL (masking not applied / raw_mean ignored / no error raised)

- [ ] **Step 7: Rewrite `stage_normalize` in `cli.py`**

```python
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
```

Update the import: `from token_drift.normalize import normalize_all, normalize_layer`.

- [ ] **Step 8: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass (the v0 `test_stage_normalize_appends_unembed_as_last_layer` still passes unchanged)

- [ ] **Step 9: Commit**

```bash
git add src/token_drift/normalize.py src/token_drift/cli.py tests/test_normalize.py tests/test_cli.py
git commit -m "normalize: pick unit/raw mean source; center corpus frames on seen tokens only

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `metrics`: eligibility, corpus frequency bins, self-sim (+ panel)

**Files:**
- Modify: `src/token_drift/labels.py`, `src/token_drift/metrics.py:compute_all`, `src/token_drift/viz.py:plot_metrics`, `src/token_drift/cli.py:stage_metrics`, `tests/test_labels.py`, `tests/test_metrics.py`, `tests/test_viz.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `counts.npy`, `self_sim.npy`, `self_sim_baseline.json` (Task 2), `corpus_run` fixture (Task 3)
- Produces:
  - `labels.corpus_freq_bins(counts, eligible, n_bins=5) -> np.ndarray` int8, `0` = most frequent, `-1` = ineligible
  - `compute_all(..., eligible=None, subsample_idx=None, freq_bins_source=None, merge_rank_bins=None, n_merge_rank_bins=6, self_sim=None, self_sim_baseline=None)`; new json keys `eligible_n`, `freq_bins_source`, `knn_change_by_merge_rank`, `n_merge_rank_bins`, `self_sim`, `self_sim_baseline`, `self_sim_adjusted` (null for v0)
  - `cli.N_CORPUS_BINS = 5`, `cli.corpus_eligibility(counts, min_count) -> tuple[np.ndarray, np.ndarray]` (eligible mask, corpus bins); used again by `v0v1` in Task 6
  - `plot_metrics` draws a 4th row (self-sim, adjusted self-sim) only when some run has `self_sim`

- [ ] **Step 1: Write the failing labels test** (append to `tests/test_labels.py`)

```python
def test_corpus_freq_bins_equal_count_most_frequent_first():
    from token_drift.labels import corpus_freq_bins

    counts = np.array([100, 5, 50, 0, 20, 10, 3, 80, 40, 30, 60, 70])
    eligible = counts >= 5  # 10 eligible tokens
    b = corpus_freq_bins(counts, eligible, n_bins=5)
    assert (b[~eligible] == -1).all()
    assert np.bincount(b[eligible]).tolist() == [2, 2, 2, 2, 2]
    assert b[0] == 0 and b[1] == 4  # 100 is the most frequent, 5 the rarest eligible
```

- [ ] **Step 2: Write the failing metrics tests** (append to `tests/test_metrics.py`)

```python
# ---- v1: eligibility, forced subsample, merge-rank table, self-sim ----

def test_subsample_is_drawn_from_eligible_only(clustered):
    x, labels = clustered
    eligible = np.zeros(200, bool)
    eligible[::2] = True
    m = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=50, eligible=eligible)
    assert len(m["subsample_idx"]) == 50 and set(m["subsample_idx"]) <= set(np.flatnonzero(eligible))
    assert m["eligible_n"] == 100


def test_all_eligible_reproduces_the_v0_draw(clustered):
    x, labels = clustered
    a = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=50)
    b = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=50,
                    eligible=np.ones(200, bool))
    assert a["subsample_idx"] == b["subsample_idx"]


def test_fewer_eligible_than_subsample_warns_and_uses_all(clustered):
    x, labels = clustered
    eligible = np.zeros(200, bool)
    eligible[:30] = True
    with pytest.warns(UserWarning, match="eligible"):
        m = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=50, eligible=eligible)
    assert m["subsample_idx"] == list(range(30))


def test_subsample_idx_overrides_the_draw(clustered):
    x, labels = clustered
    m = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=50,
                    subsample_idx=np.arange(10, 60))
    assert m["subsample_idx"] == list(range(10, 60))


def test_self_sim_curves_are_means_over_the_subsample(clustered):
    x, labels = clustered
    ss = np.full((2, 200), 0.5, np.float32)
    ss[1] = 0.3
    ss[1, :100] = np.nan  # never drawn below: only rows >= 100 are eligible
    eligible = np.arange(200) >= 100
    m = compute_all([x, x, x], ["a", "b", "unembed"], labels, knn_k=5, kmeans_k=4, seed=0,
                    subsample=None, eligible=eligible, self_sim=ss, self_sim_baseline=[0.1, 0.2])
    assert m["self_sim"] == pytest.approx([0.5, 0.3])  # one per extract frame, no unembed
    assert m["self_sim_baseline"] == pytest.approx([0.1, 0.2])
    assert m["self_sim_adjusted"] == pytest.approx([0.4, 0.1])
    import json
    json.dumps(m)


def test_corpus_bins_with_the_merge_rank_table_riding_along(clustered, rng):
    x, labels = clustered
    m = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=None,
                    freq_bins=rng.integers(0, 5, 200), n_freq_bins=5, freq_bins_source="corpus",
                    merge_rank_bins=rng.integers(0, 6, 200), n_merge_rank_bins=6)
    assert m["freq_bins_source"] == "corpus"
    assert np.asarray(m["knn_change_by_freq"]).shape == (1, 5)
    assert np.asarray(m["knn_change_by_merge_rank"]).shape == (1, 6)


def test_v1_keys_are_null_on_a_v0_style_call(clustered):
    x, labels = clustered
    m = compute_all([x, x], ["a", "b"], labels, knn_k=5, kmeans_k=4, seed=0, subsample=None)
    for k in ("self_sim", "self_sim_baseline", "self_sim_adjusted", "knn_change_by_merge_rank", "freq_bins_source"):
        assert m[k] is None, k
    assert m["eligible_n"] == 200
```

- [ ] **Step 3: Write the failing viz test** (append to `tests/test_viz.py`)

```python
def test_self_sim_row_appears_only_when_a_run_has_it(tmp_path, rng):
    import matplotlib.pyplot as plt

    v0 = _fake_metrics(rng)
    v1 = _fake_metrics(rng)
    v1["self_sim"] = rng.uniform(size=3).tolist()  # extract frames only: no unembed
    v1["self_sim_baseline"] = (rng.uniform(size=3) * 0.3).tolist()
    v1["self_sim_adjusted"] = (np.array(v1["self_sim"]) - np.array(v1["self_sim_baseline"])).tolist()
    v1["freq_bins_source"] = "corpus"
    fig = plot_metrics({"v0": v0, "v1": v1}, tmp_path / "ss.png", return_fig=True)
    assert len(fig.axes) == 8
    assert len(fig.axes[6].get_lines()) == 2  # self-sim + its anisotropy baseline, v1 only
    assert fig.axes[7].get_ylim()[0] < 0  # adjusted self-sim can go negative
    texts = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
    assert any("baseline" in t for t in texts)
    plt.close(fig)
    fig = plot_metrics({"v0": v0}, tmp_path / "no_ss.png", return_fig=True)
    assert len(fig.axes) == 6
    plt.close(fig)
```

- [ ] **Step 4: Write the failing cli test** (append to `tests/test_cli.py`)

```python
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
```

- [ ] **Step 5: Run to see them fail**

Run: `uv run pytest tests/test_labels.py tests/test_metrics.py tests/test_viz.py tests/test_cli.py -q`
Expected: the new tests FAIL (`ImportError: corpus_freq_bins`, `TypeError: unexpected keyword 'eligible'`, 6 axes instead of 8, `AttributeError: N_CORPUS_BINS`)

- [ ] **Step 6: Implement `labels.corpus_freq_bins`** (append to `labels.py`)

```python
def corpus_freq_bins(counts: np.ndarray, eligible: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Equal-count bins by *actual* corpus count: 0 = most frequent ... n_bins-1 = rarest.

    Only eligible tokens get a bin; the rest are -1 (knn_change_by_bin never asks for -1,
    so they drop out). Unlike merge rank there's no special base/byte bin: counts are
    comparable across every token.
    """
    counts = np.asarray(counts)
    eligible = np.asarray(eligible, dtype=bool)
    out = np.full(len(counts), -1, dtype=np.int8)
    m = int(eligible.sum())
    if m:
        order = np.argsort(-counts[eligible], kind="stable")
        pos = np.empty(m, dtype=np.int64)
        pos[order] = np.arange(m)
        out[eligible] = (pos * n_bins) // m
    return out
```

- [ ] **Step 7: Implement the `compute_all` changes** in `metrics.py`

Add `import warnings` at the top. New signature and the subsample block:

```python
def compute_all(
    layers: list[np.ndarray],
    layer_names: list[str],
    labels: np.ndarray,
    *,
    knn_k: int,
    kmeans_k: int,
    seed: int,
    subsample: int | None,
    raw_layers: list[np.ndarray] | None = None,
    freq_bins: np.ndarray | None = None,
    n_freq_bins: int = 5,
    freq_bins_source: str | None = None,
    eligible: np.ndarray | None = None,
    subsample_idx: np.ndarray | None = None,
    merge_rank_bins: np.ndarray | None = None,
    n_merge_rank_bins: int = 6,
    self_sim: np.ndarray | None = None,
    self_sim_baseline: list[float] | None = None,
) -> dict:
    """All per-layer curves in one JSON-serializable dict.

    `layers` should already include the unembed pseudo-layer as the last entry.
    One random subsample of tokens is drawn once and reused for every layer.
    `raw_layers` (uncentered, same order) is only used for the anisotropy curve;
    `freq_bins` (one int per token) enables the change-by-frequency table.
    v1 extras: `eligible` limits the draw (count >= min_count), `subsample_idx` skips it
    (v0v1 reuses v1's tokens), `merge_rank_bins` adds a second Voita table next to the
    corpus one, `self_sim` (extract frames x vocab) + baseline add Ethayarajh's curves.
    """
    rng = np.random.default_rng(seed)
    n = layers[0].shape[0]
    pool = np.arange(n) if eligible is None else np.flatnonzero(eligible)
    if subsample_idx is not None:
        idx = np.asarray(subsample_idx)
    elif subsample is not None and subsample < len(pool):
        # draw positions in the pool, not ids: with pool = every token this is the v0 draw exactly
        idx = np.sort(pool[rng.choice(len(pool), size=subsample, replace=False)])
    else:
        if subsample is not None and eligible is not None and subsample > len(pool):
            warnings.warn(f"only {len(pool)} eligible tokens, fewer than subsample={subsample}; "
                          "using all of them", stacklevel=2)
        idx = pool
```

The body from `xs = [...]` down to `cka = ...` stays as it is. After the `by_freq` block add:

```python
    by_rank = None
    if merge_rank_bins is not None:
        mb = np.asarray(merge_rank_bins)[idx]
        by_rank = [knn_change_by_bin(knn[i], knn[i + 1], mb, n_merge_rank_bins) for i in range(L - 1)]
    ss = ss_base = ss_adj = None
    if self_sim is not None:
        # one value per extract frame; the unembed pseudo-layer has no occurrences
        ss = [float(np.nanmean(np.asarray(row)[idx])) for row in self_sim]
        ss_base = [float(b) for b in self_sim_baseline]
        ss_adj = [s - b for s, b in zip(ss, ss_base)]  # Ethayarajh's anisotropy-adjusted self-sim
```

and extend the returned dict (keep every existing key; `"subsample_idx": idx.tolist()` works for both paths because `idx` is always an ndarray):

```python
        "freq_bins_source": (freq_bins_source or "merge_rank") if freq_bins is not None else None,
        "knn_change_by_merge_rank": by_rank,
        "n_merge_rank_bins": n_merge_rank_bins if merge_rank_bins is not None else None,
        "eligible_n": int(len(pool)),
        "self_sim": ss,
        "self_sim_baseline": ss_base,
        "self_sim_adjusted": ss_adj,
```

- [ ] **Step 8: Implement the viz changes** in `plot_metrics`

Replace the subplot creation with:

```python
    has_ss = any(m.get("self_sim") is not None for m in runs.values())
    fig, axes = plt.subplots(4 if has_ss else 3, 2, figsize=(12, 16 if has_ss else 12))
    (ax_cons, ax_drift), (ax_sil, ax_ari), (ax_aniso, ax_freq) = axes[:3]
    ax_ss, ax_ssa = axes[3] if has_ss else (None, None)
```

Inside the per-run loop, replace the frequency-bin label line with a source-aware one, and add the self-sim lines:

```python
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
```

After the tick/title block, before the `for ax in axes.flat` loop:

```python
    if has_ss:
        for ax in (ax_ss, ax_ssa):
            ax.set_xticks(np.arange(len(names) - 1), names[:-1], fontsize=7, **rot)
        ax_ss.set_title("self-similarity: mean cos between a token's occurrences (Ethayarajh 2019)",
                        loc="left", fontsize=10)
        ax_ssa.set_title("adjusted self-similarity = self-sim - baseline", loc="left", fontsize=10)
```

The freq panel title becomes source-aware:

```python
    sources = {m.get("freq_bins_source", "merge_rank") for m in runs.values() if m.get("knn_change_by_freq") is not None}
    if sources == {"corpus"}:
        freq_title = ("neighborhood change by corpus-count bin (Voita et al. 2019)\n"
                      "light -> dark: bin 0 (most frequent) ... bin 4 (rarest)")
    else:
        freq_title = ("neighborhood change by token frequency bin (Voita et al. 2019)\n"
                      "light -> dark: base/byte, bin 1 (most frequent) ... bin 5 (rarest)")
    ax_freq.set_title(freq_title, loc="left", fontsize=10)
```

After the `for ax in axes.flat: ... set_ylim(0, 1.02)` loop:

```python
    if has_ss:
        ax_ssa.set_ylim(-0.2, 1.02)  # adjusted self-sim dips below 0 when occurrences are less alike than random pairs
```

and in the shape legend, append the baseline marker when present:

```python
    if has_ss:
        shape_key.append(("v", "self-sim baseline"))
```

- [ ] **Step 9: Implement `stage_metrics` in `cli.py`**

Add near `N_FREQ_BINS`:

```python
N_CORPUS_BINS = 5  # equal-count bins of corpus count over eligible tokens; 0 = most frequent


def corpus_eligibility(counts: np.ndarray, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """(eligible mask, corpus freq bins). An average over 3 contexts mostly says which 3
    sentences they were, hence min_count; zero-count rows are never eligible."""
    eligible = counts >= max(min_count, 1)
    return eligible, corpus_freq_bins(counts, eligible, n_bins=N_CORPUS_BINS)
```

(import `corpus_freq_bins` from `token_drift.labels`). In `stage_metrics`, replace the `result = mt.compute_all(...)` call with:

```python
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
```

and after the existing echoes:

```python
    if result["self_sim"] is not None:
        typer.echo(f"[metrics] eligible={result['eligible_n']} self_sim_adjusted={np.round(result['self_sim_adjusted'], 3).tolist()}")
```

- [ ] **Step 10: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 11: Commit**

```bash
git add src/token_drift/labels.py src/token_drift/metrics.py src/token_drift/viz.py src/token_drift/cli.py tests/
git commit -m "metrics: min-count eligibility, corpus freq bins, self-sim curves + panel

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: the three v1 configs; `all` runs `corpus` first

**Files:**
- Create: `configs/pythia70m_corpus.yaml`, `configs/pythia70m_corpus_shuf.yaml`, `configs/random_init_corpus.yaml`
- Modify: `src/token_drift/cli.py` (`all`), `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4; `tests/conftest.py` helpers from Task 2
- Produces: runnable v1 configs; `token-drift all` does corpus → extract → normalize → metrics → viz when a `corpus:` block exists

- [ ] **Step 1: Write `configs/pythia70m_corpus.yaml`**

```yaml
# v1: each token's residual averaged over every time it shows up in real text,
# instead of v0's [BOS, tok] alone. Spec: docs/superpowers/specs/2026-09-24-v1-milestone-a-design.md
run_name: pythia70m_corpus
model: EleutherAI/pythia-70m
random_init: false
seed: 0
device: auto

corpus:
  source: NeelNanda/pile-10k   # any HF parquet dataset with a text column, or a local path
  text_field: text
  max_tokens: null             # null = all (15.4M for pile-10k); a budget for bigger sources
  window: 2048                 # Pythia's training context, so windows are in-distribution
  shuffle: none                # none | window (seeded permutation inside each window)

extract:
  mode: corpus                 # vocab (v0; default when the key is missing) | corpus
  batch_size: 8                # windows per forward pass
  min_context: 32              # early positions are attention sinks; norms settle by ~16
  dtype_on_disk: float16

normalize:
  source: unit_mean            # unit_mean | raw_mean (the robustness variant, same extract)
  center: true
  unit_norm: true
  drop_top_pcs: 0

metrics:
  min_count: 20                # an average over 3 contexts mostly says which 3 sentences
  subsample: 10000
  knn_k: 10
  kmeans_k: 20
  intrinsic_dim: false

viz:
  method: aligned_umap
  n_neighbors: 15
  min_dist: 0.1
  color_by: category
  trajectory_groups:           # one trajectories_<group>.png each
    virtue: [" virtue", " virtues", " courage", " justice", " wisdom", " honour", " honor",
             " humility", " patience", " generosity", " honesty", " charity", " grace",
             " faith", " hope", " friendship", " happiness", " moral", " noble", " good"]
    vice: [" vice", " vicious", " pride", " envy", " greed", " lust", " wrath", " anger",
           " vanity", " shame", " evil", " temper"]
    aristotle_mean: [" mean", " excess", " deficiency", " habit", " character", " pleasure"]
    polysemy: [" bank", " bat", " spring", " bass", " match", " light", " cell", " plant",
               " mouse", " python", " apple", " java", " pitch", " current", " table",
               " net", " tie", " seal", " mole", " trunk", " bark", " date", " fair",
               " kind", " saw", " lead", " right", " left"]
    names_function: [" Washington", " Jordan", " Paris", " China", " Turkey", " may",
                     " will", " can", " the", " it", " that", " set", " run"]

out_dir: runs
```

- [ ] **Step 2: Write the two controls**

`configs/pythia70m_corpus_shuf.yaml`: copy of the file above with the header comment replaced by
```yaml
# Control: same corpus with tokens shuffled inside each window. Keeps topic (the bag of
# words), kills word order: is the model reading, or just soaking up co-occurrence?
```
`run_name: pythia70m_corpus_shuf` and `shuffle: window`. Nothing else differs.

`configs/random_init_corpus.yaml`: copy with header
```yaml
# Control: same corpus, random weights. Random attention still mixes context in, just
# without anything learned; whatever survives here is the architecture talking.
```
`run_name: random_init_corpus` and `random_init: true`. Nothing else differs.

- [ ] **Step 3: Write the failing tests** (append to `tests/test_cli.py`; add `import copy`, `from pathlib import Path`, and extend the conftest import to `from conftest import tiny_docs, tiny_model, tiny_tokenizer`)

```python
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
```

- [ ] **Step 4: Run to see them fail**

Run: `uv run pytest tests/test_cli.py -q -k "v1 or all_"`
Expected: `test_all_runs_the_whole_v1_pipeline...` FAILS with `FileNotFoundError: .../corpus/windows.npy` (`all` never ran `corpus`); the configs test passes if Steps 1-2 were done right.

- [ ] **Step 5: Update `all` in `cli.py`**

```python
@app.command()
def all(config: Path = _CONFIG):  # noqa: A001 - it's the CLI verb we documented
    cfg = load_config(config)
    if "corpus" in cfg:  # v1 runs start from text; v0 configs have no corpus block
        stage_corpus(cfg)
    stage_extract(cfg)
    stage_normalize(cfg)
    stage_metrics(cfg)
    stage_viz(cfg)
```

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass. If stacked UMAP chokes on the tiny run (it has ~30 eligible tokens x 5 frames), raise `tiny_docs(n_docs=...)` or lower `min_count` in the test, don't touch UMAP params.

- [ ] **Step 7: Commit**

```bash
git add configs/pythia70m_corpus.yaml configs/pythia70m_corpus_shuf.yaml configs/random_init_corpus.yaml src/token_drift/cli.py tests/test_cli.py
git commit -m "v1 configs (corpus, shuffled, random init); all runs the corpus stage first

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `v0v1` command

**Files:**
- Modify: `src/token_drift/metrics.py` (+ `cross_overlap`), `src/token_drift/viz.py` (+ `plot_cross_overlap`), `src/token_drift/cli.py` (+ `run_v0v1`, `v0v1` command), `tests/test_metrics.py`, `tests/test_viz.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `compute_all(subsample_idx=...)` and `cli.corpus_eligibility` (Task 4); `fake_extract` + `corpus_run` fixtures
- Produces:
  - `metrics.cross_overlap(a_layers, b_layers, idx, k) -> list[float]`
  - `viz.plot_cross_overlap(cross, layer_names, out_path, *, title) -> Path`
  - `cli.run_v0v1(v0_dir, v1_dir) -> Path` writing `<runs>/v0v1_<v0 name>/{v0v1.json, v0v1.png, cross_overlap.png}`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:

```python
def test_cross_overlap_is_one_for_identical_frames_and_low_for_unrelated(clustered, rng):
    from token_drift.metrics import cross_overlap

    x, _ = clustered
    noise = _unit(rng.normal(size=x.shape)).astype(np.float32)
    c = cross_overlap([x, x], [x, noise], np.arange(200), k=5)
    assert c[0] == pytest.approx(1.0) and c[1] < 0.2
```

`tests/test_viz.py`:

```python
def test_plot_cross_overlap(tmp_path):
    from token_drift.viz import plot_cross_overlap

    p = plot_cross_overlap([1.0, 0.5, 0.4, 1.0], ["L0 (embed)", "L1", "L2", "unembed"], tmp_path / "c.png", title="v0 vs v1")
    assert p.exists()
```

`tests/test_cli.py`:

```python
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


def test_v0v1_refuses_runs_that_dont_line_up(fake_extract, corpus_run):
    c0, rd0 = fake_extract
    c1, rd1 = corpus_run
    cli.stage_normalize(c0)
    cli.stage_normalize(c1)
    cli.stage_metrics(c1)
    np.save(rd0 / "normalize" / "acts_norm.npy", np.load(rd0 / "normalize" / "acts_norm.npy")[:, :-1])
    with pytest.raises(ValueError, match="can't pair"):
        cli.run_v0v1(rd0, rd1)
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/ -q -k "cross_overlap or v0v1"`
Expected: FAIL, `ImportError: cannot import name 'cross_overlap'` / `AttributeError: run_v0v1`

- [ ] **Step 3: Implement `metrics.cross_overlap`** (append to `metrics.py`)

```python
def cross_overlap(
    a_layers: list[np.ndarray], b_layers: list[np.ndarray], idx: np.ndarray, k: int
) -> list[float]:
    """Per frame: kNN overlap between run A's frame f and run B's frame f, same tokens.

    1 = the two ways of building a vocab matrix (v0 alone vs v1 in context) agree on every
    token's neighbours at that depth. Q6 reads "does context take over?" off how it falls.
    """
    out = []
    for a, b in zip(a_layers, b_layers):
        na = knn_indices(np.asarray(a[idx], dtype=np.float32), k)
        nb = knn_indices(np.asarray(b[idx], dtype=np.float32), k)
        out.append(knn_overlap(na, nb))
    return out
```

- [ ] **Step 4: Implement `viz.plot_cross_overlap`** (append to `viz.py`)

```python
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
```

- [ ] **Step 5: Implement `run_v0v1` + command in `cli.py`** (next to `compare_runs`)

```python
def run_v0v1(v0: Path, v1: Path) -> Path:
    """v0 (token alone) vs v1 (corpus-averaged) on exactly v1's metrics tokens.

    Recomputes v0's curves on v1's subsample and corpus frequency bins (v0's own metrics/
    is left alone), plus the per-frame cross overlap. Writes <runs>/v0v1_<v0 name>/.
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
    _, cbins = corpus_eligibility(np.load(v1 / "extract" / "counts.npy"), mc.get("min_count", 1))
    t0 = time.time()
    frames0 = [n0[i] for i in range(n0.shape[0])]
    m0 = mt.compute_all(
        frames0, names, np.load(v1 / "extract" / "labels.npy"),
        knn_k=mc["knn_k"], kmeans_k=mc["kmeans_k"], seed=cfg1["seed"], subsample=None,
        subsample_idx=idx, freq_bins=cbins, n_freq_bins=N_CORPUS_BINS, freq_bins_source="corpus",
    )
    cross = mt.cross_overlap(frames0, [n1[i] for i in range(n1.shape[0])], idx, mc["knn_k"])
    out = stage_dir(v0.parent, f"v0v1_{v0.name}")
    result = {"v0_run": v0.name, "v1_run": v1.name, "layer_names": names, "n": int(len(idx)),
              "knn_k": mc["knn_k"], "cross_overlap": cross, "v0": m0, "v1": m1}
    (out / "v0v1.json").write_text(json.dumps(result, indent=1))
    viz.plot_metrics({f"{v0.name} (v0)": m0, f"{v1.name} (v1)": m1}, out / "v0v1.png")
    viz.plot_cross_overlap(cross, names, out / "cross_overlap.png",
                           title=f"kNN overlap, {v0.name} vs {v1.name}, same {len(idx)} tokens")
    typer.echo(f"[v0v1] cross_overlap={np.round(cross, 3).tolist()} ({time.time() - t0:.0f}s) -> {out}")
    return out
```

Command (under the typer commands):

```python
@app.command()
def v0v1(
    v0_run: Path = typer.Argument(..., help="v0 run dir, e.g. runs/pythia70m"),
    v1_run: Path = typer.Argument(..., help="v1 run dir, e.g. runs/pythia70m_corpus"),
):
    """Compare a v0 run and a v1 run on exactly the same tokens. Needs both normalized + v1's metrics."""
    run_v0v1(v0_run, v1_run)
```

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/token_drift/metrics.py src/token_drift/viz.py src/token_drift/cli.py tests/
git commit -m "v0v1: compare token-alone vs corpus-averaged on the same tokens, plus cross overlap

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: trajectory groups in `viz`

**Files:**
- Modify: `src/token_drift/viz.py` (public `plot_trajectories`, shared-limits helper), `src/token_drift/cli.py:stage_viz`, `tests/test_viz.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `counts.npy` (Task 2), `corpus_run` fixture (Task 3)
- Produces: `viz.plot_trajectories(coords, labels, layer_names, trajectories: dict[str, int], out_path, *, title=None) -> Path` where keys are the text drawn next to each path; `stage_viz` writes `trajectories_<group>.png` per group (flat `trajectory_tokens` still writes `trajectories.png`)

- [ ] **Step 1: Write the failing tests**

`tests/test_viz.py`:

```python
def test_plot_trajectories_labels_paths_with_the_given_text(tmp_path, rng):
    import matplotlib.pyplot as plt

    from token_drift.viz import plot_trajectories

    coords = rng.normal(size=(L, N, 2))
    labels = rng.integers(0, 10, size=N)
    p = plot_trajectories(coords, labels, ["a", "b", "c"], {"' vice' n=812": 3, "' envy' n=40": 7},
                          tmp_path / "trajectories_vice.png", title="vice")
    assert p.exists()
```

`tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run to see them fail**

Run: `uv run pytest tests/ -q -k trajector`
Expected: FAIL, `ImportError: cannot import name 'plot_trajectories'`

- [ ] **Step 3: Refactor `viz.py`**

Add a helper and use it inside `plot_flipbook` in place of the inline limit code:

```python
def _shared_limits(coords: np.ndarray):
    # Same limits for every frame, otherwise the eye reads a zoom as motion.
    lo, hi = coords.min(axis=(0, 1)), coords.max(axis=(0, 1))
    pad = 0.03 * (hi - lo)
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])
```

Rename `_plot_trajectories` to a public function with an explicit output path, annotating with the key as given:

```python
def plot_trajectories(
    coords: np.ndarray, labels: np.ndarray, layer_names: list[str], trajectories: dict[str, int],
    out_path: str | Path, *, title: str | None = None,
) -> Path:
    """Paths of a few tokens across frames, on top of a faint last-layer scatter.

    `trajectories` maps the text to draw next to each path (e.g. "' vice' n=812") to its
    row in `coords`.
    """
    xlim, ylim = _shared_limits(coords)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(coords[-1, :, 0], coords[-1, :, 1], s=1, alpha=0.08, color=_MUTED, linewidths=0)
    L = coords.shape[0]
    alphas = np.linspace(0.25, 1.0, L)  # fade in: early layers faint, last layer solid
    for j, (text, idx) in enumerate(trajectories.items()):
        color = RUN_COLORS[j % len(RUN_COLORS)] if j < len(RUN_COLORS) else CATEGORY_COLORS[CATEGORIES[labels[idx]]]
        path = coords[:, idx, :]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.2, alpha=0.7)
        for i in range(L):
            ax.scatter(path[i, 0], path[i, 1], s=18, color=color, alpha=alphas[i], linewidths=0)
        ax.annotate(text, path[-1], fontsize=7, color=_INK, xytext=(3, 3), textcoords="offset points")
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
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
```

In `plot_flipbook`, the trajectory call becomes:

```python
    if trajectories:
        written.append(plot_trajectories(
            coords, labels, layer_names, {repr(t): i for t, i in trajectories.items()},
            out_dir / "trajectories.png",
        ))
```

- [ ] **Step 4: Rewrite the trajectory part of `stage_viz` in `cli.py`**

Replace everything from `metrics = _load_json(...)` down to (not including) `t0 = time.time()` with:

```python
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
    np.save(out / "viz_idx.npy", idx)
    pos = {row: k for k, row in enumerate(idx)}
```

and replace the tail (from `viz.plot_flipbook(...)` on) with:

```python
    viz.plot_flipbook(coords, labels[idx], names, out)
    for g, rows in group_rows.items():
        if not rows:
            continue
        fname = "trajectories.png" if g == "" else f"trajectories_{g}.png"
        viz.plot_trajectories(coords, labels[idx], names, {text: pos[r] for text, r in rows.items()},
                              out / fname, title=g or None)
    typer.echo(f"[viz] wrote flipbook -> {out / 'flipbook.gif'}")
    return rd
```

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass, including the untouched v0 `test_stage_viz_uses_metric_subsample_plus_trajectory_tokens` (flat list → `trajectories.png`).

- [ ] **Step 6: Commit**

```bash
git add src/token_drift/viz.py src/token_drift/cli.py tests/test_viz.py tests/test_cli.py
git commit -m "viz: trajectory groups, one png each, labelled with corpus counts

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: the real runs

No code. Every command's output gets read, not just its exit code. **If a number looks
surprising, stop and tell Andrey before moving on** (CLAUDE.md: surprising is the point).

- [ ] **Step 1: Disk check**

Run: `df -h /home/andrey/PycharmProjects/token-drift`
Expected: >= 8 GB free (~1.4 GB per v1 run, ~6 GB total). If less, stop and ask.

- [ ] **Step 2: Re-extract the two v0 runs** (their `extract/` + `normalize/` were deleted 2026-09-24; `metrics/` and `viz/` still exist, `v0v1` only needs `normalize/`)

```bash
uv run token-drift extract --config configs/pythia70m.yaml && uv run token-drift normalize --config configs/pythia70m.yaml
uv run token-drift extract --config configs/random_init.yaml && uv run token-drift normalize --config configs/random_init.yaml
```

- [ ] **Step 3: The real v1 run**

Run: `uv run token-drift all --config configs/pythia70m_corpus.yaml`
Check in the output:
- `[corpus]` ~10k docs → about `(7500, 2048)` windows
- `[extract]` roughly 40k of 50,277 tokens with count >= 20 (`eligible=` in the metrics line), ~49k seen at all
- self-sim at frame 0 = 1.000 (the embedding is the same vector every time); if not, stop, something is wrong upstream
- L0 flipbook frame still clusters by surface form (it should: frame 0 is the embedding direction)
- If extract takes more than ~30 min on the GPU, stop and report instead of waiting.

- [ ] **Step 4: The two controls**

```bash
uv run token-drift all --config configs/pythia70m_corpus_shuf.yaml
uv run token-drift all --config configs/random_init_corpus.yaml
```

The shuffled run should have exactly the same `counts.npy` as the real run for every token
(shuffling within a window moves tokens but only across positions; the `>= 32` cut makes this
approximate, not exact, so compare `eligible_n` and expect them close, not equal).

- [ ] **Step 5: v0 vs v1 and the overlay**

```bash
uv run token-drift v0v1 runs/pythia70m runs/pythia70m_corpus
uv run token-drift v0v1 runs/random_init runs/random_init_corpus
uv run token-drift compare runs/pythia70m_corpus runs/pythia70m_corpus_shuf runs/random_init_corpus -o runs/compare_v1.png
```

Sanity: cross overlap at the unembed frame = 1.000 exactly (identical matrices, identical
normalization). Frame 0 is high but *not* exactly 1: v1's frame 0 is the embedding unit-normed
before centering (and centered on seen tokens only), v0's is centered raw.

- [ ] **Step 6: Raw-mean robustness variant** (cheap: same extract)

Copy `configs/pythia70m_corpus.yaml` to `configs/pythia70m_corpus_rawmean.yaml` with
`run_name: pythia70m_corpus_rawmean` and `normalize.source: raw_mean`, then copy
`runs/pythia70m_corpus/corpus` and `runs/pythia70m_corpus/extract` into
`runs/pythia70m_corpus_rawmean/` and run `normalize`, `metrics`, `viz` only. Compare curves;
if they disagree a lot, that's a finding about attention-sink occurrences, not a bug to hide.

- [ ] **Step 7: Show Andrey** the `metrics.png` of each v1 run, `compare_v1.png`, both `v0v1/`
folders and the `trajectories_*.png` files, via PNGs (PyCharm can't play GIFs; point at
individual `umap_layer_*.png` frames instead of `flipbook.gif`).

---

### Task 9: write-up

Done together with Andrey after looking at Task 8's plots; the numbers come from the runs.

**Files:**
- Modify: `docs/FINDINGS.md` (new section 11), `docs/OPEN_QUESTIONS.md` (Q6, Q7, Q3b, Q9 entries), `README.md` (v1 results section), `CLAUDE.md` (repo layout + pipeline: `corpus.py`, `corpus` stage, `v0v1` command, `acts_rawmean.npy`/`counts.npy`/`self_sim*`), `GLOSSARY.md` (any new terms)
- Copy selected PNGs into `docs/results/` (runs/ is gitignored)

- [ ] **Step 1:** FINDINGS §11: one sub-section per question in the spec's "Where each open question gets answered" table, each with the number that answers it and where to read it off. Surprises called out as surprises.
- [ ] **Step 2:** OPEN_QUESTIONS: mark Q6/Q7/Q3b/Q9 answered or narrowed, with a pointer to FINDINGS §11; add whatever new question the runs raised.
- [ ] **Step 3:** README: "Results (v1, corpus-averaged)" with `compare_v1.png`, the pythia `cross_overlap.png`, and three sentences.
- [ ] **Step 4:** CLAUDE.md layout/pipeline update; GLOSSARY append.
- [ ] **Step 5:** `uv run pytest -q`, then commit (`docs: v1 milestone A findings`) with the attribution line.
