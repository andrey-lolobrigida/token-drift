# v1 milestone B (virtue between vices, per occurrence) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test Andrey's Q16 hunch ("virtue words land between their vices") on per-occurrence
residuals of Pythia-70m and its random-init twin, using occurrences pulled from moral-philosophy
books and 3 Pile shards, scored against frequency-matched null words and a role-swap control.

**Architecture:** A new `probe_corpus` stage finds every whole-word hit of the probe, polysemy
and null-pool words in the books and the Pile, keeps up to 1000 per word per source group, and
cuts one window ending at each hit. `extract` (probe mode) runs those windows and keeps the
residual at the last token, giving `occ.npy (8 frames, n_occ, d)`. `metrics` averages occurrences
into one point per word per group per frame and scores each triple with `betweenness.py`;
`viz` draws a summary heatmap, the exact (t, d) plane per triple, and per-occurrence histograms.
Normalize is skipped in probe mode.

**Tech Stack:** Python 3.11+, uv, torch + transformers (GPT-NeoX), numpy, pyarrow, matplotlib,
typer, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-03-v1-milestone-b-design.md` (read it first; this plan
argues from it). Also `CLAUDE.md` (how we work) and `docs/OPEN_QUESTIONS.md` Q16.

## Global Constraints

- Run everything through uv: `uv run pytest -q`, `uv run token-drift ...`.
- Tokenizer, EOS id and max context always come from the model (`cfg["model"]`), never hardcoded (spec: model-agnostic).
- Word forms: lowercase, space-prefixed only (`" courage"`); words are written bare in the word file, the code prepends the space.
- Word boundary: the token after a hit must not start with a letter, digit, hyphen or apostrophe; a hit on a document's last token counts.
- Multi-token words: the vector is read at the last piece.
- Context: everything before the hit in the same document, capped at the model's max context (2048 for Pythia); fewer than `min_context` = 32 tokens before the last piece -> hit dropped. One EOS before a document's first token when the window reaches the doc start.
- A book is one document. Cap = 1000 occurrences per word per source group (`books` | `pile`), seeded.
- Points = `unit_mean` (unit-norm each occurrence, then average) per word x source group x frame.
- `min_count` = 20 per word per source group, else the triple is reported *missing* for that group (not an error).
- Null: per triple, keep the vice pair, swap the middle word for the `null_k` = 20 pool words nearest in log-count (same part of speech, same group). `between_pct` = 5.
- float16 on disk, upcast to float32 per frame. Seed everything from `cfg["seed"]`.
- Functions take arrays and return arrays. Only `cli.py` reads another stage's files (exception: `probe.fetch_gutenberg` reads/writes its own download cache).
- Tests: tiny, fast, no network. The real Pythia tokenizer is allowed (from the HF cache, `local_files_only=True`).
- Comments and docs: casual tone; a one-line *why* on every non-obvious choice (CLAUDE.md "Working with me").
- One commit per task, `uv run pytest -q` green before each. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **A Pile shard tokenized in one call.** One shard is ~120M tokens; HF returns them as Python
   ints in lists (~4+ GB) before we can shrink them, on a 15 GB box -> the run should tokenize in
   chunks and never hold more than a chunk as Python ints. Pinned by
   `test_tokenize_texts_feeds_the_tokenizer_in_chunks` (Task 4).
2. **Empty documents.** The Pile has empty/null texts (`read_parquet` turns them into `""`) ->
   an empty doc in the middle must not shift the doc index of later hits. Pinned by
   `test_find_hits_empty_doc_in_the_middle_keeps_doc_indices` (Task 2).
3. **A word on a document's last token.** There is no next token to check -> it counts as a
   whole word. Pinned by `test_find_hits_word_at_the_end_of_a_doc_counts` (Task 2) and the
   same case in `test_count_single_token_words` (Task 3).
4. **Capped words and the null.** Every frequent Pile word has exactly 1000 kept occurrences, so
   "nearest in log-count" on kept counts is a tie between all of them -> null neighbours must be
   ranked by the uncapped `found` counts. Pinned by
   `test_run_q16_null_neighbours_use_found_counts_not_capped_counts` (Task 6).
5. **Andrey edits `configs/probe_words.yaml` and reruns only `metrics`.** The occurrences on disk
   were cut for the old list -> metrics must use the word list saved with the occurrences
   (`probe_corpus/probe_words.yaml`), and refuse an `occ.npy` that no longer matches
   `occ_meta.parquet`. Pinned by `test_metrics_uses_the_word_list_saved_with_the_occurrences`
   and `test_metrics_refuses_a_stale_extract` (Task 7).

## Deviations from the spec (flag these to Andrey before executing)

- **D1 (new, recommended): unwrap Gutenberg hard line breaks.** Gutenberg texts are wrapped at
  ~70 characters. Checked on the Summa I-II: 33.6k of 434k lowercase words (~8%) start a line,
  and a line-start word tokenizes without its space (`"courage"`, not `" courage"`), so the
  matcher would silently skip it. `probe.unwrap` turns single newlines into spaces and keeps
  blank lines as paragraph breaks, so book text looks like ordinary prose (and like most Pile text).
- **D2: `count_report.md` is written by `probe_corpus`, not `metrics`.** It only needs counts, and
  this way Andrey can review counts and the null pool *before* spending ~20 min of GPU on extract.
- **D3: the "reservoir" is a bottom-k random-priority sample.** Every hit gets a seeded uniform
  priority; each word x group keeps its `cap` lowest. Same distribution as streaming reservoir
  sampling (uniform without replacement), but vectorized, since all hits fit in memory.
- **D4: the `min_context` filter runs before the cap, outside `cut_windows`.** Filter -> sample ->
  cut, so the cap only ever picks usable hits and we never cut a window we'd throw away.
  `cut_windows` therefore takes already-filtered hits and returns a list of windows.
- **D5: boundary check is a per-id lookup table**, `boundary_table(tokenizer) -> bool[vocab]`,
  instead of `boundary_ok(next_ids, tokenizer)`. Same rule, decided once per id.
- **D6: null neighbours are ranked by `found` counts** (uncapped), see Review Focus 4.
- **D7: the word file is copied into `probe_corpus/`** and metrics/viz/occ read that copy.
- **D8: size estimate.** The null pool (2 x 50 words) and the 28 polysemy words roughly double
  the spec's ~100k hits: expect ~200-250k occurrences, `occ.npy` ~2 GB, windows ~0.8 GB,
  extract ~15-20 min on the RTX 5060. 15 GB free on disk today: fine, but check before Task 10.
- **D9: config-load checks are structural only** (both `corpus:` and `probe:`, or `probe:` without
  `extract.mode: probe`). The "word tokenizes to EOS / nothing" check needs the tokenizer, so it
  runs first thing in `probe_corpus`, before any download.

---

## File Structure

| file | change | responsibility |
|---|---|---|
| `configs/probe_words.yaml` | create | the triples (classical + everyday), polysemy list, null_exclude |
| `configs/pythia70m_probe.yaml` | create | milestone B run config |
| `configs/random_init_probe.yaml` | create | same, random init |
| `src/token_drift/probe.py` | create | word file, Gutenberg fetch/slice/unwrap, token matching, counts, null pool, sampling, windows, snippets, count report |
| `src/token_drift/betweenness.py` | create | points, t / d / seg, null percentile, role swap, verdicts, set summary, `run_q16` |
| `src/token_drift/extract.py` | modify | add `extract_probe` |
| `src/token_drift/viz.py` | modify | add `plot_q16_summary`, `plot_q16_triples`, `plot_q16_occ` |
| `src/token_drift/cli.py` | modify | config checks, `stage_probe_corpus`, probe dispatch in extract/normalize/metrics/viz/all, `occ` command |
| `tests/conftest.py` | modify | `pythia_tok` fixture |
| `tests/test_probe.py` | create | pure `probe.py` tests |
| `tests/test_betweenness.py` | create | pure `betweenness.py` tests |
| `tests/test_probe_stages.py` | create | probe stages end to end on a tiny fake shelf |
| `tests/test_extract.py`, `tests/test_viz.py`, `tests/test_cli.py` | modify | new function tests, config tests |

Frames in probe mode: 8 for Pythia (embed, blocks 1-5, block 6 pre-LN, block 6 post-LN). No
unembed pseudo-frame: that's one row per token, not per occurrence. Names =
`layer_names(n_layers)[:-1]`.

---

### Task 1: Word file, probe configs, config checks

**Files:**
- Create: `configs/probe_words.yaml`, `configs/pythia70m_probe.yaml`, `configs/random_init_probe.yaml`
- Create: `src/token_drift/probe.py` (first two functions)
- Modify: `src/token_drift/cli.py` (`load_config`, new `_mode`)
- Modify: `tests/conftest.py` (add `pythia_tok` fixture, used from Task 2 on)
- Test: `tests/test_probe.py` (create), `tests/test_cli.py` (append)

**Interfaces:**
- Produces:
  - `probe.POS = ("noun", "adj")`, `probe.ROLES = ("deficiency", "mean", "excess")`, `probe.GROUPS = ("books", "pile")`
  - `probe.load_words(path) -> dict` with keys `"triples"` (list of `{"id": str, "set": str, "concept": str, "pos": str, "words": tuple[str, str, str]}`), `"polysemy"` (list[str]), `"null_exclude"` (list[str])
  - `probe.probe_word_pos(words: dict) -> dict[str, str]`: every triple word -> its part of speech, first-seen order, each word once
  - `cli._mode(cfg) -> str` (`"vocab"` when missing)
  - `conftest.pythia_tok` (session fixture, skips if the tokenizer isn't cached)

- [ ] **Step 1: Write the failing tests**

`tests/test_probe.py`:

```python
"""Milestone B probe helpers: pure functions, no network."""
from pathlib import Path

import pytest

from token_drift import probe as pb


def _write(tmp_path, text):
    p = tmp_path / "words.yaml"
    p.write_text(text)
    return p


WORDS = """
sets:
  classical:
    fear:
      noun: [[cowardice, courage, rashness], [fear, courage, fearlessness]]
      adj: [[cowardly, brave, rash]]
  everyday:
    fear:
      noun: [[cowardice, bravery, recklessness]]
polysemy: [bank, bat]
null_exclude: [business]
"""


def test_load_words_flattens_triples_in_file_order(tmp_path):
    w = pb.load_words(_write(tmp_path, WORDS))
    assert [t["words"] for t in w["triples"]] == [
        ("cowardice", "courage", "rashness"), ("fear", "courage", "fearlessness"),
        ("cowardly", "brave", "rash"), ("cowardice", "bravery", "recklessness"),
    ]
    t = w["triples"][2]
    assert (t["set"], t["concept"], t["pos"]) == ("classical", "fear", "adj")
    assert t["id"] == "classical/fear/adj/cowardly,brave,rash"
    assert w["polysemy"] == ["bank", "bat"] and w["null_exclude"] == ["business"]


def test_probe_word_pos_lists_each_word_once():
    w = {"triples": [
        {"words": ("cowardice", "courage", "rashness"), "pos": "noun"},
        {"words": ("fear", "courage", "fearlessness"), "pos": "noun"},
        {"words": ("cowardly", "brave", "rash"), "pos": "adj"},
    ]}
    assert pb.probe_word_pos(w) == {
        "cowardice": "noun", "courage": "noun", "rashness": "noun", "fear": "noun",
        "fearlessness": "noun", "cowardly": "adj", "brave": "adj", "rash": "adj",
    }


@pytest.mark.parametrize("bad, match", [
    ("sets: {s: {c: {verb: [[a, b, c]]}}}", "part of speech"),
    ("sets: {s: {c: {noun: [[a, b]]}}}", "deficiency/mean/excess"),
    ("sets: {s: {c: {noun: [[' a', b, c]]}}}", "bare lowercase"),
    ("sets: {s: {c: {noun: [[Courage, b, c]]}}}", "bare lowercase"),
])
def test_load_words_rejects_malformed_files(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        pb.load_words(_write(tmp_path, bad))


def test_the_real_word_file_loads():
    w = pb.load_words(Path(__file__).parents[1] / "configs" / "probe_words.yaml")
    assert {t["set"] for t in w["triples"]} == {"classical", "everyday"}
    assert len(w["triples"]) >= 40 and "bank" in w["polysemy"]
```

Append to `tests/test_cli.py`:

```python
# ---------- probe (milestone B) config tests ----------

def test_probe_configs_differ_only_where_they_should():
    load = lambda n: yaml.safe_load((CONFIGS / n).read_text())  # noqa: E731
    base, rand = load("pythia70m_probe.yaml"), load("random_init_probe.yaml")
    assert base["extract"]["mode"] == "probe" and "corpus" not in base
    r = copy.deepcopy(rand)
    r["run_name"], r["random_init"] = base["run_name"], False
    assert r == base and rand["random_init"] is True


def _write_cfg(tmp_path, c):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(c))
    return p


def test_load_config_refuses_corpus_and_probe_together(tmp_path):
    c = {"run_name": "x", "corpus": {}, "probe": {}, "extract": {"mode": "probe"}}
    with pytest.raises(ValueError, match="pick one"):
        cli.load_config(_write_cfg(tmp_path, c))


def test_load_config_probe_block_and_probe_mode_go_together(tmp_path):
    with pytest.raises(ValueError, match="extract.mode: probe"):
        cli.load_config(_write_cfg(tmp_path, {"run_name": "x", "probe": {}, "extract": {"mode": "corpus"}}))
    with pytest.raises(ValueError, match="extract.mode: probe"):
        cli.load_config(_write_cfg(tmp_path, {"run_name": "x", "extract": {"mode": "probe"}}))
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `uv run pytest tests/test_probe.py tests/test_cli.py -q -k "words or probe or pick_one or together"`
Expected: FAIL / ERROR (`token_drift.probe` doesn't exist, configs missing).

- [ ] **Step 3: Write the word file and configs**

`configs/probe_words.yaml` (first draft = the spec's two tables; Andrey edits it after the count report):

```yaml
# Milestone B word file. Each row is deficiency / mean / excess (the virtue is the middle one).
# Words are bare lowercase; the code prepends the space (" courage"). Rows whose words fall below
# metrics.min_count in a source group are reported as missing for that group, not dropped here.
# Spec: docs/superpowers/specs/2026-10-03-v1-milestone-b-design.md
sets:
  classical:            # Chase's Aristotle + Aquinas's wording
    fear:
      noun: [[cowardice, courage, rashness],
             [fear, courage, fearlessness],          # flagged: "fear" is the passion, the vice is cowardice; kept because the Pile has it
             [timidity, fortitude, daring]]
      adj:  [[cowardly, brave, rash]]
    pleasure:
      noun: [[insensibility, temperance, intemperance]]
      adj:  [[insensible, temperate, intemperate]]
    giving:
      noun: [[covetousness, liberality, prodigality],
             [avarice, liberality, prodigality]]
      adj:  [[covetous, liberal, prodigal],
             [stingy, liberal, prodigal]]
    spending:
      noun: [[meanness, magnificence, vulgarity]]
    honour:
      noun: [[pusillanimity, magnanimity, presumption],
             [pusillanimity, magnanimity, vainglory],
             [pusillanimity, magnanimity, vanity]]
    anger:
      noun: [[insensibility, meekness, anger]]       # flagged: "insensibility" is also pleasure's deficiency (Aquinas)
      adj:  [[spiritless, meek, passionate]]
    truth:
      noun: [[irony, truthfulness, boasting]]
    shame:
      adj:  [[shameless, modest, bashful]]
    neighbour:
      noun: [[spite, indignation, envy]]
  everyday:             # modern words for the same concepts
    fear:
      noun: [[cowardice, bravery, recklessness]]
      adj:  [[cowardly, brave, reckless]]
    money:
      noun: [[stinginess, generosity, extravagance]]
      adj:  [[stingy, generous, wasteful],
             [stingy, generous, extravagant]]
    pleasure:
      noun: [[asceticism, moderation, indulgence]]
      adj:  [[ascetic, moderate, indulgent]]
    self:
      noun: [[insecurity, confidence, arrogance]]
      adj:  [[insecure, confident, arrogant],
             [timid, confident, arrogant]]
    anger:
      noun: [[apathy, patience, irritability]]
      adj:  [[apathetic, patient, irritable],
             [passive, calm, irritable]]
    truth:
      noun: [[secrecy, honesty, boastfulness]]
      adj:  [[secretive, honest, boastful]]
    wit:
      adj:  [[boring, witty, silly]]
    company:
      noun: [[hostility, friendliness, obsequiousness]]
      adj:  [[hostile, friendly, obsequious]]
    drive:
      noun: [[laziness, ambition, ruthlessness]]
      adj:  [[lazy, ambitious, ruthless]]
    shame:
      noun: [[shyness, modesty, shamelessness]]
      adj:  [[shy, modest, shameless]]
# A's polysemy trajectory group; extracted now, analysed in B2
polysemy: [bank, bat, spring, bass, match, light, cell, plant, mouse, python, apple, java,
           pitch, current, table, net, tie, seal, mole, trunk, bark, date, fair, kind, saw,
           lead, right, left]
null_exclude: []        # junk the suffix heuristic lets into the null pool; add and rerun probe_corpus
```

`configs/pythia70m_probe.yaml`:

```yaml
# v1 milestone B: per-occurrence vectors for the Q16 triples (virtue between its vices), from
# targeted retrieval in moral-philosophy books + 3 Pile shards.
# Spec: docs/superpowers/specs/2026-10-03-v1-milestone-b-design.md
run_name: pythia70m_probe
model: EleutherAI/pythia-70m
random_init: false
seed: 0
device: auto

probe:
  words: configs/probe_words.yaml
  cache_dir: ~/.cache/token-drift      # Gutenberg downloads; outside the repo
  books:                               # start/end = marker lines (first exact match, start kept, end cut);
    - {name: chase_ethics, gutenberg: 8438}          # without them: the whole Gutenberg body
    - {name: summa_1_2, gutenberg: 17897, start: "QUESTION 49", end: "QUESTION 90"}
    - {name: summa_2_2, gutenberg: 18755,
       start: "TREATISE ON THE THEOLOGICAL VIRTUES (QQ. 1-46)",
       end: "TREATISE ON GRATUITOUS GRACES (QQ. 171-182)"}
    - {name: republic, gutenberg: 1497}
    - {name: utilitarianism, gutenberg: 11224}
    - {name: kant_groundwork, gutenberg: 5682}
    - {name: seneca_morals, gutenberg: 56075}
    - {name: epictetus, gutenberg: 10661}
    - {name: hume_morals, gutenberg: 4320}
    - {name: smith_sentiments, gutenberg: 67363}
  pile:
    source: EleutherAI/the_pile_deduplicated
    text_field: text
    shards: 3                          # ~121M tokens each; files in sorted order
  cap: 1000                            # per word per source group (books | pile)
  min_context: 32                      # same as A: positions < 32 are attention sinks
  max_context: null                    # null = model.config.max_position_embeddings

extract:
  mode: probe                          # vocab (v0) | corpus (A) | probe (B)
  batch_size: 16                       # windows per forward, longest first
  dtype_on_disk: float16

metrics:
  min_count: 20                        # per word per source group
  null_pool: 50                        # words per part of speech
  null_k: 20                           # nearest-in-log-count pool words per triple
  between_pct: 5                       # "beats the null" = closer than every one of the 20 null words

viz:
  occ_frames: [0, 3, 6]                # frames for the per-occurrence histograms

out_dir: runs
```

`configs/random_init_probe.yaml`: identical except these two lines and the header comment:

```yaml
# Control: same windows, random weights. Whatever "betweenness" survives here is the
# architecture (or the tokenizer) talking, not anything learned.
run_name: random_init_probe
model: EleutherAI/pythia-70m
random_init: true
```

(copy every other line of `pythia70m_probe.yaml` verbatim below these.)

- [ ] **Step 4: Write `probe.py` (first part) and the config checks**

`src/token_drift/probe.py`:

```python
"""Milestone B: find a word's occurrences in targeted texts and cut a window ending at each.

Pythia is causal, so a hit's residual only depends on the tokens before it: find the hits
first, then run the model on just those windows instead of on billions of tokens.
Spec: docs/superpowers/specs/2026-10-03-v1-milestone-b-design.md
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

POS = ("noun", "adj")
ROLES = ("deficiency", "mean", "excess")
GROUPS = ("books", "pile")
_BARE = re.compile(r"[a-z][a-z'-]*")


def load_words(path) -> dict:
    """configs/probe_words.yaml -> {"triples": [...], "polysemy": [...], "null_exclude": [...]}.

    Triples come out flat, in file order: {"id", "set", "concept", "pos", "words"} with
    words = (deficiency, mean, excess).
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    triples = []
    for set_name, concepts in (raw.get("sets") or {}).items():
        for concept, by_pos in concepts.items():
            for pos, rows in by_pos.items():
                where = f"{set_name}/{concept}/{pos}"
                if pos not in POS:
                    raise ValueError(f"{where}: part of speech {pos!r} must be one of {POS}")
                for row in rows:
                    if len(row) != 3:
                        raise ValueError(f"{where}: {row} is not deficiency/mean/excess")
                    triples.append({"id": f"{where}/{','.join(row)}", "set": set_name,
                                    "concept": concept, "pos": pos, "words": tuple(row)})
    words = {"triples": triples, "polysemy": list(raw.get("polysemy") or []),
             "null_exclude": list(raw.get("null_exclude") or [])}
    for w in [w for t in triples for w in t["words"]] + words["polysemy"]:
        # capitals and leading spaces are different tokens; the code adds the one space itself
        if not isinstance(w, str) or not _BARE.fullmatch(w):
            raise ValueError(f"probe word {w!r} must be bare lowercase (the space is added by the code)")
    return words


def probe_word_pos(words: dict) -> dict[str, str]:
    """Every triple word -> its part of speech, each word once (courage is in 3 triples)."""
    out: dict[str, str] = {}
    for t in words["triples"]:
        for w in t["words"]:
            out.setdefault(w, t["pos"])
    return out
```

In `src/token_drift/cli.py`, replace `load_config` with:

```python
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
```

and in `stage_extract` replace `mode = cfg["extract"].get("mode", "vocab")  # v0 configs don't have the key` with `mode = _mode(cfg)`.

In `tests/conftest.py`, add `import pytest` at the top and append:

```python
@pytest.fixture(scope="session")
def pythia_tok():
    """The real Pythia tokenizer, from the HF cache only (tests never hit the network)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", local_files_only=True)
    except OSError:
        pytest.skip("EleutherAI/pythia-70m tokenizer is not in the HF cache")
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`
Expected: all pass (old tests included).

- [ ] **Step 6: Commit**

```bash
git add configs/probe_words.yaml configs/pythia70m_probe.yaml configs/random_init_probe.yaml \
  src/token_drift/probe.py src/token_drift/cli.py tests/conftest.py tests/test_probe.py tests/test_cli.py
git commit -m "probe: word file draft, B configs, config checks

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Books in, hits out (Gutenberg, unwrap, token matching)

**Files:**
- Modify: `src/token_drift/probe.py`
- Test: `tests/test_probe.py` (append)

**Interfaces:**
- Consumes: `pythia_tok` fixture (Task 1).
- Produces:
  - `fetch_gutenberg(book_id: int, cache_dir: str | Path) -> str`
  - `gutenberg_body(text: str, start: str | None = None, end: str | None = None) -> str` (raises `ValueError` naming a missing marker)
  - `unwrap(text: str) -> str`
  - `to_uint16(ids) -> np.ndarray[uint16]` (raises if an id > 65535)
  - `concat_docs(docs: list[array-like]) -> (ids: uint16[n], offsets: int64[n_docs + 1])`
  - `word_token_ids(tokenizer, words: list[str], *, eos_id: int, prefix: str = " ") -> dict[str, tuple[int, ...]]`
  - `boundary_table(tokenizer) -> bool[len(tokenizer)]` (True = a hit followed by this id is a whole word)
  - `find_hits(ids, offsets, patterns: list[tuple[int, ...]], ok_next) -> (pat: int64[h], doc: int64[h], end: int64[h])`, `end` = global index of the last piece, sorted by pattern then position

- [ ] **Step 1: Write the failing tests** (append to `tests/test_probe.py`; add `import io` and `import numpy as np` to its imports)

```python
# ---------- Gutenberg ----------

BOOK = ("The Project Gutenberg eBook\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n"
        "intro\r\nQUESTION 49\r\nhabit is\r\na quality\r\n\r\nnext para\r\nQUESTION 90\r\nlaw\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\nlicense\r\n")


def test_gutenberg_body_cuts_header_and_footer():
    assert pb.gutenberg_body(BOOK) == "intro\nQUESTION 49\nhabit is\na quality\n\nnext para\nQUESTION 90\nlaw"


def test_gutenberg_body_markers_start_inclusive_end_exclusive():
    assert pb.gutenberg_body(BOOK, "QUESTION 49", "QUESTION 90") == "QUESTION 49\nhabit is\na quality\n\nnext para"


def test_gutenberg_body_missing_marker_says_which():
    with pytest.raises(ValueError, match="QUESTION 12"):
        pb.gutenberg_body(BOOK, "QUESTION 12")
    with pytest.raises(ValueError, match="START OF"):
        pb.gutenberg_body("no markers here")


def test_unwrap_joins_hard_wrapped_lines_keeps_paragraphs():
    # Gutenberg wraps at ~70 chars; a line-start word would tokenize without its space
    assert pb.unwrap("the soldier showed\ncourage here\n\n  next\npara  ") == "the soldier showed courage here\n\nnext para"


def test_fetch_gutenberg_downloads_once_then_uses_the_cache(tmp_path, monkeypatch):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO("﻿hello".encode("utf-8"))  # Gutenberg files often start with a BOM

    monkeypatch.setattr(pb.urllib.request, "urlopen", fake_urlopen)
    assert pb.fetch_gutenberg(8438, tmp_path) == "hello"
    assert calls == ["https://www.gutenberg.org/cache/epub/8438/pg8438.txt"]

    def offline(url, timeout):
        raise OSError("no network")

    monkeypatch.setattr(pb.urllib.request, "urlopen", offline)
    assert pb.fetch_gutenberg(8438, tmp_path) == "hello"  # cached copy, no network needed


# ---------- token matching ----------

def test_concat_docs_and_uint16_overflow():
    ids, offs = pb.concat_docs([[1, 2], [], [3]])
    assert ids.dtype == np.uint16 and ids.tolist() == [1, 2, 3] and offs.tolist() == [0, 2, 2, 3]
    with pytest.raises(ValueError, match="uint16"):
        pb.to_uint16([70000])


def test_word_token_ids_space_prefix_and_eos_check(pythia_tok):
    ids = pb.word_token_ids(pythia_tok, ["courage", "temperance"], eos_id=pythia_tok.eos_token_id)
    assert pythia_tok.convert_ids_to_tokens(list(ids["courage"])) == ["Ġcourage"]
    assert len(ids["temperance"]) == 2  # " temper" + "ance": multi-token, read at the last piece
    bare = pb.word_token_ids(pythia_tok, ["Courage"], eos_id=pythia_tok.eos_token_id, prefix="")
    assert bare["Courage"] != ids["courage"]
    with pytest.raises(ValueError, match="EOS"):
        pb.word_token_ids(pythia_tok, ["<|endoftext|>"], eos_id=pythia_tok.eos_token_id, prefix="")


def test_boundary_table(pythia_tok):
    ok = pb.boundary_table(pythia_tok)
    tid = lambda s: pythia_tok.convert_tokens_to_ids(s)  # noqa: E731
    assert ok[tid(",")] and ok[tid("Ġthe")] and ok[pythia_tok.eos_token_id]
    assert not ok[tid("ous")] and not ok[tid("-")] and not ok[tid("'s")] and not ok[tid("7")]


def _docs(tok, *texts):
    return pb.concat_docs([tok(t, add_special_tokens=False)["input_ids"] for t in texts])


def test_find_hits_whole_lowercase_words_only(pythia_tok):
    ids, offs = _docs(pythia_tok, "He showed courage, not courageous zeal; Courage too. self-mastery and self.")
    ok = pb.boundary_table(pythia_tok)
    w = pb.word_token_ids(pythia_tok, ["courage", "self"], eos_id=0)
    pat, doc, end = pb.find_hits(ids, offs, [w["courage"], w["self"]], ok)
    # courageous, Courage and self-mastery are all skipped
    assert pat.tolist() == [0, 1] and doc.tolist() == [0, 0]
    assert ids[end[0]] == w["courage"][-1]


def test_find_hits_multi_token_word_ends_at_its_last_piece(pythia_tok):
    ids, offs = _docs(pythia_tok, "true temperance is rare, and a temper is not")
    ok = pb.boundary_table(pythia_tok)
    w = pb.word_token_ids(pythia_tok, ["temperance", "temper"], eos_id=0)
    pat, _, end = pb.find_hits(ids, offs, [w["temperance"], w["temper"]], ok)
    # " temper" inside "temperance" is followed by "ance" (a letter), so it isn't a hit
    assert pat.tolist() == [0, 1]
    assert tuple(ids[end[0] - 1 : end[0] + 1]) == w["temperance"]


def test_find_hits_never_spans_two_documents():
    ids, offs = pb.concat_docs([[5, 7], [8, 9]])
    pat, doc, end = pb.find_hits(ids, offs, [(7, 8)], np.ones(10, bool))
    assert len(end) == 0


def test_find_hits_word_at_the_end_of_a_doc_counts():
    # Review Focus 3: no next token to check -> whole word
    ids, offs = pb.concat_docs([[1, 2, 3], [4, 3, 5]])
    ok = np.zeros(10, bool)  # every next token "continues the word"
    pat, doc, end = pb.find_hits(ids, offs, [(3,)], ok)
    assert end.tolist() == [2] and doc.tolist() == [0]


def test_find_hits_empty_doc_in_the_middle_keeps_doc_indices():
    # Review Focus 2: the Pile has empty texts
    ids, offs = pb.concat_docs([[1, 3], [], [3, 1]])
    _, doc, end = pb.find_hits(ids, offs, [(3,)], np.ones(10, bool))
    assert doc.tolist() == [0, 2] and end.tolist() == [1, 2]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_probe.py -q`
Expected: the new tests FAIL with `AttributeError: module 'token_drift.probe' has no attribute ...`.

- [ ] **Step 3: Implement** (append to `src/token_drift/probe.py`; add `import urllib.request` to the imports)

```python
# ---------- books ----------

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def fetch_gutenberg(book_id: int, cache_dir) -> str:
    """Plain text of a Gutenberg book, downloaded once into cache_dir (outside the repo)."""
    path = Path(cache_dir).expanduser() / f"pg{book_id}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(GUTENBERG_URL.format(id=book_id), timeout=60) as r:
            data = r.read()
        tmp = path.with_suffix(".part")  # a killed download never looks like a cached book
        tmp.write_bytes(data)
        tmp.replace(path)
    return path.read_text(encoding="utf-8-sig")  # -sig: drop the BOM many Gutenberg files have


def _first_line(lines: list[str], pred, frm: int, what: str) -> int:
    for i in range(frm, len(lines)):
        if pred(lines[i]):
            return i
    raise ValueError(f"marker {what!r} not found")


def gutenberg_body(text: str, start: str | None = None, end: str | None = None) -> str:
    """Text between the *** START / *** END lines, then from the first line equal to `start`
    (kept) up to the first line equal to `end` after it (cut). Raises naming a missing marker."""
    lines = text.replace("\r\n", "\n").split("\n")
    a = _first_line(lines, lambda s: s.startswith("*** START OF"), 0, "*** START OF") + 1
    b = _first_line(lines, lambda s: s.startswith("*** END OF"), a, "*** END OF")
    body = lines[a:b]
    if start is not None:
        body = body[_first_line(body, lambda s: s.strip() == start, 0, start):]
    if end is not None:
        body = body[:_first_line(body, lambda s: s.strip() == end, 1, end)]
    return "\n".join(body).strip("\n")


def unwrap(text: str) -> str:
    """Hard-wrapped lines -> paragraphs: single newlines become spaces, blank lines stay.

    Gutenberg wraps at ~70 chars, so ~8% of words start a line (Summa I-II, measured), and a
    line-start word tokenizes without its space ("courage", not " courage"): we'd miss it.
    """
    paras = re.split(r"\n\s*\n", text.strip())
    return "\n\n".join(re.sub(r"\s*\n\s*", " ", p) for p in paras)


# ---------- token matching ----------

def to_uint16(ids) -> np.ndarray:
    """Token ids as uint16 (half the RAM of int32 for 360M Pile tokens). Pythia/GPT-2 fit."""
    a = np.asarray(ids, dtype=np.int64)
    if a.size and a.max() > np.iinfo(np.uint16).max:
        raise ValueError(f"token id {a.max()} doesn't fit in uint16; switch the dtype for this tokenizer")
    return a.astype(np.uint16)


def concat_docs(docs) -> tuple[np.ndarray, np.ndarray]:
    """List of token-id docs -> (flat uint16 ids, int64 offsets with offsets[i]:offsets[i+1] = doc i)."""
    parts = [to_uint16(d) for d in docs]
    offsets = np.zeros(len(parts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(p) for p in parts])
    ids = np.concatenate(parts) if parts else np.empty(0, np.uint16)
    return ids, offsets


def word_token_ids(tokenizer, words, *, eos_id: int, prefix: str = " ") -> dict[str, tuple[int, ...]]:
    """Token ids of prefix + word for each word (prefix "" for the capitalized-skip count)."""
    out = {}
    for w in words:
        ids = tuple(tokenizer(prefix + w, add_special_tokens=False)["input_ids"])
        if not ids or eos_id in ids:
            raise ValueError(f"{prefix + w!r} tokenizes to {ids}: empty or contains EOS")
        out[w] = ids
    return out


def boundary_table(tokenizer) -> np.ndarray:
    """ok[i] = a hit followed by token i is a whole word: token i doesn't start with a letter,
    digit, hyphen or apostrophe. Else " courage" matches inside "courageous" and " self"
    inside "self-mastery". Decided once per id on the decoded string."""
    ok = np.ones(len(tokenizer), dtype=bool)
    for i in range(len(tokenizer)):
        s = tokenizer.decode([i])
        if s and (s[0].isalnum() or s[0] in "-'’"):
            ok[i] = False
    return ok


def find_hits(ids, offsets, patterns, ok_next) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every whole-word occurrence of each pattern (a tuple of ids), inside one document.

    -> (pattern index, doc index, global index of the pattern's last piece). One vectorized
    scan per pattern; a hit on a doc's last token counts (nothing follows it).
    """
    ids = np.asarray(ids)
    offs = np.asarray(offsets, dtype=np.int64)
    n = len(ids)
    pats, docs, ends = [], [], []
    for p, pat in enumerate(patterns):
        k = len(pat)
        end = np.flatnonzero(ids[k - 1:] == pat[-1]) + (k - 1)
        for j in range(1, k):
            end = end[ids[end - j] == pat[-1 - j]]
        doc = np.searchsorted(offs, end, side="right") - 1  # empty docs never contain `end`
        inside = end - (k - 1) >= offs[doc]
        nxt = end + 1
        whole = (nxt >= offs[doc + 1]) | ok_next[ids[np.minimum(nxt, n - 1)]]
        keep = inside & whole
        pats.append(np.full(int(keep.sum()), p, dtype=np.int64))
        docs.append(doc[keep])
        ends.append(end[keep])
    cat = lambda xs: np.concatenate(xs) if xs else np.empty(0, np.int64)  # noqa: E731
    return cat(pats), cat(docs).astype(np.int64), cat(ends).astype(np.int64)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_probe.py -q`
Expected: PASS (the tokenizer tests skip only if the Pythia tokenizer isn't cached; it is on Andrey's box).

- [ ] **Step 5: Commit**

```bash
git add src/token_drift/probe.py tests/test_probe.py
git commit -m "probe: Gutenberg fetch/slice/unwrap, whole-word token matching

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Counts, null pool, sampling, windows, snippets

**Files:**
- Modify: `src/token_drift/probe.py`
- Test: `tests/test_probe.py` (append)

**Interfaces:**
- Consumes: `concat_docs`, `find_hits` (Task 2).
- Produces:
  - `count_single_token_words(ids, offsets, ok_next, vocab_size: int) -> int64[vocab_size]`
  - `guess_pos(word: str) -> str | None`
  - `null_candidates(tokens: list[str], counts_books, counts_pile, *, min_count: int, exclude: set[str]) -> dict[pos, dict[word, book_count]]`
  - `pick_null_pool(cands: dict[pos, dict[word, int]], targets: dict[pos, list[int]], *, n: int, seed: int) -> dict[pos, list[str]]` (sorted lists, both POS keys always present)
  - `reservoir(keys: int[h], cap: int, seed: int) -> int64[kept]` (sorted indices into keys)
  - `cut_windows(ids, offsets, doc, end, *, max_context: int, eos_id: int) -> list[np.ndarray[uint16]]`
  - `snippets(tokenizer, windows: list[np.ndarray], n_tokens: int = 48, n_chars: int = 150) -> list[str]`
  - `role_tags(words: dict, pool: dict[pos, list[str]]) -> dict[str, str]`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_probe.py`)

```python
# ---------- pass 1 counts + null pool ----------

def test_count_single_token_words():
    ids, offs = pb.concat_docs([[3, 4, 3], [3, 5]])
    ok = np.array([True, True, True, True, False, True])  # id 4 continues a word
    # doc 0: 3 then 4 -> not whole; 4 then 3 -> whole; 3 at doc end -> whole. doc 1: 3 then 5, 5 at end.
    assert pb.count_single_token_words(ids, offs, ok, vocab_size=6).tolist() == [0, 0, 0, 2, 1, 1]


def test_guess_pos_by_suffix():
    assert pb.guess_pos("kindness") == "noun" and pb.guess_pos("government") == "noun"
    assert pb.guess_pos("famous") == "adj" and pb.guess_pos("possible") == "adj"
    assert pb.guess_pos("table") is None


def test_null_candidates_need_both_groups_and_skip_excluded():
    tokens = [" kindness", " famous", " table", " Goodness", "darkness", " sadness", " courage",
              " ion", " vanity"]
    books = np.array([30, 25, 40, 50, 50, 5, 90, 90, 40])
    pile = np.array([30, 25, 40, 50, 50, 99, 90, 90, 40])
    c = pb.null_candidates(tokens, books, pile, min_count=20, exclude={"vanity"})
    # table: no suffix; Goodness: capital; darkness: no space; sadness: 5 book hits; ion: too short
    assert c == {"noun": {"kindness": 30}, "adj": {"famous": 25}}


def test_pick_null_pool_matches_book_counts_then_fills_seeded():
    nouns = {f"w{i}ness": c for i, c in enumerate([10, 20, 40, 80, 160, 320, 640])}
    cands = {"noun": nouns, "adj": {"aous": 50, "bous": 60}}
    pool = pb.pick_null_pool(cands, {"noun": [39, 700], "adj": []}, n=4, seed=0)
    assert len(pool["noun"]) == 4 and {"w2ness", "w6ness"} <= set(pool["noun"])  # nearest to 39 and 700
    assert pool["adj"] == ["aous", "bous"]  # fewer candidates than n: take them all
    assert pool == pb.pick_null_pool(cands, {"noun": [39, 700], "adj": []}, n=4, seed=0)


# ---------- sampling + windows ----------

def test_reservoir_caps_each_key_seeded_and_uniform():
    keys = np.repeat([0, 1, 2], [5, 3000, 10000])
    kept = pb.reservoir(keys, cap=1000, seed=0)
    assert np.bincount(keys[kept]).tolist() == [5, 1000, 1000]
    assert np.array_equal(kept, pb.reservoir(keys, 1000, 0))
    assert not np.array_equal(kept, pb.reservoir(keys, 1000, 1))
    pos = kept[keys[kept] == 2] - 3005  # where in key 2's stream the picks came from
    hist = np.bincount(pos // 1000, minlength=10)  # expect ~100 per tenth
    assert hist.min() > 60 and hist.max() < 140


def test_cut_windows_caps_context_and_adds_eos_only_at_doc_start():
    ids, offs = pb.concat_docs([np.arange(1, 11)])  # one doc, tokens 1..10 at positions 0..9
    w = pb.cut_windows(ids, offs, np.array([0, 0, 0]), np.array([3, 4, 9]), max_context=6, eos_id=0)
    assert w[0].tolist() == [0, 1, 2, 3, 4]        # whole prefix + EOS fits in 6
    assert w[1].tolist() == [0, 1, 2, 3, 4, 5]     # exactly 6 with the EOS
    assert w[2].tolist() == [5, 6, 7, 8, 9, 10]    # capped at 6, starts mid-doc: no EOS
    assert all(x.dtype == np.uint16 for x in w)


def test_snippets_decode_the_tail(pythia_tok):
    win = np.array(pythia_tok(" the soldier showed courage", add_special_tokens=False)["input_ids"], np.uint16)
    s = pb.snippets(pythia_tok, [np.r_[np.uint16(0), win]], n_chars=20)
    assert s == ["soldier showed courage"[-20:]]


def test_role_tags():
    words = {"triples": [{"set": "classical", "concept": "fear", "pos": "noun",
                          "words": ("cowardice", "courage", "rashness")}],
             "polysemy": ["bank", "courage"]}
    tags = pb.role_tags(words, {"noun": ["kindness"], "adj": []})
    assert tags["courage"] == "classical/fear/noun/mean;polysemy"
    assert tags["rashness"] == "classical/fear/noun/excess"
    assert tags["kindness"] == "null:noun" and tags["bank"] == "polysemy"
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_probe.py -q`
Expected: new tests FAIL with `AttributeError`.

- [ ] **Step 3: Implement** (append to `src/token_drift/probe.py`)

```python
# ---------- pass 1: counts and the null pool ----------

def count_single_token_words(ids, offsets, ok_next, vocab_size: int) -> np.ndarray:
    """Whole-word hits of every id at once (pass 1, for the null pool): a position counts if
    the next token doesn't continue the word, or it's the doc's last token."""
    ids = np.asarray(ids)
    offs = np.asarray(offsets, dtype=np.int64)
    whole = np.ones(len(ids), dtype=bool)
    whole[:-1] = ok_next[ids[1:]]
    last = offs[1:][offs[1:] > offs[:-1]] - 1  # skip empty docs
    whole[last] = True
    return np.bincount(ids[whole], minlength=vocab_size).astype(np.int64)


# Part of speech by suffix. Crude on purpose: the pool only has to be "abstract nouns" and
# "adjectives" in bulk; junk that slips through goes in the word file's null_exclude.
SUFFIXES = {
    "noun": ("ness", "ity", "ence", "ance", "ion", "ment", "ism", "ship", "dom"),
    "adj": ("ous", "ful", "ive", "ent", "ant", "less", "able", "ible", "al", "ic", "ish"),
}


def guess_pos(word: str) -> str | None:
    for pos in POS:  # nouns first: "-ment" also ends in the adjective suffix "-ent"
        if word.endswith(SUFFIXES[pos]):
            return pos
    return None


def null_candidates(tokens, counts_books, counts_pile, *, min_count: int, exclude: set[str]) -> dict:
    """Single-token lowercase words (" kindness") with >= min_count whole-word hits in both
    groups, not in the word file -> {pos: {word: book count}}."""
    out: dict[str, dict[str, int]] = {p: {} for p in POS}
    for i, s in enumerate(tokens):
        if not s.startswith(" "):
            continue
        w = s[1:]
        if len(w) < 5 or not (w.isascii() and w.isalpha() and w.islower()) or w in exclude:
            continue
        if counts_books[i] < min_count or counts_pile[i] < min_count:
            continue
        pos = guess_pos(w)
        if pos is not None:
            out[pos][w] = int(counts_books[i])
    return out


def pick_null_pool(cands: dict, targets: dict, *, n: int, seed: int) -> dict[str, list[str]]:
    """Per pos, n pool words whose book counts look like the probe words' book counts.

    Books are the binding group (most candidates are plentiful in the Pile): for each probe
    word, the unused candidate nearest in log book-count, then a seeded random fill up to n.
    """
    rng = np.random.default_rng(seed)
    pool = {}
    for pos in POS:
        words = sorted(cands.get(pos, {}))
        logc = np.log(np.maximum([cands[pos][w] for w in words], 1)) if words else np.empty(0)
        used = np.zeros(len(words), dtype=bool)
        for t in sorted(targets.get(pos, [])):
            if used.sum() >= n or used.all():
                break
            dist = np.abs(logc - np.log(max(t, 1)))
            dist[used] = np.inf
            used[int(np.argmin(dist))] = True
        rest = np.flatnonzero(~used)
        used[rng.permutation(rest)[: max(0, n - int(used.sum()))]] = True
        pool[pos] = [w for w, u in zip(words, used) if u]
    return pool


# ---------- sampling and windows ----------

def reservoir(keys, cap: int, seed: int) -> np.ndarray:
    """At most `cap` items per key, uniform without replacement, seeded -> sorted indices.

    Bottom-k by random priority: the same distribution reservoir sampling gives, in one
    vectorized pass, since all hits fit in memory anyway.
    """
    keys = np.asarray(keys)
    if len(keys) == 0:
        return np.empty(0, dtype=np.int64)
    pri = np.random.default_rng(seed).random(len(keys))
    order = np.lexsort((pri, keys))  # by key, then priority
    sk = keys[order]
    first = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1]
    rank = np.arange(len(sk)) - np.repeat(first, np.diff(np.r_[first, len(sk)]))
    return np.sort(order[rank < cap]).astype(np.int64)


def cut_windows(ids, offsets, doc, end, *, max_context: int, eos_id: int) -> list[np.ndarray]:
    """One window per hit, ending at the hit's last piece: EOS + the whole doc prefix if that
    fits in max_context (a doc start always follows an EOS in Pythia's training data),
    else the last max_context tokens. Hits are already filtered for min_context."""
    offs = np.asarray(offsets, dtype=np.int64)
    eos = np.array([eos_id], dtype=np.uint16)
    out = []
    for d, e in zip(np.asarray(doc), np.asarray(end)):
        start = offs[d]
        if e - start + 2 <= max_context:
            out.append(np.concatenate([eos, ids[start : e + 1]]))
        else:
            out.append(np.asarray(ids[e + 1 - max_context : e + 1], dtype=np.uint16))
    return out


def snippets(tokenizer, windows, n_tokens: int = 48, n_chars: int = 150) -> list[str]:
    """Decoded tail of each window, for reading the sense of an occurrence later."""
    texts = tokenizer.batch_decode([w[-n_tokens:].tolist() for w in windows], skip_special_tokens=True)
    return [" ".join(t.split())[-n_chars:] for t in texts]


def role_tags(words: dict, pool: dict) -> dict[str, str]:
    """word -> "set/concept/pos/role;..." | "null:<pos>" | "polysemy", for occ_meta.parquet."""
    tags: dict[str, list[str]] = {}
    for t in words["triples"]:
        for role, w in zip(ROLES, t["words"]):
            tags.setdefault(w, []).append(f"{t['set']}/{t['concept']}/{t['pos']}/{role}")
    for w in words.get("polysemy", []):
        tags.setdefault(w, []).append("polysemy")
    for pos, ws in pool.items():
        for w in ws:
            tags.setdefault(w, []).append(f"null:{pos}")
    return {w: ";".join(dict.fromkeys(v)) for w, v in tags.items()}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_probe.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/token_drift/probe.py tests/test_probe.py
git commit -m "probe: pass-1 counts, null pool, capped sampling, windows, snippets

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `probe_corpus` stage end to end (+ count report)

**Files:**
- Modify: `src/token_drift/probe.py` (add `count_report`)
- Modify: `src/token_drift/cli.py` (stage, helpers, `probe_corpus` command, `all` dispatch)
- Create: `tests/test_probe_stages.py`
- Test: `tests/test_probe.py` (append), `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: everything in `probe.py` so far; `corpus.iter_parquet`, `corpus.read_parquet`; `extract.vocab_tokens`.
- Produces:
  - `probe.count_report(words: dict, counts: dict, pool: dict[pos, list[str]], *, min_count: int) -> str`
  - `cli.stage_probe_corpus(cfg) -> Path` writing to `runs/<run>/probe_corpus/`:
    `windows_tokens.npy` (uint16, flat), `windows_offsets.npy` (int64, n_windows + 1),
    `occ_meta.parquet` (columns `word, role_tags, group, source, doc, position, n_pieces, snippet`;
    rows in window order), `counts.json` (`{word: {group: {"found", "context_ok", "kept", "capitalized"}}}`),
    `null_pool.json` (`{pos: [{"word", "books", "pile"}]}` with found counts),
    `books.json` (`{name: {"gutenberg", "start", "end", "sha256", "n_tokens"}}`),
    `probe_words.yaml` (copy), `meta.json`, `count_report.md`
  - `cli._tokenize_texts(tok, texts: list[str], chunk: int = 1000) -> list[np.ndarray[uint16]]`
  - `cli._model_max_context(name: str) -> int`
  - `cli.GROUPS` (= `probe.GROUPS`)
  - `tests/test_probe_stages.py`: fixture `probe_cfg` (a config path; tokenizer, Gutenberg and model monkeypatched) reused by Tasks 5, 7, 8, 9

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_probe.py`:

```python
def test_count_report_marks_short_words_and_lists_the_pool():
    words = {"triples": [
        {"set": "classical", "concept": "fear", "pos": "noun", "words": ("cowardice", "courage", "rashness")},
    ], "polysemy": []}
    c = lambda b, p, cap=0: {"books": {"found": b, "context_ok": b, "kept": b, "capitalized": cap},  # noqa: E731
                             "pile": {"found": p, "context_ok": p, "kept": p, "capitalized": 0}}
    counts = {"cowardice": c(44, 72), "courage": c(118, 944, cap=66), "rashness": c(32, 3),
              "kindness": c(30, 500)}
    md = pb.count_report(words, counts, {"noun": ["kindness"], "adj": []}, min_count=20)
    assert "| fear | noun | cowardice / courage / rashness | 44 / 118 / 32 | 72 / 944 / 3✗ | books |" in md
    assert "| noun | kindness | 30 | 500 |" in md
    assert "| courage | 118 | 66 | 944 | 0 |" in md  # capitalized hits we skipped, per group
```

Create `tests/test_probe_stages.py`:

```python
"""Probe (milestone B) stages end to end on a tiny fake shelf: the real Pythia tokenizer (HF
cache), a 2-layer random GPT-NeoX, two 'books' and a two-shard local 'Pile'. No network."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from token_drift import cli
from conftest import tiny_model

# The hard wrap before "courage" is on purpose: without unwrap() the book hits would be 0.
S = ("Courage is rare. The soldier showed\ncourage and not cowardice or rashness, so a brave man "
     "is neither cowardly nor rash. Business and community and experience and performance and "
     "information and government are famous, careful, active, different, important, possible.")
BOOK = ("Title page\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nPREFACE\n\n"
        + "\n\n".join([S] * 30) + "\n\nTHE END\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nlicense\n")
WORDS = """
sets:
  classical:
    fear:
      noun: [[cowardice, courage, rashness]]
      adj: [[cowardly, brave, rash]]
  everyday:
    fear:
      noun: [[timidity, courage, rashness]]
polysemy: [bank]
null_exclude: [government]
"""


@pytest.fixture
def probe_cfg(tmp_path, monkeypatch, pythia_tok):
    monkeypatch.setattr(cli, "_load_tokenizer", lambda name: pythia_tok)
    monkeypatch.setattr(cli.pb, "fetch_gutenberg", lambda book_id, cache_dir: BOOK)
    model = tiny_model(pythia_tok, window=64)
    monkeypatch.setattr(cli.ex, "build_model", lambda name, **kw: (model, pythia_tok))
    wf = tmp_path / "words.yaml"
    wf.write_text(WORDS)
    src = tmp_path / "pile"
    src.mkdir()
    doc = S.replace("\n", " ")
    pq.write_table(pa.table({"text": [doc + " " + doc] * 20 + [None]}), src / "part0.parquet")
    pq.write_table(pa.table({"text": [doc] * 5}), src / "part1.parquet")  # shard 1: not read
    c = {
        "run_name": "tiny_probe", "model": "from-cache", "random_init": False, "seed": 0, "device": "cpu",
        "probe": {
            "words": str(wf), "cache_dir": str(tmp_path / "cache"),
            "books": [{"name": "book_a", "gutenberg": 1},
                      {"name": "book_b", "gutenberg": 2, "start": "PREFACE", "end": "THE END"}],
            "pile": {"source": str(src), "text_field": "text", "shards": 1},
            "cap": 6, "min_context": 8, "max_context": 64,
        },
        "extract": {"mode": "probe", "batch_size": 4, "dtype_on_disk": "float16"},
        "metrics": {"min_count": 3, "null_pool": 4, "null_k": 2, "between_pct": 5},
        "viz": {"occ_frames": [0, 3]},
        "out_dir": str(tmp_path / "runs"),
    }
    p = tmp_path / "probe.yaml"
    p.write_text(yaml.safe_dump(c))
    return p


def _windows(pc):
    t, o = np.load(pc / "windows_tokens.npy"), np.load(pc / "windows_offsets.npy")
    return [t[o[i]:o[i + 1]] for i in range(len(o) - 1)]


def test_probe_corpus_writes_windows_meta_counts_and_report(probe_cfg, pythia_tok):
    rd = cli.stage_probe_corpus(cli.load_config(probe_cfg))
    pc = rd / "probe_corpus"
    counts = json.loads((pc / "counts.json").read_text())
    # unwrap() recovered the line-start " courage" in both books: 30 paragraphs each
    assert counts["courage"]["books"]["found"] == 60
    assert counts["courage"]["pile"]["found"] == 40  # shard 0 only: 20 docs x 2
    assert counts["courage"]["pile"]["capitalized"] > 0
    assert all(counts[w][g]["kept"] <= 6 for w in counts for g in ("books", "pile"))
    assert counts["bank"]["pile"]["found"] == 0
    meta = pq.read_table(pc / "occ_meta.parquet").to_pydict()
    wins = _windows(pc)
    assert len(wins) == len(meta["word"]) and len(wins) > 0
    ids = {w: tuple(pythia_tok(" " + w, add_special_tokens=False)["input_ids"]) for w in set(meta["word"])}
    for win, w, pos in zip(wins, meta["word"], meta["position"]):
        assert len(win) <= 64 and pos >= 8
        assert tuple(win[-len(ids[w]):]) == ids[w]  # every window ends on its word's last piece
    assert {s for s, g in zip(meta["source"], meta["group"]) if g == "pile"} == {"pile:0"}
    assert set(meta["source"]) >= {"book_a", "book_b"}
    tag = dict(zip(meta["word"], meta["role_tags"]))
    assert tag["courage"] == "classical/fear/noun/mean;everyday/fear/noun/mean"
    pool = json.loads((pc / "null_pool.json").read_text())
    assert len(pool["noun"]) == 4 and "government" not in [r["word"] for r in pool["noun"]]
    assert (pc / "count_report.md").read_text().startswith("# Probe count report")
    assert (pc / "probe_words.yaml").read_text() == WORDS
    books = json.loads((pc / "books.json").read_text())
    assert set(books) == {"book_a", "book_b"} and len(books["book_a"]["sha256"]) == 64


def test_probe_corpus_warns_when_a_cached_book_changed(probe_cfg, monkeypatch, capsys):
    cli.stage_probe_corpus(cli.load_config(probe_cfg))
    capsys.readouterr()
    monkeypatch.setattr(cli.pb, "fetch_gutenberg", lambda book_id, cache_dir: BOOK.replace("rare", "scarce"))
    cli.stage_probe_corpus(cli.load_config(probe_cfg))
    out = capsys.readouterr().out
    assert "WARNING" in out and "book_a" in out


def test_probe_corpus_too_few_null_candidates_fails_before_the_gpu(probe_cfg):
    c = cli.load_config(probe_cfg)
    c["metrics"]["null_k"] = 10
    with pytest.raises(ValueError, match="null-pool"):
        cli.stage_probe_corpus(c)
```

Append to `tests/test_cli.py`:

```python
def test_tokenize_texts_feeds_the_tokenizer_in_chunks():
    # Review Focus 1: a whole Pile shard in one call is ~4 GB of Python ints
    seen = []

    class Tok:
        def __call__(self, texts, add_special_tokens=False):
            seen.append(len(texts))
            return {"input_ids": [[1, 2]] * len(texts)}

    docs = cli._tokenize_texts(Tok(), ["a"] * 25, chunk=10)
    assert seen == [10, 10, 5] and len(docs) == 25 and docs[0].dtype == np.uint16


def test_all_on_a_probe_config_runs_probe_corpus_and_skips_normalize(tmp_path, monkeypatch):
    calls = []
    for s in ("corpus", "probe_corpus", "extract", "normalize", "metrics", "viz"):
        monkeypatch.setattr(cli, f"stage_{s}", lambda c, s=s: calls.append(s))
    c = {"run_name": "x", "probe": {}, "extract": {"mode": "probe"}}
    cli.all(_write_cfg(tmp_path, c))
    assert calls == ["probe_corpus", "extract", "metrics", "viz"]
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_probe.py tests/test_probe_stages.py tests/test_cli.py -q`
Expected: new tests FAIL (`count_report`, `stage_probe_corpus`, `_tokenize_texts` missing; `all` still calls normalize).

- [ ] **Step 3: Implement `count_report`** (append to `src/token_drift/probe.py`)

```python
# ---------- count report ----------

def count_report(words: dict, counts: dict, pool: dict, *, min_count: int) -> str:
    """Markdown for Andrey to review before the GPU pass: kept counts per triple and group
    (✗ = below min_count, so that triple is missing in that group), the null pool, and the
    capitalized / sentence-initial hits we skipped."""
    kept = lambda w, g: counts.get(w, {}).get(g, {}).get("kept", 0)  # noqa: E731
    lines = ["# Probe count report", "",
             f"Kept occurrences per word (after min_context and the cap). ✗ = below "
             f"min_count = {min_count}: the triple is reported missing in that group.", ""]
    for set_name in dict.fromkeys(t["set"] for t in words["triples"]):
        lines += [f"## {set_name}", "", "| concept | pos | deficiency / mean / excess | books | pile | usable in |",
                  "|---|---|---|---|---|---|"]
        for t in (t for t in words["triples"] if t["set"] == set_name):
            cells, usable = [], []
            for g in GROUPS:
                cells.append(" / ".join(f"{kept(w, g)}{'✗' if kept(w, g) < min_count else ''}" for w in t["words"]))
                if all(kept(w, g) >= min_count for w in t["words"]):
                    usable.append(g)
            lines.append(f"| {t['concept']} | {t['pos']} | {' / '.join(t['words'])} | {cells[0]} | {cells[1]} "
                         f"| {' + '.join(usable) or 'neither'} |")
        lines.append("")
    lines += ["## Null pool", "", "Whole-word hits found (before the cap).", "",
              "| pos | word | books | pile |", "|---|---|---|---|"]
    for pos in POS:
        for w in pool.get(pos, []):
            lines.append(f"| {pos} | {w} | {counts[w]['books']['found']} | {counts[w]['pile']['found']} |")
    lines += ["", "## Capitalized / sentence-initial hits skipped", "",
              "| word | books lowercase | books capitalized | pile lowercase | pile capitalized |",
              "|---|---|---|---|---|"]
    for w in dict.fromkeys(w for t in words["triples"] for w in t["words"]):
        c = counts.get(w)
        if c and (c["books"]["capitalized"] or c["pile"]["capitalized"]):
            lines.append(f"| {w} | {c['books']['found']} | {c['books']['capitalized']} "
                         f"| {c['pile']['found']} | {c['pile']['capitalized']} |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Implement the stage** (in `src/token_drift/cli.py`)

Add to the imports: `import hashlib`, `import itertools`, `import pyarrow as pa`, `import pyarrow.parquet as pq`, `from token_drift import probe as pb`. Add `GROUPS = pb.GROUPS` next to `N_CORPUS_BINS`. Then add after `stage_corpus`:

```python
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


def _load_books(p: dict, tok, out: Path) -> tuple[list[np.ndarray], list[str]]:
    """Each book = one document. Warns if a cached download differs from the run's last one."""
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
    prev.write_text(json.dumps(meta, indent=1))
    return docs, names


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
    # metrics reads this copy: the list these occurrences were cut for, whatever the yaml says later
    shutil.copy(p["words"], out / "probe_words.yaml")
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

    book_docs, book_names = _load_books(p, tok, out)
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
    (out / "counts.json").write_text(json.dumps(counts, indent=1))
    (out / "null_pool.json").write_text(json.dumps(
        {pos: [{"word": w, "books": counts[w]["books"]["found"], "pile": counts[w]["pile"]["found"]}
               for w in pool[pos]] for pos in pb.POS}, indent=1))
    (out / "count_report.md").write_text(pb.count_report(words, counts, pool, min_count=m["min_count"]))
    (out / "meta.json").write_text(json.dumps({
        "max_context": int(max_ctx), "n_windows": len(windows), "n_window_tokens": int(len(wt)),
        "n_tokens": {g: int(len(corpora[g][0])) for g in GROUPS}, "seconds": round(time.time() - t0),
    }, indent=1))
    typer.echo(f"[probe_corpus] {len(windows)} windows, {len(wt)} tokens ({time.time() - t0:.0f}s) -> {out}")
    return rd
```

Replace the `all` command body and add a `probe_corpus` command next to `corpus_cmd`:

```python
@app.command("probe_corpus")
def probe_corpus_cmd(config: Path = _CONFIG):
    """Find probe-word hits in the books + Pile and cut a window ending at each (milestone B)."""
    stage_probe_corpus(load_config(config))
```

```python
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
```

Update the module docstring's first line to list `probe_corpus` and `occ` among the commands.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`
Expected: PASS. If `test_probe_corpus_writes_windows_meta_counts_and_report` fails on the `found == 60`
count, print `pb.unwrap(pb.gutenberg_body(BOOK))[:300]` and the tokens around " courage" before
touching the assertion: the number is what the fixture text says it should be.

- [ ] **Step 6: Commit**

```bash
git add src/token_drift/probe.py src/token_drift/cli.py tests/test_probe.py tests/test_probe_stages.py tests/test_cli.py
git commit -m "cli: probe_corpus stage (books + Pile -> hit windows, counts, null pool, count report)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `extract_probe` + extract/normalize dispatch

**Files:**
- Modify: `src/token_drift/extract.py` (add `extract_probe`)
- Modify: `src/token_drift/cli.py` (`stage_extract`, `stage_normalize`)
- Test: `tests/test_extract.py` (append), `tests/test_probe_stages.py` (append)

**Interfaces:**
- Consumes: `probe_corpus/windows_tokens.npy`, `windows_offsets.npy` (Task 4).
- Produces:
  - `extract.extract_probe(model, tokens, offsets, *, pad_id: int, batch_size: int, device: str, out: np.ndarray | None = None) -> np.ndarray` of shape `(n_layers + 2, n_windows, d_model)` float16, rows in the original window order
  - `runs/<run>/extract/occ.npy` (same shape, float16, written via `open_memmap`), plus the usual `embed.npy`, `unembed.npy`, `final_ln.npz`, `labels.npy`, `freq_ranks.npy`, `freq_bins.npy`, `tokens.json`, `layer_names.json`; **no** `acts.npy` in probe mode

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_extract.py` (add `extract_probe` to the import from `token_drift.extract`):

```python
def _ragged(lengths, seed=0):
    rng = np.random.default_rng(seed)
    wins = [rng.integers(1, VOCAB, size=n) for n in lengths]
    offs = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    return np.concatenate(wins).astype(np.uint16), offs, wins


def test_extract_probe_padded_batches_match_one_window_at_a_time(model):
    # right-padding is safe only because attention is causal; check it instead of trusting it
    tokens, offs, wins = _ragged([3, 16, 7, 12, 5, 16, 9])
    one = extract_probe(model, tokens, offs, pad_id=0, batch_size=1, device="cpu")
    batched = extract_probe(model, tokens, offs, pad_id=0, batch_size=4, device="cpu")
    assert batched.shape == (LAYERS + 2, len(wins), D) and batched.dtype == np.float16
    np.testing.assert_allclose(batched.astype(np.float32), one.astype(np.float32), atol=1e-2)


@torch.no_grad()
def test_extract_probe_reads_each_windows_last_token_in_original_order(model):
    tokens, offs, wins = _ragged([4, 11, 6])
    acts = extract_probe(model, tokens, offs, pad_id=0, batch_size=3, device="cpu")
    for i, w in enumerate(wins):
        hf = model(input_ids=torch.as_tensor(w[None], dtype=torch.long), output_hidden_states=True).hidden_states
        np.testing.assert_allclose(acts[-1, i].astype(np.float32), hf[-1][0, -1].numpy(), atol=1e-2)
        np.testing.assert_allclose(acts[0, i].astype(np.float32), hf[0][0, -1].numpy(), atol=1e-2)


def test_extract_probe_writes_into_a_given_array(model):
    tokens, offs, _ = _ragged([5, 6])
    out = np.zeros((LAYERS + 2, 2, D), dtype=np.float16)
    res = extract_probe(model, tokens, offs, pad_id=0, batch_size=2, device="cpu", out=out)
    assert res is out and np.abs(out).sum() > 0
```

Append to `tests/test_probe_stages.py`:

```python
def test_probe_extract_writes_occ_per_window_and_no_acts(probe_cfg):
    c = cli.load_config(probe_cfg)
    cli.stage_probe_corpus(c)
    rd = cli.stage_extract(c)
    occ = np.load(rd / "extract" / "occ.npy")
    n = len(np.load(rd / "probe_corpus" / "windows_offsets.npy")) - 1
    assert occ.shape == (4, n, 16) and occ.dtype == np.float16  # tiny model: 2 layers -> 4 frames
    assert not (rd / "extract" / "acts.npy").exists()
    assert json.loads((rd / "extract" / "layer_names.json").read_text())[-1] == "unembed"


def test_normalize_refuses_a_probe_run(probe_cfg):
    with pytest.raises(ValueError, match="skip normalize"):
        cli.stage_normalize(cli.load_config(probe_cfg))
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_extract.py tests/test_probe_stages.py -q`
Expected: FAIL (`extract_probe` missing; `stage_extract` rejects mode probe).

- [ ] **Step 3: Implement `extract_probe`** (append to `src/token_drift/extract.py`)

```python
@torch.no_grad()
def extract_probe(
    model, tokens: np.ndarray, offsets: np.ndarray, *, pad_id: int, batch_size: int, device: str,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Residual at each window's last token, same frames as `extract_activations`.

    -> (n_layers+2, n_windows, d) float16, in the original window order. Windows are ragged
    (flat `tokens`, window i = tokens[offsets[i]:offsets[i+1]]); each batch is right-padded with
    `pad_id`. That's safe because attention is causal: padding after the last token can't reach
    it and positions before it don't shift. `out` lets the caller hand in a disk-backed array
    (occ.npy is ~2 GB for the real run).
    """
    tokens = np.asarray(tokens)
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    n = len(lengths)
    if n and lengths.min() < 1:
        raise ValueError("empty window in the probe corpus")
    n_frames = model.config.num_hidden_layers + 2
    if out is None:
        out = np.empty((n_frames, n, model.config.hidden_size), dtype=np.float16)
    # longest first: batches hold similar lengths (little padding), and an out-of-memory shows
    # up on batch 1 instead of 15 minutes in
    order = np.argsort(-lengths, kind="stable")
    grabbed: dict[str, torch.Tensor] = {}
    hook = final_norm(model).register_forward_pre_hook(
        lambda mod, args: grabbed.__setitem__("pre_ln", args[0])
    )
    try:
        for s in tqdm(range(0, n, batch_size), desc="extract", unit="batch"):
            idx = order[s : s + batch_size]
            batch = np.full((len(idx), int(lengths[idx].max())), pad_id, dtype=np.int64)
            for r, i in enumerate(idx):
                batch[r, : lengths[i]] = tokens[offsets[i] : offsets[i + 1]]
            ids = torch.as_tensor(batch, device=device)
            hs = model.base_model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
            frames = [*hs[:-1], grabbed.pop("pre_ln"), hs[-1]]
            rows = torch.arange(len(idx), device=device)
            last = torch.as_tensor(lengths[idx] - 1, device=device)
            for f, h in enumerate(frames):
                out[f, idx] = h[rows, last].to(torch.float16).cpu().numpy()
    finally:
        hook.remove()
    return out
```

- [ ] **Step 4: Dispatch in `cli.py`**

In `stage_extract`:
1. change the mode check to
   ```python
   if mode not in ("vocab", "corpus", "probe"):
       raise ValueError(f"extract.mode must be vocab, corpus or probe, got {mode!r}")
   ```
2. set `acts = None` before `if mode == "vocab":`, turn the corpus `else:` into `elif mode == "corpus":`, and add:
   ```python
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
   ```
3. replace `np.save(out / "acts.npy", acts)` with `if acts is not None:` + that line, and the final echo with
   `typer.echo(f"[extract] done in {time.time() - t0:.0f}s -> {out}")`.

At the top of `stage_normalize`, after `rd = _prepare_run_dir(cfg)`:

```python
    if _mode(cfg) == "probe":
        raise ValueError("probe runs skip normalize: t, d and seg are ratios of differences, so a "
                         "shared offset (the anisotropy cone) or a uniform scale can't move them")
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/token_drift/extract.py src/token_drift/cli.py tests/test_extract.py tests/test_probe_stages.py
git commit -m "extract: probe mode (ragged windows, last-token residual per occurrence -> occ.npy)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `betweenness.py`

**Files:**
- Create: `src/token_drift/betweenness.py`
- Test: `tests/test_betweenness.py`

**Interfaces:**
- Consumes: `extract.self_similarity(sq_norm, n)`.
- Produces:
  - `ENDS = {0: (1, 2), 1: (0, 2), 2: (0, 1)}` (placement k = word k in the middle)
  - `unit_mean(occ, word_idx, group_idx, n_words, n_groups) -> (points float32[n_words, n_groups, d] (NaN where no occurrences), counts int64[n_words, n_groups], self_sim float[n_words, n_groups])`
  - `segment_stats(a, v, b) -> (t, d, seg)` (broadcasts over leading dims of `v`)
  - `null_percentile(a, b, v, null_vs) -> float` (0..100)
  - `role_swap(trip: (3, d), nulls: list of 3 arrays (k, d)) -> list[float]` (pct per placement, ROLES order)
  - `triple_verdict(pcts, between_pct) -> {"beats_null": bool, "best_of_three": bool}`
  - `binom_sf(k, n, p) -> float`
  - `set_summary(verdicts: list[dict]) -> {"n", "beats_null", "best_of_three", "p_best_of_three"}`
  - `occurrence_stats(occ_v, a, b) -> (t, d, seg)` arrays, one per occurrence
  - `nearest_in_log_count(target: int, cands: dict[str, int], k: int) -> list[str]`
  - `run_q16(points: list[np.ndarray], counts, w_index: dict[str, int], triples: list[dict], pool: dict[pos, list[str]], found: dict, *, groups, min_count, null_k, between_pct) -> (entries: list[dict], missing: list[dict])`
  - `summarize(entries, groups, n_frames) -> {set: {group: [set_summary per frame]}}`

  Each entry: `{"id", "set", "concept", "pos", "words": [d, m, e], "groups": {g: {"missing": [words]} | {...}}}` where a present group holds per-frame lists:
  `counts` (3 ints), `null_words` (mean role), `t`, `d`, `seg`, `null_pct`, `swap_pct` ([3] per frame),
  `td` ([[t, d]] x 3 placements per frame), `null_td` ([[t, d]] x null_k per frame), `beats_null`, `best_of_three`.

- [ ] **Step 1: Write the failing tests** (`tests/test_betweenness.py`)

```python
"""Q16 maths on hand-built points: t, d, seg, nulls, role swaps, verdicts."""
import numpy as np
import pytest

from token_drift import betweenness as bt

A, B = np.array([0.0, 0, 0]), np.array([2.0, 0, 0])


def test_segment_stats_midpoint_and_beyond():
    t, d, seg = bt.segment_stats(A, np.array([1.0, 0, 0]), B)
    assert (t, d, seg) == pytest.approx((0.5, 0, 0))
    t, d, seg = bt.segment_stats(A, np.array([3.0, 0, 0]), B)  # on the line, past b
    assert t == pytest.approx(1.5) and d == pytest.approx(0) and seg == pytest.approx(0.5)
    t, d, seg = bt.segment_stats(A, np.array([1.0, 1, 0]), B)
    assert (t, d, seg) == pytest.approx((0.5, 0.5, 0.5))


def test_segment_stats_ignores_shared_offset_and_scale():
    rng = np.random.default_rng(0)
    a, v, b = rng.normal(size=(3, 8))
    shift = rng.normal(size=8)
    for x, y in zip(bt.segment_stats(a, v, b), bt.segment_stats(3 * a + shift, 3 * v + shift, 3 * b + shift)):
        assert x == pytest.approx(y)


def test_segment_stats_broadcasts_over_many_v():
    vs = np.array([[1.0, 0, 0], [3.0, 0, 0]])
    t, d, seg = bt.segment_stats(A, vs, B)
    assert t.tolist() == pytest.approx([0.5, 1.5]) and seg.tolist() == pytest.approx([0, 0.5])


def test_unit_mean_points_counts_and_self_sim():
    occ = np.array([[2.0, 0], [0, 3.0], [5.0, 0], [0, 1.0]], dtype=np.float16)
    pts, counts, ss = bt.unit_mean(occ, np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]), n_words=2, n_groups=2)
    assert counts.tolist() == [[2, 0], [1, 1]]
    np.testing.assert_allclose(pts[0, 0], [0.5, 0.5])  # mean of unit vectors, not re-normed
    assert np.isnan(pts[0, 1]).all()
    assert ss[0, 0] == pytest.approx(0.0)  # two orthogonal occurrences
    assert np.isnan(ss[1, 0])  # one occurrence: no pairs


def test_null_percentile():
    nulls = np.array([[1.0, 0.5, 0], [1.0, 1, 0], [5.0, 0, 0], [1.0, 0.05, 0]])
    # real seg 0.1; null segs 0.25, 0.5, 1.5, 0.025 -> one of four is at least as close
    assert bt.null_percentile(A, B, np.array([1.0, 0.2, 0]), nulls) == pytest.approx(25.0)


TRIP = np.array([[0.0, 0, 0], [1.0, 0.05, 0], [2.0, 0, 0]])  # deficiency, mean, excess
NULLS = np.array([[1, 0.5, 0], [1.5, 0.3, 0], [0.5, 0.4, 0], [1, -0.6, 0], [1.2, 0.2, 0.1], [0.3, 0.25, 0]])


def test_role_swap_and_verdict_mean_between():
    pcts = bt.role_swap(TRIP, [NULLS] * 3)
    assert pcts[1] == 0.0 and pcts[0] > 0 and pcts[2] > 0
    assert bt.triple_verdict(pcts, between_pct=5) == {"beats_null": True, "best_of_three": True}


def test_verdict_ties_are_not_best():
    assert bt.triple_verdict([0.0, 0.0, 50.0], 5) == {"beats_null": True, "best_of_three": False}


def test_set_summary_binomial():
    vs = [{"beats_null": True, "best_of_three": True}] * 5 + [{"beats_null": False, "best_of_three": False}]
    s = bt.set_summary(vs)
    assert (s["n"], s["beats_null"], s["best_of_three"]) == (6, 5, 5)
    assert s["p_best_of_three"] == pytest.approx(13 / 729)  # P(X >= 5), X ~ Bin(6, 1/3)
    assert bt.set_summary([])["p_best_of_three"] is None


def test_occurrence_stats_unit_norms_each_occurrence():
    a, b = np.array([1.0, 0]), np.array([0.0, 1])
    t, d, seg = bt.occurrence_stats(np.array([[10.0, 10.0], [3.0, 0.0]]), a, b)
    assert t.tolist() == pytest.approx([0.5, 0.0]) and seg[1] == pytest.approx(0.0)


def test_nearest_in_log_count():
    assert bt.nearest_in_log_count(50, {"a": 45, "b": 55, "c": 5000, "d": 60, "e": 40}, 3) == ["b", "a", "d"]


# ---------- run_q16 ----------

WORDS = ["cowardice", "courage", "rashness"] + [f"n{i}" for i in range(6)] + ["timidity"]


def _setup(found=None):
    w_index = {w: i for i, w in enumerate(WORDS)}
    P = np.zeros((len(WORDS), 1, 3), dtype=np.float32)
    P[:3, 0] = TRIP
    P[3:9, 0] = NULLS
    counts = np.full((len(WORDS), 1), 30)
    counts[w_index["timidity"], 0] = 2
    found = found or {w: {"books": 100} for w in WORDS}
    triples = [
        {"id": "c/fear/noun/a", "set": "c", "concept": "fear", "pos": "noun", "words": ("cowardice", "courage", "rashness")},
        {"id": "c/fear/noun/b", "set": "c", "concept": "fear", "pos": "noun", "words": ("courage", "cowardice", "rashness")},
        {"id": "c/fear/noun/c", "set": "c", "concept": "fear", "pos": "noun", "words": ("timidity", "courage", "rashness")},
    ]
    pool = {"noun": [f"n{i}" for i in range(6)], "adj": []}
    return [P, P], counts, w_index, triples, pool, found


def test_run_q16_verdicts_missing_and_summary():
    points, counts, w_index, triples, pool, found = _setup()
    entries, missing = bt.run_q16(points, counts, w_index, triples, pool, found, groups=("books",),
                                  min_count=20, null_k=6, between_pct=5)
    good, swapped, short = (e["groups"]["books"] for e in entries)
    assert good["beats_null"] == [True, True] and good["best_of_three"] == [True, True]
    assert good["t"][0] == pytest.approx(0.5) and len(good["null_td"][0]) == 6 and len(good["td"][0]) == 3
    assert swapped["beats_null"] == [False, False] and swapped["best_of_three"] == [False, False]
    assert short == {"missing": ["timidity"]}
    assert missing == [{"id": "c/fear/noun/c", "group": "books", "short": ["timidity"]}]
    s = bt.summarize(entries, ("books",), n_frames=2)
    assert s["c"]["books"][0]["n"] == 2 and s["c"]["books"][0]["best_of_three"] == 1


def test_run_q16_null_neighbours_use_found_counts_not_capped_counts():
    # Review Focus 4: kept counts are all capped (30 here), found counts are not
    found = {w: {"books": 50} for w in WORDS}
    found.update({"n0": {"books": 45}, "n1": {"books": 55}, "n2": {"books": 5000},
                  "n3": {"books": 9000}, "n4": {"books": 60}, "n5": {"books": 40}})
    points, counts, w_index, triples, pool, _ = _setup()
    entries, _ = bt.run_q16(points, counts, w_index, triples[:1], pool, found, groups=("books",),
                            min_count=20, null_k=3, between_pct=5)
    assert set(entries[0]["groups"]["books"]["null_words"]) == {"n0", "n1", "n4"}


def test_run_q16_too_few_null_words_raises():
    points, counts, w_index, triples, pool, found = _setup()
    with pytest.raises(ValueError, match="null pool"):
        bt.run_q16(points, counts, w_index, triples[:1], pool, found, groups=("books",),
                   min_count=20, null_k=7, between_pct=5)
```

Note on `found` in `_setup`: real `counts.json` holds `{word: {group: {"found": n, ...}}}`; `run_q16`
takes `found[word][group]` as a plain int, and the cli flattens it (Task 7). Keep that split.

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_betweenness.py -q`
Expected: ERROR, `No module named token_drift.betweenness`.

- [ ] **Step 3: Implement** (`src/token_drift/betweenness.py`)

```python
"""Q16: does a virtue's point sit between its two vices' points? Pure numpy, one frame at a time.

The trap (spec, "Why"): in 512 dims three random words form a near-equilateral triangle, so a
random "virtue" lands at t = 0.5, d = 0.87 at every frame. A raw t means nothing; every number
here is judged against frequency-matched null words and against the role-swapped triple.
"""
from __future__ import annotations

from math import comb

import numpy as np

from token_drift.extract import self_similarity

ENDS = {0: (1, 2), 1: (0, 2), 2: (0, 1)}  # placement k: word k in the middle, the other two are the ends


def unit_mean(occ, word_idx, group_idx, n_words: int, n_groups: int):
    """One frame of occurrences -> (points, counts, self_sim), each indexed [word, group].

    Points average unit-normed occurrences (A's unit_mean): a few huge-norm occurrences
    (L6pre, FINDINGS 11.3) can't drag the point. No occurrences -> NaN point, on purpose.
    """
    occ = np.asarray(occ, dtype=np.float32)
    u = occ / np.maximum(np.linalg.norm(occ, axis=1, keepdims=True), 1e-8)
    key = np.asarray(word_idx) * n_groups + np.asarray(group_idx)
    K = n_words * n_groups
    sums = np.zeros((K, occ.shape[1]), dtype=np.float64)
    np.add.at(sums, key, u)
    counts = np.bincount(key, minlength=K)
    with np.errstate(invalid="ignore", divide="ignore"):
        points = (sums / counts[:, None]).astype(np.float32)
    ss = self_similarity((sums**2).sum(1), counts)
    shape = (n_words, n_groups)
    return points.reshape(*shape, -1), counts.reshape(shape), ss.reshape(shape)


def segment_stats(a, v, b):
    """t = where v projects on the a->b line (0 = a, 1 = b), d = distance off the line,
    seg = distance to the closest point of the *segment* [a, b] (= d when 0 <= t <= 1),
    all in units of |b - a|. Broadcasts over leading dims of v."""
    a, b, v = (np.asarray(x, dtype=np.float64) for x in (a, b, v))
    ab = b - a
    L = np.sqrt(ab @ ab)
    t = np.asarray(((v - a) @ ab) / (L * L))
    d = np.linalg.norm(v - (a + t[..., None] * ab), axis=-1) / L
    tc = np.clip(t, 0.0, 1.0)
    seg = np.linalg.norm(v - (a + tc[..., None] * ab), axis=-1) / L
    return t, d, seg


def null_percentile(a, b, v, null_vs) -> float:
    """% of null words at least as close to segment [a, b] as v (0 = closer than all of them)."""
    real = segment_stats(a, v, b)[2]
    nulls = segment_stats(a, np.asarray(null_vs), b)[2]
    return float(100.0 * np.mean(nulls <= real))


def role_swap(trip, nulls) -> list[float]:
    """Null percentile with each word of the triple in the middle (ROLES order), each against
    its own null words. Aristotle's claim = the mean (index 1) beats both swaps."""
    return [null_percentile(trip[ENDS[k][0]], trip[ENDS[k][1]], trip[k], nulls[k]) for k in range(3)]


def triple_verdict(pcts, between_pct: float) -> dict:
    return {"beats_null": bool(pcts[1] < between_pct),
            "best_of_three": bool(pcts[1] < pcts[0] and pcts[1] < pcts[2])}  # ties: not best


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return float(sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def set_summary(verdicts: list[dict]) -> dict:
    """How many triples beat the null / have the virtue best of three, and how surprising the
    best-of-three count is if the middle word were picked at random (chance = 1/3)."""
    n = len(verdicts)
    best = sum(v["best_of_three"] for v in verdicts)
    return {"n": n, "beats_null": sum(v["beats_null"] for v in verdicts), "best_of_three": best,
            "p_best_of_three": binom_sf(best, n, 1 / 3) if n else None}


def occurrence_stats(occ_v, a, b):
    """(t, d, seg) of each unit-normed occurrence of the middle word vs the two end points."""
    occ_v = np.asarray(occ_v, dtype=np.float64)
    u = occ_v / np.maximum(np.linalg.norm(occ_v, axis=1, keepdims=True), 1e-8)
    return segment_stats(a, u, b)


def nearest_in_log_count(target: int, cands: dict[str, int], k: int) -> list[str]:
    """The k candidates closest in log count (frequency moves norms and neighbourhoods, Q7)."""
    words = sorted(cands)
    lc = np.log(np.maximum([cands[w] for w in words], 1))
    order = np.argsort(np.abs(lc - np.log(max(target, 1))), kind="stable")
    return [words[i] for i in order[:k]]


def _f(x) -> float:
    return float(x)


def run_q16(points, counts, w_index, triples, pool, found, *, groups, min_count, null_k, between_pct):
    """Score every triple in every group at every frame -> (entries, missing).

    points: one (n_words, n_groups, d) array per frame. counts: occurrences behind each point.
    found[word][group]: uncapped hit count, for picking null words (kept counts all sit at the cap).
    """
    entries, missing = [], []
    for tr in triples:
        entry = {k: tr[k] for k in ("id", "set", "concept", "pos")} | {"words": list(tr["words"]), "groups": {}}
        for g_i, g in enumerate(groups):
            short = [w for w in tr["words"] if w not in w_index or counts[w_index[w], g_i] < min_count]
            if short:
                entry["groups"][g] = {"missing": short}
                missing.append({"id": tr["id"], "group": g, "short": short})
                continue
            usable = {w: found[w][g] for w in pool.get(tr["pos"], [])
                      if w in w_index and counts[w_index[w], g_i] >= min_count}
            if len(usable) < null_k:
                c = sorted(usable.values())
                raise ValueError(f"{tr['pos']} null pool has {len(usable)} usable words in {g} "
                                 f"(found counts {c[:1]}..{c[-1:]}), need null_k={null_k}; widen null_pool")
            nulls = [nearest_in_log_count(found[w][g], usable, null_k) for w in tr["words"]]
            rows = [w_index[w] for w in tr["words"]]
            nrows = [[w_index[w] for w in ns] for ns in nulls]
            e = {"counts": [int(counts[r, g_i]) for r in rows], "null_words": nulls[1],
                 **{k: [] for k in ("t", "d", "seg", "null_pct", "swap_pct", "td", "null_td",
                                    "beats_null", "best_of_three")}}
            for P in points:
                trip = P[rows, g_i].astype(np.float64)
                null_pts = [P[nr, g_i].astype(np.float64) for nr in nrows]
                pcts = role_swap(trip, null_pts)
                t, d, seg = segment_stats(trip[0], trip[1], trip[2])
                nt, nd, _ = segment_stats(trip[0], null_pts[1], trip[2])
                td = [segment_stats(trip[ENDS[k][0]], trip[k], trip[ENDS[k][1]])[:2] for k in range(3)]
                v = triple_verdict(pcts, between_pct)
                e["t"].append(_f(t))
                e["d"].append(_f(d))
                e["seg"].append(_f(seg))
                e["null_pct"].append(pcts[1])
                e["swap_pct"].append(pcts)
                e["td"].append([[_f(x), _f(y)] for x, y in td])
                e["null_td"].append([[_f(x), _f(y)] for x, y in zip(nt, nd)])
                e["beats_null"].append(v["beats_null"])
                e["best_of_three"].append(v["best_of_three"])
            entry["groups"][g] = e
        entries.append(entry)
    return entries, missing


def summarize(entries, groups, n_frames: int) -> dict:
    """{set: {group: [set_summary per frame]}} over the triples present in that group."""
    out: dict = {}
    for s in dict.fromkeys(e["set"] for e in entries):
        out[s] = {}
        for g in groups:
            present = [e["groups"][g] for e in entries if e["set"] == s and "missing" not in e["groups"][g]]
            out[s][g] = [set_summary([{"beats_null": p["beats_null"][f], "best_of_three": p["best_of_three"][f]}
                                      for p in present]) for f in range(n_frames)]
    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_betweenness.py -q`
Expected: PASS. If `test_role_swap_and_verdict_mean_between` fails on `pcts[0] > 0`, print the
three `role_swap` percentiles and each null word's seg before changing anything: the fixture was
built so the swapped placements have seg ~1.0 and most null words are closer.

- [ ] **Step 5: Commit**

```bash
git add src/token_drift/betweenness.py tests/test_betweenness.py
git commit -m "betweenness: t/d/seg, null percentile, role swaps, verdicts, run_q16

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: probe metrics -> `q16.json`

**Files:**
- Modify: `src/token_drift/cli.py` (`_Probe`, `_load_probe`, `_stage_metrics_probe`, dispatch in `stage_metrics`)
- Test: `tests/test_probe_stages.py` (append)

**Interfaces:**
- Consumes: `probe_corpus/{probe_words.yaml, occ_meta.parquet, counts.json, null_pool.json}`, `extract/{occ.npy, layer_names.json}`; `bt.unit_mean`, `bt.run_q16`, `bt.summarize`.
- Produces:
  - `cli._load_probe(rd, columns=("word", "group")) -> _Probe` with fields `meta: dict[str, list]`, `vocab: list[str]` (sorted), `w_index: dict[str, int]`, `widx: int64[n_occ]`, `gidx: int64[n_occ]`, `occ` (memmap `(F, n_occ, d)`); raises if `occ.npy` and `occ_meta.parquet` disagree on the occurrence count
  - `runs/<run>/metrics/q16.json`:
    `{"layer_names": [F], "groups": ["books", "pile"], "min_count", "null_k", "between_pct", "triples": [entries from run_q16], "summary": summarize(...), "missing": [...], "self_sim": {word: {group: [F floats or null]}}}`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_probe_stages.py`)

```python
@pytest.fixture
def extracted(probe_cfg):
    c = cli.load_config(probe_cfg)
    cli.stage_probe_corpus(c)
    cli.stage_extract(c)
    return c, cli.run_dir(c)


def test_probe_metrics_writes_q16(extracted):
    c, rd = extracted
    cli.stage_metrics(c)
    q = json.loads((rd / "metrics" / "q16.json").read_text())
    assert q["groups"] == ["books", "pile"] and len(q["layer_names"]) == 4
    assert [t["id"] for t in q["triples"]] == [
        "classical/fear/noun/cowardice,courage,rashness", "classical/fear/adj/cowardly,brave,rash",
        "everyday/fear/noun/timidity,courage,rashness"]
    present = q["triples"][0]["groups"]["books"]
    assert len(present["null_pct"]) == 4 and len(present["swap_pct"][0]) == 3 and len(present["null_words"]) == 2
    assert q["triples"][2]["groups"]["pile"] == {"missing": ["timidity"]}
    assert q["summary"]["classical"]["books"][0]["n"] == 2 and q["summary"]["everyday"]["pile"][0]["n"] == 0
    assert {(m["id"].split("/")[0], m["group"]) for m in q["missing"]} == {("everyday", "books"), ("everyday", "pile")}
    assert q["self_sim"]["courage"]["books"][0] == pytest.approx(1.0, abs=1e-3)  # frame 0: embedding, same every time


def test_metrics_uses_the_word_list_saved_with_the_occurrences(extracted):
    # Review Focus 5: Andrey edits the yaml after extraction and reruns only metrics
    c, rd = extracted
    with open(c["probe"]["words"], "w") as f:
        f.write("sets: {}\n")
    cli.stage_metrics(c)
    assert len(json.loads((rd / "metrics" / "q16.json").read_text())["triples"]) == 3


def test_metrics_refuses_a_stale_extract(extracted):
    c, rd = extracted
    occ = np.load(rd / "extract" / "occ.npy")
    np.save(rd / "extract" / "occ.npy", occ[:, :-1])
    with pytest.raises(ValueError, match="re-run extract"):
        cli.stage_metrics(c)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_probe_stages.py -q -k "metrics or q16"`
Expected: FAIL (`stage_metrics` looks for `normalize/acts_norm.npy`).

- [ ] **Step 3: Implement** (in `src/token_drift/cli.py`)

Add `from dataclasses import dataclass` and `from token_drift import betweenness as bt` to the imports. Then, before `stage_metrics`:

```python
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
    return _Probe(meta, vocab, w_index, widx, gidx, occ)


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
    ss = np.stack(self_sim)  # (frames, words, groups)
    result = {
        "layer_names": names, "groups": list(GROUPS), "min_count": m["min_count"], "null_k": m["null_k"],
        "between_pct": m["between_pct"], "triples": entries,
        "summary": bt.summarize(entries, GROUPS, len(names)), "missing": missing,
        "self_sim": {w: {g: [None if np.isnan(x) else float(x) for x in ss[:, i, g_i]]
                         for g_i, g in enumerate(GROUPS)} for w, i in data.w_index.items()},
    }
    out = stage_dir(rd, "metrics")
    (out / "q16.json").write_text(json.dumps(result, indent=1))
    for s, by_g in result["summary"].items():
        for g, per_frame in by_g.items():
            best = [x["best_of_three"] for x in per_frame]
            typer.echo(f"[metrics] {s}/{g}: n={per_frame[0]['n']} best_of_three per frame={best}")
    typer.echo(f"[metrics] {len(missing)} triple x group missing ({time.time() - t0:.0f}s) -> {out / 'q16.json'}")
    return rd
```

At the top of `stage_metrics`, before `rd = _prepare_run_dir(cfg)`:

```python
    if _mode(cfg) == "probe":
        return _stage_metrics_probe(cfg)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/token_drift/cli.py tests/test_probe_stages.py
git commit -m "metrics: probe mode -> q16.json (points per word x group x frame, nulls, role swaps)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: the three Q16 figures + `all` end to end

**Files:**
- Modify: `src/token_drift/viz.py` (three plot functions)
- Modify: `src/token_drift/cli.py` (`_stage_viz_probe`, dispatch in `stage_viz`)
- Test: `tests/test_viz.py` (append), `tests/test_probe_stages.py` (append)

**Interfaces:**
- Consumes: `metrics/q16.json` (Task 7), `_load_probe`, `bt.unit_mean`, `bt.occurrence_stats`.
- Produces:
  - `viz.plot_q16_summary(q16: dict, out_path, *, return_fig=False) -> Path | Figure`
  - `viz.plot_q16_triples(entries: list[dict], layer_names: list[str], out_path, *, title: str, return_fig=False) -> Path | Figure | None` (None when every row is missing)
  - `viz.plot_q16_occ(rows: list[tuple[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]]], frame_names: list[str], out_path, *, title: str, return_fig=False) -> Path | Figure`
  - `runs/<run>/viz/q16_summary.png`, `q16_<set>_<concept>_<pos>.png`, `q16_occ_<set>_<concept>_<pos>.png`

Colour choices (dataviz skill): the heatmap is a magnitude on one hue, so it uses the reference
sequential blue ramp, dark = low percentile = "between" (the interesting end); missing cells are
hatched grey, not a colour. The (t, d) panels use the existing `RUN_COLORS` in fixed order for the
three placements plus grey for null words, keyed by a figure legend; identity is also carried by
fill (virtue filled, swaps hollow), so it isn't colour alone.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_viz.py` (and add `from token_drift import viz` to its imports; the file currently imports functions by name):

```python
def _fake_q16(n_frames=3):
    rng = np.random.default_rng(0)
    present = lambda: {  # noqa: E731
        "counts": [30, 40, 50], "null_words": ["a", "b"],
        "t": [0.5] * n_frames, "d": [0.3] * n_frames, "seg": [0.3] * n_frames,
        "null_pct": [0.0, 40.0, 90.0][:n_frames], "swap_pct": [[50.0, 0.0, 60.0]] * n_frames,
        "td": [[[1.2, 0.8], [0.5, 0.3], [-0.2, 0.9]]] * n_frames,
        "null_td": [rng.uniform(0, 1, size=(2, 2)).tolist() for _ in range(n_frames)],
        "beats_null": [True, False, False][:n_frames], "best_of_three": [True, True, False][:n_frames],
    }
    triples = [
        {"id": "classical/fear/noun/x,y,z", "set": "classical", "concept": "fear", "pos": "noun",
         "words": ["x", "y", "z"], "groups": {"books": present(), "pile": {"missing": ["z"]}}},
        {"id": "everyday/fear/noun/p,q,r", "set": "everyday", "concept": "fear", "pos": "noun",
         "words": ["p", "q", "r"], "groups": {"books": {"missing": ["p"]}, "pile": {"missing": ["p"]}}},
    ]
    return {"layer_names": ["L0 (embed)", "L1 (pre-LN)", "L1 (post-LN)"][:n_frames],
            "groups": ["books", "pile"], "triples": triples}


def test_plot_q16_summary_writes_png(tmp_path):
    p = viz.plot_q16_summary(_fake_q16(), tmp_path / "s.png")
    assert p.exists()


def test_plot_q16_triples_shares_axes_and_skips_all_missing(tmp_path):
    q = _fake_q16()
    fig = viz.plot_q16_triples(q["triples"][:1], q["layer_names"], tmp_path / "t.png", title="x", return_fig=True)
    axes = fig.axes[:3]  # one present row x 3 frames
    assert len({ax.get_xlim() for ax in axes}) == 1 and len({ax.get_ylim() for ax in axes}) == 1
    assert viz.plot_q16_triples(q["triples"][1:], q["layer_names"], tmp_path / "u.png", title="x") is None


def test_plot_q16_occ_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    rows = [("x / y / z", {"books": [(rng.normal(0.5, 0.2, 40), rng.uniform(0, 1, 40))] * 2,
                           "pile": [(rng.normal(0.4, 0.3, 60), rng.uniform(0, 1, 60))] * 2})]
    assert viz.plot_q16_occ(rows, ["L0 (embed)", "L1"], tmp_path / "o.png", title="x").exists()
```

Append to `tests/test_probe_stages.py`:

```python
def test_all_runs_the_whole_probe_pipeline(probe_cfg):
    cli.all(probe_cfg)
    rd = cli.run_dir(cli.load_config(probe_cfg))
    assert sorted(x.name for x in rd.iterdir()) == ["config.yaml", "extract", "metrics", "probe_corpus", "viz"]
    v = rd / "viz"
    for f in ("q16_summary.png", "q16_classical_fear_noun.png", "q16_classical_fear_adj.png",
              "q16_occ_classical_fear_noun.png"):
        assert (v / f).exists(), f
    assert not (v / "q16_everyday_fear_noun.png").exists()  # every row missing: nothing to draw


def test_viz_rejects_occ_frames_out_of_range(extracted):
    c, rd = extracted
    cli.stage_metrics(c)
    c["viz"]["occ_frames"] = [0, 9]
    with pytest.raises(ValueError, match="occ_frames"):
        cli.stage_viz(c)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_viz.py tests/test_probe_stages.py -q`
Expected: FAIL (`plot_q16_*` missing; `stage_viz` reads `normalize/`).

- [ ] **Step 3: Implement the plots** (append to `src/token_drift/viz.py`; add `from matplotlib.colors import LinearSegmentedColormap` and `from matplotlib.patches import Rectangle` to the imports, with `# noqa: E402` like the others)

```python
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
    null percentile. Dot = virtue beats both role swaps; hatched = a word below min_count."""
    rows, names, groups = q16["triples"], q16["layer_names"], q16["groups"]
    cmap = LinearSegmentedColormap.from_list("q16", _SEQ_BLUE)
    fig, axes = plt.subplots(1, len(groups), squeeze=False, sharey=True, layout="constrained",
                             figsize=(3.8 + 0.45 * len(names) * len(groups), 1.6 + 0.26 * len(rows)))
    im = None
    for ax, g in zip(axes[0], groups):
        M = np.full((len(rows), len(names)), np.nan)
        for i, t in enumerate(rows):
            if "missing" not in t["groups"][g]:
                M[i] = t["groups"][g]["null_pct"]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
        for i, t in enumerate(rows):
            e = t["groups"][g]
            if "missing" in e:
                ax.add_patch(Rectangle((-0.5, i - 0.5), len(names), 1, facecolor="none",
                                       edgecolor=_MUTED, hatch="////", linewidth=0))
                continue
            best = np.flatnonzero(e["best_of_three"])
            ax.scatter(best, np.full(len(best), i), s=14, color="white", edgecolors=_INK, linewidths=0.6, zorder=3)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.tick_params(colors=_MUTED, labelsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(g, loc="left", fontsize=10, color=_INK)
    axes[0][0].set_yticks(range(len(rows)), [_triple_label(t) for t in rows])
    cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.6)
    cb.set_label("virtue's null percentile (0 = closer than every null word)", fontsize=8, color=_INK)
    fig.suptitle("Q16: does the virtue sit between its vices?  dot = virtue beats both role swaps, "
                 "hatched = a word below min_count", x=0.01, ha="left", fontsize=9, color=_INK)
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
            nt = np.asarray(e["null_td"][f])
            ax.scatter(nt[:, 0], nt[:, 1], s=8, color=_MUTED, alpha=0.5, linewidths=0)
            for (_, col, filled), (tt, dd) in zip(_PLACEMENTS, e["td"][f]):
                ax.scatter([tt], [dd], s=36, facecolors=col if filled else "none", edgecolors=col,
                           linewidths=1.4, zorder=3)
            if r == 0:
                ax.set_title(layer_names[f], fontsize=8, color=_INK)
        axes[r, 0].set_ylabel(f"{' / '.join(t['words'])}\n{g}\nd", fontsize=7, color=_INK)
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
                bins = np.linspace(vals.min(), vals.max() + 1e-9, 30)
                for g in per_group:
                    ax.hist(per_group[g][f][j], bins=bins, density=True, histtype="step",
                            color=_GROUP_COLORS[g], linewidth=1.4)
                if stat == "t":
                    for x in (0, 1):
                        ax.axvline(x, color=_MUTED, linewidth=0.8, linestyle="--")
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"{frame_names[f]}: {stat}", fontsize=8, color=_INK)
        axes[r, 0].set_ylabel(label, fontsize=7, color=_INK)
    fig.suptitle(f"{title}: per-occurrence position of the virtue", x=0.01, ha="left", fontsize=10, color=_INK)
    fig.tight_layout()
    groups = [g for g in _GROUP_COLORS if any(g in pg for _, pg in rows)]
    leg = fig.legend([Line2D([], [], color=_GROUP_COLORS[g], linewidth=1.4) for g in groups], groups,
                     loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    _style_legend(leg)
    return _save(fig, out_path, return_fig)
```

- [ ] **Step 4: Wire the viz stage** (in `src/token_drift/cli.py`, before `stage_viz`)

```python
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
```

At the top of `stage_viz`, before `rd = _prepare_run_dir(cfg)`:

```python
    if _mode(cfg) == "probe":
        return _stage_viz_probe(cfg)
```

- [ ] **Step 5: Run the tests, then look at the pictures**

Run: `uv run pytest -q`
Expected: PASS.

Then render the tiny run once and open the PNGs (Read tool on the image files) to eyeball label
collisions and overflow; the tests can't see those:

```bash
uv run pytest tests/test_probe_stages.py::test_all_runs_the_whole_probe_pipeline -q --basetemp=/tmp/claude-1000/q16viz
ls /tmp/claude-1000/q16viz/*/runs/tiny_probe/viz/
```

Fix anything unreadable (overlapping y labels, legend over panels) before committing.

- [ ] **Step 6: Commit**

```bash
git add src/token_drift/viz.py src/token_drift/cli.py tests/test_viz.py tests/test_probe_stages.py
git commit -m "viz: Q16 summary heatmap, (t, d) plane per triple, per-occurrence histograms

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: `token-drift occ` (read the senses)

**Files:**
- Modify: `src/token_drift/cli.py` (`occ_report`, `occ` command)
- Test: `tests/test_probe_stages.py` (append)

**Interfaces:**
- Consumes: `_load_probe(rd, columns=("word", "group", "source", "snippet"))`, `bt.unit_mean`, `bt.occurrence_stats`.
- Produces: `cli.occ_report(cfg, words: list[str], frame: int, group: str, n: int) -> str`; CLI
  `token-drift occ -c CFG --triple cowardice,courage,rashness [--frame 3] [--group books] [-n 5]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_probe_stages.py`)

```python
def test_occ_report_lists_most_and_least_between_with_snippets(extracted):
    c, _ = extracted
    txt = cli.occ_report(c, ["cowardice", "courage", "rashness"], frame=2, group="pile", n=2)
    assert "most between" in txt and "least between" in txt
    assert txt.count("seg=") == 4 and "soldier" in txt and "[pile:0]" in txt


def test_occ_report_explains_bad_input(extracted):
    c, _ = extracted
    with pytest.raises(ValueError, match="timidity"):
        cli.occ_report(c, ["timidity", "courage", "rashness"], frame=0, group="books", n=2)
    with pytest.raises(ValueError, match="group"):
        cli.occ_report(c, ["cowardice", "courage", "rashness"], frame=0, group="web", n=2)
    with pytest.raises(ValueError, match="three"):
        cli.occ_report(c, ["courage"], frame=0, group="books", n=2)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/test_probe_stages.py -q -k occ_report`
Expected: FAIL (`occ_report` missing).

- [ ] **Step 3: Implement** (in `src/token_drift/cli.py`, after `_stage_viz_probe`)

```python
def occ_report(cfg: dict, words: list[str], frame: int, group: str, n: int) -> str:
    """The n most and least 'between' occurrences of the middle word, with their text.
    Reading the senses is half the point: is the between-ness the ethical sense or not?"""
    if len(words) != 3:
        raise ValueError(f"--triple needs three words (deficiency,mean,excess), got {words}")
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}")
    rd = run_dir(cfg)
    data = _load_probe(rd, columns=("word", "group", "source", "snippet"))
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


@app.command("occ")
def occ_cmd(
    config: Path = _CONFIG,
    triple: str = typer.Option(..., "--triple", help="deficiency,mean,excess e.g. cowardice,courage,rashness"),
    frame: int = typer.Option(3, "--frame", help="frame index (0 = embed)"),
    group: str = typer.Option("books", "--group", help="books | pile"),
    n: int = typer.Option(5, "-n", help="how many occurrences at each end"),
):
    """Print the most and least 'between' occurrences of a triple's virtue, with snippets."""
    typer.echo(occ_report(load_config(config), triple.split(","), frame, group, n))
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/token_drift/cli.py tests/test_probe_stages.py
git commit -m "cli: occ command, most/least between occurrences with snippets

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Real runs (GPU, with Andrey at the count report)

Not a code task: run, look, stop at each checkpoint. Do it in the main session, not a subagent:
Andrey reviews the count report and null pool before the GPU pass.

- [ ] **Step 1: Preflight**

```bash
nvidia-smi                      # GPU visible? (kernel/driver mismatch bit us on 09-25)
df -h .                         # need ~6 GB: 0.8 GB Pile download, ~1 GB windows, ~2 GB occ.npy x2 runs
uv run pytest -q
```

- [ ] **Step 2: Probe corpus for the trained run**

```bash
uv run token-drift probe_corpus -c configs/pythia70m_probe.yaml
```

Expected: downloads 10 books into `~/.cache/token-drift` and 3 Pile shards (~0.78 GB) into the HF
cache; a few minutes of tokenizing per shard. If a book marker raises, check the line in the
cached `pg<id>.txt` and fix the config marker, not the code.

- [ ] **Step 3: Checkpoint with Andrey.** Show `runs/pythia70m_probe/probe_corpus/count_report.md`
  (which triples are usable where, compared with the spec's regex counts), the null pool
  (junk -> `null_exclude`), `meta.json` (window count; compare with D8's ~200-250k). Edit
  `configs/probe_words.yaml` if Andrey wants, rerun Step 2. Don't run extract until Andrey says go.

- [ ] **Step 4: Extract, metrics, viz (trained)**

```bash
uv run token-drift extract -c configs/pythia70m_probe.yaml
uv run token-drift metrics -c configs/pythia70m_probe.yaml
uv run token-drift viz -c configs/pythia70m_probe.yaml
```

Expected: extract ~15-20 min on GPU. Open `viz/q16_summary.png` and a couple of triple figures.

- [ ] **Step 5: Random-init control.** The windows are identical (same seed, same texts), so copy
  instead of re-tokenizing 360M tokens:

```bash
mkdir -p runs/random_init_probe && cp -r runs/pythia70m_probe/probe_corpus runs/random_init_probe/
uv run token-drift extract -c configs/random_init_probe.yaml
uv run token-drift metrics -c configs/random_init_probe.yaml
uv run token-drift viz -c configs/random_init_probe.yaml
```

- [ ] **Step 6: Sanity checks before anyone reads results**
  - frame 0 self-sim of every word = 1.0 in `q16.json` (the embedding doesn't depend on context);
  - `uv run token-drift occ -c configs/pythia70m_probe.yaml --triple cowardice,courage,rashness --frame 3 --group books`
    shows Aristotle/Aquinas text, `--group pile` shows Pile text;
  - random-init summary near chance (best_of_three ~1/3 of n). If it isn't, say so: that's a finding, not a bug to smooth over.

- [ ] **Step 7: Commit configs only if the word file changed** (runs/ is gitignored)

```bash
git add configs/probe_words.yaml
git commit -m "probe words: edits after the first count report

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Write-up (together with Andrey)

Spec step 10. Same way as A's Task 9: drafts, Andrey's verdicts, then commit.

- [ ] **Step 1:** `docs/FINDINGS.md` section 12: per triple set (classical / everyday) x group
  (books / pile), how many triples beat the null and how many have the virtue best of three, at
  which frames, random-init alongside; per-occurrence pictures for the triples that work;
  surprises called out, not smoothed over.
- [ ] **Step 2:** `docs/OPEN_QUESTIONS.md`: Q16 verdict; Q9 gets the raw per-word self-sim numbers
  as a note; Q13 hint (occ.npy keeps raw residuals for the L6pre norm check).
- [ ] **Step 3:** `README.md`: a v1 milestone B paragraph with the summary heatmap.
- [ ] **Step 4:** `CLAUDE.md`: probe mode in the pipeline section (`probe_corpus` stage, no
  normalize, `occ.npy`, `q16.json`, the `occ` command), `probe.py` / `betweenness.py` in the repo layout.
- [ ] **Step 5:** `uv run pytest -q`, then commit:

```bash
git add docs/FINDINGS.md docs/OPEN_QUESTIONS.md README.md CLAUDE.md
git commit -m "docs: v1 milestone B findings (Q16), README, CLAUDE.md probe mode

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
