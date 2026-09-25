# v1 milestone A: corpus-averaged vocab (design)

Status: design approved in chat 2026-09-24, spec awaiting review.
Scope: milestone A only. Milestone B (probe corpus, per-occurrence polysemy for
virtue/vice words) gets its own spec later and reuses `corpus.py` from this one.

## Why

v0 manufactured a per-layer vocab matrix by feeding `[BOS, tok]` alone, so every layer
was a function of the token and nothing else. That left four questions untestable
(OPEN_QUESTIONS Q6, Q7, Q9, and Q3 candidate (b)). v1 replaces "the token alone" with
"the token as the model usually sees it": run real text, and per layer average each
token's residual over all of its occurrences.

Milestone A redoes the v0 metrics and flipbook on those averages and puts v0 and v1
side by side on the same tokens.

## Decisions (and why)

| decision | choice | why |
|---|---|---|
| overall v1 shape | A first (vocab-wide averages), then B (polysemy) and C (self-sim) | A is directly comparable to everything in FINDINGS; C turned out to be free inside A |
| corpora | two-corpus split: background = Pile slice (A); probe = targeted texts (B) | in the Pile "virtue" is 57% "by virtue of" and "vice" is 93% "vice president / versa"; the ethical senses need their own texts |
| background corpus | `NeelNanda/pile-10k` (first 10k Pile docs, 15.4M Pythia tokens) | 40k of 50k vocab seen >= 20 times; 33 MB; scaling to `EleutherAI/the_pile_deduplicated` shards is a config change (`source`, `max_tokens`) |
| input format | pack docs joined by `<\|endoftext\|>` into 2048-token windows | how Pythia was trained, so in-distribution |
| positions counted | position >= 32 and token != `<\|endoftext\|>` | position 0 and every mid-window EOS are attention sinks (L3 norm ~120 vs ~12); norms settle by ~16; 32 guarantees real context; costs 1.6% of positions |
| averaging | unit-norm each occurrence, then mean (default); plain mean saved too | an occasional ordinary token also sinks (norm 169 at pos 8 in one window) and would dominate a plain mean; the plain mean stays as a robustness variant, same pass |
| min count | 20 (configurable) | an average over 3 contexts mostly reflects which 3 sentences; drops ~10k mostly-junk rare tokens (and fixes Q12 for v1 as a side effect) |
| controls | `random_init_corpus` (same corpus, random weights) and `pythia70m_corpus_shuf` (tokens shuffled within each window) | within-window shuffle keeps topic, kills word order: "is it reading, or just co-occurrence?" |
| GPT-2 | deferred | not trained on the Pile |
| code shape | new `corpus` stage before `extract` (approach 1 of 3) | shuffle is inspectable on disk; testable with fake docs; B reuses it |

Probe results that informed these (2026-09-24, scratch scripts, not kept):
pile-10k source mix is Pile-CC 25%, OpenWebText2 15%, PubMed Abstracts 14%,
StackExchange 14%, GitHub 9%, Wikipedia 8%, rest < 5% each. Coverage at >= 20
occurrences: 7.8k types at 1.5M tokens, 18.8k at 4.6M, 40.1k at 15.4M.

## Config

New `configs/pythia70m_corpus.yaml`; `pythia70m_corpus_shuf.yaml` differs by
`corpus.shuffle: window`, `random_init_corpus.yaml` by `random_init: true`.

```yaml
run_name: pythia70m_corpus
model: EleutherAI/pythia-70m
random_init: false
seed: 0
device: auto

corpus:
  source: NeelNanda/pile-10k   # any HF parquet dataset with a text column
  text_field: text
  max_tokens: null             # null = all; token budget for bigger sources
  window: 2048                 # Pythia's training context
  shuffle: none                # none | window (seeded permutation inside each window)

extract:
  mode: corpus                 # vocab (v0; default when the key is missing) | corpus
  batch_size: 8                # windows per forward pass
  min_context: 32
  dtype_on_disk: float16

normalize:
  source: unit_mean            # unit_mean | raw_mean
  center: true
  unit_norm: true
  drop_top_pcs: 0

metrics:
  min_count: 20
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

Existing v0 configs keep working untouched: no `corpus:` block, `extract.mode`
defaults to `vocab`, `normalize.source` is ignored, no `min_count` means all tokens are
eligible, and a flat `trajectory_tokens` list is still accepted.

## Data flow

```
corpus    -> corpus/windows.npy        int32 (n_windows, window); ~7.5k windows for pile-10k
             corpus/meta.json          source, n_docs, n_tokens, n_windows, shuffle, seed, source mix
extract   -> extract/acts.npy          (n_layers+2, vocab, d) f16, unit-normed mean
             extract/acts_rawmean.npy  same shape, plain mean
             extract/counts.npy        int64 (vocab,), counted occurrences
             extract/self_sim.npy      f32 (n_layers+2, vocab), NaN where count < 2
             extract/self_sim_baseline.json   one float per frame
             + everything v0 extract writes (labels, tokens, layer_names, embed, unembed,
               final_ln, freq_ranks, freq_bins)
normalize -> reads acts.npy or acts_rawmean.npy per normalize.source; otherwise unchanged
metrics   -> eligible = counts >= min_count; subsample from eligible only;
             corpus-count freq bins; self-sim curves
viz       -> unchanged, plus trajectory groups and counts in labels
v0v1      -> runs/v0v1_<v0 run name>/v0v1.json + v0v1.png  (new cross-run command)
```

`token-drift all` runs `corpus` first when the config has a `corpus:` block.
Frames are the v0 frames: embed, blocks 1..n-1, block n pre-final-LN, block n
post-final-LN. The unembed is appended as a pseudo-layer from `normalize` on, as in v0.
Rows are indexed by token id. The on-device sums are sized to the model's embedding
rows (50,304 for Pythia, padded); saved arrays are trimmed to the tokenizer's 50,277
rows, as v0 does for embed/unembed. Zero-count rows are zero vectors and are never
eligible.

## Components

### `corpus.py` (new)

- `download_parquet(source) -> Path`: `huggingface_hub` into the HF cache. Not unit-tested.
- `read_texts(path, text_field) -> list[str]` via `pyarrow` (new dependency).
- `pack(docs_token_ids, eos_id, window, max_tokens) -> np.ndarray (n_windows, window)`:
  concatenate docs with one EOS after each, cut into windows, drop the partial tail.
- `shuffle_within_windows(windows, seed) -> np.ndarray`: independent seeded
  permutation per row.
- `source_mix(meta_column) -> dict`: counts of `pile_set_name`, for `meta.json` only.

### `extract.py`: corpus mode (new function)

`extract_corpus_means(model, windows, *, eos_id, min_context, batch_size, device)`
returns `(unit_mean, raw_mean, counts, self_sim, self_sim_baseline)`.

Per batch, with the existing pre-LN hook producing the same frames as v0:

1. keep = `position >= min_context & token != eos_id`
2. per frame: `h` = kept vectors upcast to float32;
   `sum_unit[f].index_add_(0, ids, h / ||h||)`, `sum_raw[f].index_add_(0, ids, h)`
3. `counts += bincount(ids)`

Sums live on the device in float32: 2 x 8 x 50,304 x 512 x 4 B = 1.65 GB. float32 is
enough: the most frequent token (" the") gets ~430k additions, so the typical relative
rounding error is ~sqrt(n) x 1e-7 = 1e-4.

At the end: `mean = sum / count` (zero rows stay zero), cast to float16.

Self-similarity (Ethayarajh 2019) from the unit sums, exactly: for a token with n
occurrences and unit-vector sum S, `||S||^2 = n + sum over ordered pairs i != j of cos_ij`, so

```
self_sim = (||S||^2 - n) / (n (n - 1))      # mean pairwise cosine; NaN for n < 2
```

The anisotropy baseline (mean cosine between any two counted occurrences) uses the
same formula on the grand total `S_all = sum over tokens of S` with `N = sum of counts`,
one scalar per frame. Both use uncentered cosine, matching Ethayarajh's definitions.

### `metrics.py`

- `compute_all(..., eligible=None, subsample_idx=None, merge_rank_bins=None, self_sim=None,
  self_sim_baseline=None)`:
  - subsample drawn from `eligible` rows (same seed logic as v0); `subsample_idx`
    overrides it (used by `v0v1`)
  - `knn_change_by_freq` uses whatever `freq_bins` it's given (corpus bins for corpus
    runs); `knn_change_by_merge_rank` added when `merge_rank_bins` is passed
  - new keys: `self_sim` (mean over the subsample, per frame), `self_sim_baseline`,
    `self_sim_adjusted` = self_sim - baseline; `null` for v0 runs
- `labels.corpus_freq_bins(counts, eligible, n_bins=5)`: equal-count bins over
  eligible tokens, -1 for ineligible.

### `viz.py`

- A self-sim panel on `metrics.png` (self_sim solid, baseline dashed), drawn only when
  present.
- `trajectory_groups`: one `trajectories_<group>.png` per group; labels carry `n=<count>`.
  The viz subsample stays "metrics subsample + all trajectory tokens", so a trajectory
  word below `min_count` is still drawn (and labelled with its low n). Words that are not
  a single token, or have count 0, are skipped with a warning.

### `cli.py`

- `corpus` command + `stage_corpus`; `all` calls it first when `corpus:` is present.
- `stage_extract` branches on `extract.mode`.
- `stage_normalize` picks the source array.
- `stage_metrics` builds `eligible`, corpus freq bins, and passes self-sim through.
- `v0v1 <v0 run dir> <v1 run dir>`: loads both normalized stacks and v1's
  `subsample_idx`; recomputes v0's metrics on exactly those tokens; computes a new
  per-frame cross overlap (kNN overlap between v0's frame f and v1's frame f, same
  tokens); writes `runs/v0v1_<v0 name>/v0v1.json` and a png that overlays v0 and v1
  curves plus the cross-overlap curve. v0's own `metrics/` is never touched.

## Where each open question gets answered

| question | read off |
|---|---|
| Q6 current-token info fades? | `knn_vs_first`, v0 vs v1; the v0v1 cross overlap by depth |
| Q7 Voita frequency effect with context? | `knn_change_by_freq` with corpus bins |
| Q3(b) does the last state move toward the unembed? | `knn_vs_last` at L6, v0 vs v1 |
| Q9 Ethayarajh self-similarity | self-sim panel; adjusted self-sim by depth |

## Errors

- Trajectory word not a single token, or count 0: warn and skip.
- Fewer eligible tokens than `subsample`: use all of them, warn.
- `normalize.source: raw_mean` on a vocab-mode run: fail fast with a clear message.
- Download errors propagate.

## Testing

Tiny and offline, reusing the random GPT-NeoX fixture from `tests/test_extract.py`.

- corpus: exactly one EOS between docs; partial tail dropped; `max_tokens` respected;
  shuffle keeps each window's token multiset and is deterministic per seed.
- extract corpus mode: sums equal a plain Python loop over the same windows; frame 0 unit
  mean equals `embed / ||embed||` and its self-sim is 1; the self-sim formula equals
  brute-force pairwise cosines; positions < `min_context` and EOS are never counted;
  results don't depend on `batch_size`.
- metrics: the subsample is a subset of eligible; corpus freq bins are equal-count.
- cli: fake corpus -> extract -> normalize -> metrics -> viz end to end; `v0v1` writes
  json with frame-0 cross overlap = 1.0; the v0 cli tests still pass unchanged.

Baseline before starting: 86 tests pass.

## Order of work (one small commit each, `pytest` before each)

1. `corpus` stage (`corpus.py`, cli wiring, `pyarrow` dependency)
2. `extract` corpus mode
3. `metrics`: eligible, corpus freq bins, self-sim (+ panel)
4. the three configs; `all` runs `corpus` first
5. `v0v1` command
6. trajectory groups in viz
7. run: re-extract v0 `pythia70m` and `random_init`
   (their `extract/`/`normalize/` were deleted 2026-09-24 to free disk), run the three v1
   runs, then `v0v1` for (pythia70m, pythia70m_corpus) and (random_init, random_init_corpus)
8. FINDINGS section 11, OPEN_QUESTIONS updates, README section

Disk: ~1.4 GB per v1 run, ~6 GB total with the two v0 re-extracts (15 GB free).

## Out of scope for A

Probe corpus and per-occurrence storage (milestone B, with its own min-count);
GPT-2 v1; global-shuffle control; position-binned averages; intrinsic dimension.
