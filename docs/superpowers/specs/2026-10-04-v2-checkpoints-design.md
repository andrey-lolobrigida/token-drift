# v2: Pythia-70m training checkpoints (design)

Status: design approved in chat 2026-10-04, spec awaiting review.
Scope: v0 mode (token-alone) only, Pythia-70m only, ~10 checkpoints. Kept small on purpose.

## Why

Every run so far looks at a finished model. EleutherAI saved Pythia every so often during
training (154 snapshots, `step0` .. `step143000`, each a git branch on the Hub called a
*revision*). Running the v0 pipeline on a handful of them adds a time axis and answers
two questions OPEN_QUESTIONS has carried since v0:

1. When during training does layer 0 cluster by surface form (space-prefix, digits,
   punctuation, casing)?
2. When does the stable middle block (high consecutive kNN overlap in L1..L5) form?

Plus one prediction written down before running anything (section C below): embedding
rows only get a gradient when their token shows up in a batch, unembedding rows get one
at every position (softmax pushes every wrong token down). So the unembed should move
early and evenly across frequency bins, the embed's rare tokens should lag far behind,
and never-seen tokens should stay at their random start apart from weight decay.

## Decisions (and why)

- **v0 mode only.** Cheapest mode, and the one where layer-0 surface-form clustering is
  clearest. Corpus / probe modes on checkpoints are out of scope.
- **Log-spaced checkpoints:** `step0, step1, step8, step64, step512, step1000, step4000,
  step16000, step64000, step143000`. Most change is expected early; even spacing would
  mostly sample the slow tail. All 10 verified to exist on the Hub (2026-10-04).
- **One config, one run dir, a subfolder per revision** (chosen over one config per
  checkpoint: ten near-identical yamls would drift apart).
- **No per-checkpoint flipbooks** (AlignedUMAP is 20-50 min each). Instead,
  **training-time flipbooks**: frames = checkpoints, for three fixed frames:
  L0 (embedding), L3 (middle of the stable block), unembed.
- **Tokenizer from the final revision.** It's the same for every checkpoint.
- `step143000` = the same weights as `runs/pythia70m`, which gives a built-in check that
  revision loading works.

## Config

`configs/pythia70m_ckpt.yaml` = `configs/pythia70m.yaml` plus:

```yaml
run_name: pythia70m_ckpt
revisions: [step0, step1, step8, step64, step512, step1000, step4000, step16000, step64000, step143000]
timeline:
  frames: [0, 3, unembed]   # frames that get a training-time flipbook + with-final curves
  keep_acts: false          # delete each revision's extract/ + normalize/ after copying frames
```

`frames` takes frame indices into the normalized stack (0 = embed) or the name `unembed`
(the pseudo-layer appended by normalize).

## Data flow

```
runs/pythia70m_ckpt/
  config.yaml
  step0000000/                     # zero-padded to 7 digits so folders sort in step order
    extract/  normalize/           # deleted after use unless keep_acts
    metrics/metrics.json, metrics.png, cka.png
    weights/embed.npy, unembed.npy # raw matrices, full vocab, float16 (~50 MB each)
    frames/frames.npy              # (n_timeline_frames, n_sub, d) float16, normalized,
                                   # rows = the metrics subsample (+ trajectory tokens)
    frames/idx.npy                 # which token ids those rows are
  step0000001/ ...
  timeline/
    timeline.json                  # with-final overlap + CKA, drift per bin, the step list
    timeline.png                   # panels A + B
    drift.png                      # panel C
    flipbook_L0.gif, flipbook_L3.gif, flipbook_unembed.gif
    umap_<frame>_<step>.png, umap_coords_<frame>.npy
```

`token-drift all --config configs/pythia70m_ckpt.yaml`:

1. Check every revision exists on the Hub (`huggingface_hub.list_repo_refs`) before any
   download; stop with a message naming the missing ones.
2. For each revision: if its `metrics/metrics.json` exists, skip (resume after a crash).
   Otherwise run the existing extract -> normalize -> metrics on a sub-config whose run dir
   is the step folder, copy `frames/` and `weights/`, then delete `extract/` + `normalize/`
   unless `keep_acts`.
3. `timeline` stage across all revisions.

The metrics subsample and viz rows come from the config seed, so every revision uses the
same tokens. The timeline stage asserts it (identical `idx.npy` and `subsample_idx`
everywhere) and refuses to run otherwise.

Disk: peak ~1 GB for the revision in flight, plus ~1 GB `weights/` and ~300 MB `frames/`
in total. HF cache grows ~160 MB per revision (~1.6 GB); mention at the end so Andrey can
clear it.

## What timeline measures

**A. Per-checkpoint curves (no new math, read from the 10 metrics.json):**

| Question | Metric | Frames |
|---|---|---|
| When does surface-form clustering appear? | kNN purity minus shuffled baseline; silhouette | all, L0 + unembed highlighted |
| When does the stable middle block form? | consecutive kNN overlap | transitions L1->L2 .. L4->L5 |
| When does the cone form? | anisotropy, top-PC share | all |

**B. How finished is the geometry?** For each timeline frame and step, against `step143000`,
on the subsample:
- kNN overlap with final (mean Jaccard of k=10 neighbour sets) = local;
- linear CKA with final = global.

Both are reused from `metrics.py`. They can disagree (globally in place but neighbourhoods
still reshuffling, or the reverse); that split is the interesting part (cf. FINDINGS 11.2).

**C. Row drift from init.** For W in {embed, unembed} and every step t:
drift_i(t) = ||W_t[i] - W_0[i]|| / ||W_0[i]||, median per frequency bin (the same merge-rank
bins as metrics). Full vocab, from `weights/`. Prediction above.

**Plots:** `timeline.png` (A + B, x = step on a symlog axis, so step 0 sits at the left edge;
plain log can't show 0), `drift.png` (C, embed solid / unembed dashed, one colour per freq
bin), three flipbooks (one AlignedUMAP fit each over the 10 checkpoints, same axes on every
page, coloured by category, page titles = step names).

## Components

- `extract.py`: `load_model(..., revision=None)` passes `revision` to `from_pretrained`.
  Setting it together with `random_init: true` is an error.
- `cli.py`: `revisions:` in the config switches `all` to the checkpoint loop above;
  `_revision_cfg(cfg, rev)` builds the sub-config; new `token-drift timeline --config ...`.
  No stage reads another stage's files except through `cli.py` (as now).
- `timeline.py` (new), arrays in, arrays out: `overlap_with_final`, `cka_with_final`,
  `row_drift(W_t, W_0) -> per-row array`, `median_by_bin`.
- `viz.py`: `plot_timeline`, `plot_drift`; flipbooks reuse `project_layers` + `plot_flipbook`
  with step names as frame labels.

## Errors

- Missing revision on the Hub -> stop before downloading anything.
- `revisions:` together with `corpus:` / probe mode -> refuse (out of scope).
- Mismatched subsample across revisions -> timeline refuses, naming the bad revision.
- Timeline with a missing step folder -> refuse, list which revisions still need `all`.

## Testing (pytest, synthetic, no downloads)

- revision reaches `from_pretrained` (monkeypatched);
- step-folder naming is zero-padded and sorts in step order;
- `timeline.py` on toy arrays: identical arrays -> overlap 1, CKA 1, drift 0; rows left
  untouched in a toy "training" have drift 0; median per bin on a hand-checked case;
- resume skips a revision with an existing metrics.json;
- mismatched `idx.npy` -> refusal.

## Sanity checks on the real run (before believing a curve)

1. `step143000` metrics match `runs/pythia70m/metrics/metrics.json` within ~1e-3.
2. `step0` vs `runs/random_init`: expect similar, not identical (HF's init may not be
   EleutherAI's). Report the gap either way.
3. Never-seen tokens: the embed rows with the lowest drift at `step143000` should be
   tokens that (nearly) never appear in the Pile. Eyeball the bottom 20.

## Order of work (one small commit each, `pytest` before each)

1. `revision` in `load_model` + config validation + Hub check.
2. Checkpoint loop in `all` (sub-config, resume, frames/weights copy, cleanup).
3. `timeline.py` math.
4. `timeline` stage + `plot_timeline` / `plot_drift`.
5. Training-time flipbooks.
6. Real run + sanity checks (with Andrey).
7. Write-up: FINDINGS 13, README v2 paragraph, OPEN_QUESTIONS v2 entry, CLAUDE.md mode note.

## Definition of done

- `token-drift all --config configs/pythia70m_ckpt.yaml` runs end to end.
- `timeline.png`, `drift.png` and the three flipbooks exist; sanity checks 1-3 reported.
- FINDINGS 13 says when (which step) L0 clustering and the stable middle block appear,
  and whether the drift prediction held. Surprises flagged, not smoothed over.

## Out of scope

Other sizes (160m / 410m), corpus or probe modes on checkpoints, per-layer flipbooks per
checkpoint, more than ~10 checkpoints. Filling a gap (e.g. between step512 and step1000)
is a one-line config change later.
