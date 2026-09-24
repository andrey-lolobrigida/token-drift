# token-drift

Small mech-interp experiment: take every token in a small LM's vocab, push it through
the model, and watch how the *geometry of the vocabulary* changes layer by layer.
Layer 0 is the embedding matrix (a fixed dictionary). Deeper layers don't have a
"vocab matrix", so we manufacture one by running each token through the model.

Read `docs/EXPERIMENT.md` before writing code. It has the design, the metrics, the
controls, and the "things that will bite you" list. This file is about *how we work*.

## Goal, in one sentence

Produce (a) a per-layer numeric curve of how much the vocab's cluster structure
changes, and (b) a set of aligned UMAP plots you can flip through like a flipbook,
for a real model and a random-init control.

## Stack

- Python 3.11, `uv` for env/deps (`uv sync`, `uv run ...`)
- `torch` + `transformers` for the model. Use `output_hidden_states=True`; no need
  for TransformerLens unless we hit something HF can't do.
- `numpy`, `scikit-learn`, `umap-learn` (for `AlignedUMAP`), `matplotlib`
- Model: `EleutherAI/pythia-70m` (6 layers, d_model=512, vocab ~50k, **untied**
  embed/unembed — we want that for the layer-0 vs unembed comparison).
  Same config with random init is the control (`configs/random_init.yaml`).
- Tests: `pytest`, kept tiny and fast (use a 200-token vocab slice fixture).

## Repo layout

```
token-drift/
  CLAUDE.md
  README.md
  pyproject.toml
  configs/            # yaml: model name, layers, k for kNN, seeds, output dir
  docs/EXPERIMENT.md  # the science (design, metrics, predictions)
  docs/FINDINGS.md    # what v0 actually showed, with numbers
  docs/OPEN_QUESTIONS.md  # what's unresolved and how we'd attack it; read before starting a phase
  src/token_drift/
    extract.py        # vocab -> per-layer activations (cached to .npy)
    normalize.py      # per-layer centering / anisotropy correction
    labels.py         # heuristic token categories for coloring & silhouette
    metrics.py        # kNN overlap, CKA, silhouette, ARI
    viz.py            # AlignedUMAP flipbook + stacked-fit fallback
    cli.py            # `token-drift extract|metrics|viz|all --config ...`
  tests/
  runs/               # gitignored; one subdir per run, contains config copy + outputs
  papers/             # gitignored; local HTML copies of the reading list, for reference
```

## Pipeline (each stage caches to `runs/<name>/<stage>/`)

A run dir is `config.yaml` plus one subfolder per stage (`extract/`, `normalize/`,
`metrics/`, `viz/`), so you can nuke and redo one stage without hunting through a pile.

1. `extract` → `acts.npy` shape `(n_layers+2, vocab, d_model)`, float16.
   Input per token is `[BOS, tok]`; take the residual at position 1.
   Frames: embed, blocks 1..n-1, block n *pre*-final-LN (hook), block n post-LN (HF).
   Also save `embed.npy` (input embedding matrix), `unembed.npy` (output matrix) and
   `final_ln.npz` (final LayerNorm gain and bias).
2. `normalize` → `acts_norm.npy`. Center per layer, unit-norm rows. Optionally
   drop the top-k PCs ("all-but-the-top"). Never skip this stage; see EXPERIMENT.md.
3. `metrics` → `metrics.json` + `metrics.png` + `cka.png`. Curves over layers, plus the
   two literature checks: anisotropy on the *raw* acts (Ethayarajh) and neighborhood
   change per token-frequency bin (Voita). Frequency = BPE merge rank, saved at extract.
4. `viz` → `umap_layer_{i}.png` + `flipbook.gif` + `trajectories.png`.

The unembedding matrix is appended as a pseudo-layer from `normalize` onward, so
downstream arrays have `n_layers+3` frames.

`token-drift all --config configs/pythia70m.yaml` runs everything.

## Conventions

- Every run is reproducible from its yaml + seed. Copy the yaml into the run dir.
- Seed everything: torch, numpy, UMAP `random_state`, k-means `random_state`.
- Prefer numpy over torch once activations are extracted. Free the model after extraction.
- Batch the vocab pass (batch size ~512). Whole thing should take < 2 min on CPU.
- Functions take arrays and return arrays. No stage reads another stage's files
  except through `cli.py`.
- Label token categories heuristically (see `labels.py` spec in EXPERIMENT.md);
  don't overthink it, they're for coloring and a rough silhouette score.
- Plots: one figure per layer, same axes limits across the flipbook, colored by
  category, legend outside the plot. Also save the 2D coordinates as `.npy`.
- Small commits, one stage per PR-sized chunk. Run `pytest` before saying done.

## Don'ts (each one has burned someone)

- Don't compare raw UMAP/t-SNE coordinates across *separately fitted* projections.
  Use `AlignedUMAP`, or fit one UMAP on all layers stacked and then split.
- Don't run metrics on un-centered activations. Anisotropy will dominate and every
  layer will look like "one big cone".
- Don't use the final-layer hidden state without noting whether the final LayerNorm
  was applied. HF `hidden_states[-1]` for GPT-NeoX is *post*-final-LN. Document which.
- Don't index the residual at position 0 (that's BOS, same for every token).
- Don't trust t-SNE cluster *distances* between clusters. Only within-cluster
  neighborhoods mean anything.
- Don't load the whole `(7, 50k, 512)` tensor in float32 by accident. float16 on disk,
  upcast per layer when computing.

## Definition of done for v0

- `token-drift all` runs end to end on pythia-70m and on random-init.
- `metrics.png` shows kNN-overlap and CKA curves for both, on the same axes.
- Flipbook exists and layer 0 visibly clusters by surface form (space-prefix,
  digits, punctuation, casing). If it doesn't, something is wrong upstream.
- README has a "results" section with the two plots and three sentences of what
  we saw.

## Working with me (the human)

- I'm doing this to learn. When you make a non-obvious choice (normalization,
  k, UMAP params), leave a one-line comment saying *why*, not just what.
- If a result looks surprising, say so instead of smoothing it over. Surprising
  is the point.
- Keep the tone of docs and comments casual. No corporate voice.
