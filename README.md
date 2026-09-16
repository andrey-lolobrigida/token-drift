# token-drift

Push every token in `pythia-70m`'s vocabulary through the model and watch how the
vocab's cluster structure changes layer by layer. Numeric drift curves plus an
aligned-UMAP flipbook. Random-init control included so we can tell learned structure
from architectural structure.

Design: `docs/EXPERIMENT.md`. How we work: `CLAUDE.md`.

## Quickstart

```
uv sync
uv run token-drift all --config configs/pythia70m.yaml
uv run token-drift all --config configs/random_init.yaml
```

Outputs land in `runs/<run_name>/`.

## Results (v0, context-free `[BOS, tok]`, first run 2026-09-16)

![metric curves, trained vs random init](docs/results/metrics_compare.png)

| layer 0 (input embedding) | unembedding matrix |
|---|---|
| ![](docs/results/pythia70m_L0.png) | ![](docs/results/pythia70m_unembed.png) |

Flipbook: [`docs/results/pythia70m_flipbook.gif`](docs/results/pythia70m_flipbook.gif).
CKA heatmap: [`docs/results/pythia70m_cka.png`](docs/results/pythia70m_cka.png).
Random-init layer 0 for comparison: [`docs/results/random_init_L0.png`](docs/results/random_init_L0.png)
(a featureless blob, as it should be).

**What we saw, in three sentences.** Surface form is baked in at layer 0 (digits and
punctuation are islands, space-prefix vs not is the big split) and it *never fades*:
72% of a token's 10 nearest neighbours share its category at layer 0, ~70% in the
middle, and 88% in the unembedding, which is the most surface-form-organised matrix
of the lot. The two big reorganisations are at the ends, embed -> L1 (kNN overlap
0.36) and L6 -> unembed (0.14), while the middle layers L1–L4 are a stable block
(CKA 0.8–0.9, overlap ~0.5). The random-init control is flat noise on every
category metric, but its consecutive-layer overlap *rises* with depth (0.06 -> 0.48),
so "the upper layers look stable" is partly the architecture talking, not learning.

**Predictions from `docs/EXPERIMENT.md` vs reality.**

- *"Layer 0: clear clusters by category."* Yes in the UMAP and in kNN purity. But
  silhouette on the categories is ~0.005 at every layer, indistinguishable from
  shuffled labels. Silhouette wants compact convex clusters and these are ribbons in
  512-d; kNN purity is the number to look at. (Added as a metric after the first run.)
- *"Middle layers: the biggest consecutive-layer drop."* Wrong. Middle layers are the
  *most* stable. The drops are L0->L1 and L6->unembed.
- *"Surface-form silhouette falls in later layers."* Purity dips a little (0.79 at L1
  to 0.67 at L5) and then jumps to 0.88 at the unembed. Surface form is reinforced,
  not forgotten, at the prediction end. Plausible reason: in a context-free pass the
  model's best guess about "what comes next" is mostly "what kind of token is this".
- *"`the`/`a`/`an` might cluster at the end."* They do; their labels land on top of
  each other in `trajectories.png`.
- *"Control: every consecutive-layer overlap high."* Wrong. A random first layer
  scrambles neighbourhoods almost completely (0.06), then each further random layer
  perturbs the accumulated residual proportionally less, so overlap climbs with depth.
  The trained L6 state and the unembed have nothing in common with each other in the
  control (overlap 0.001), while in the trained model L0 is *closer* to the unembed
  (0.22) than L6 is (0.14). That last one is worth a closer look.

**Timing.** extract 18 s on an RTX 5060, metrics ~30 s, AlignedUMAP on 10k tokens x 8
frames **48 minutes**. The flipbook is the whole budget; `viz.method: stacked_umap`
or a smaller viz subsample is the knob if you want a fast loop.
