# token-drift

Push every token in `pythia-70m`'s vocabulary through the model and watch how the
vocab's cluster structure changes layer by layer. Numeric drift curves plus an
aligned-UMAP flipbook. Random-init control included so we can tell learned structure
from architectural structure.

Design: `docs/EXPERIMENT.md`. How we work: `CLAUDE.md`. What we found, in full:
`docs/FINDINGS.md`. What we don't know yet: `docs/OPEN_QUESTIONS.md`.

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

## Checked against the literature

Local copies of the papers live in `papers/` (gitignored). Only the two required
references are checked here; Cheng et al. and Viswanathan et al. are v1 material.

### Ethayarajh (2019): anisotropy. Reproduces.

His headline for GPT-2: random word pairs have mean cosine ~0.6 in layers 2–8, rising
to almost 1.0 at layer 12. Ours (bottom-left panel above, computed on the *raw*
activations before centering):

| | L0 | L1–L5 | L6 (post-LN) | unembed |
|---|---|---|---|---|
| pythia-70m | 0.01 | 0.54–0.72 | 0.96 | 0.92 |
| random init | 0.00 | 0.19–0.41 | 0.43 | 0.00 |

Same shape, same near-1.0 final layer. Three things his paper didn't have:

- **Layer 0 is isotropic here.** Pythia uses rotary position, so `hidden_states[0]` is the
  bare embedding. GPT-2's layer 0 has a learned positional embedding added, which is
  why his curve starts at 0.6 instead of 0.
- **The random-init control is anisotropic too, and it grows with depth.** He calls
  anisotropy "inherent to, or a by-product of, contextualization". The control says a
  chunk of it is architectural. In our `[BOS, tok]` setup there's an obvious mechanism:
  position 1 attends to BOS, the BOS value is the same vector for every token, and the
  residual stream accumulates it. Partly a v0 artefact; v1 will tell.
- **The final LayerNorm is violent.** Mean row norm goes 14 (L5) -> 438 (L6), and the top
  PC explains 45% of L6's centered variance vs 11% at L5 (the triangle line in the
  anisotropy panel). That's a massive-activation dimension,
  and it's where `drop_top_pcs` would bite. The unembed's 0.92 turns out to be a mean
  offset rather than a variance direction (top PC only 5%), so centering handles it.

His other measures (self-similarity across contexts, intra-sentence similarity,
maximum explainable variance) need multiple contexts per word: v1.

### Voita, Sennrich & Titov (2019): bottom-up evolution. Mostly needs v1.

- *LM representations lose information about the current token with depth and build
  information about the next token.* **Untestable in v0 by construction**: with
  `[BOS, tok]` input every layer is a function of the token alone, so identity is never
  lost. Our purity curve is the coarse version of "identity retained" and it doesn't
  fade. This is the strongest argument for the corpus-averaged v1.
- *Change between consecutive layers is non-monotonic for LMs, with a spike at the top
  (their fig. 3b).* Partial echo: our change is high at L4->L5 and at L6->unembed, but
  our biggest jump is L0->L1, which they don't see. Different measure (PWCCA vs kNN
  Jaccard), different data (contextual vs context-free). Don't over-read.
- *Frequent tokens change more per layer, and the effect fades at the top (their
  fig. 4b).* **Does not reproduce** (bottom-right panel). Frequency here is BPE merge
  rank, which for both tokenizers is exactly token-id order. Across the five merged-token
  quantile bins the per-layer change differs by at most 0.03 and, if anything, rarer
  tokens change slightly *more*. Informative rather than disappointing: their effect
  comes from frequent tokens receiving more contextual updating, and there is no
  context here. The one bin that stands out is base/byte tokens (bin 0), which change
  *less* through L1–L4: those are the digits and punctuation that sit in tight islands.
- *Tokens with similar next-token distributions merge in upper LM layers (their
  "is/are/was/were" t-SNE).* Same phenomenon as our `" the"`/`" a"`/`" an"` convergence
  at the unembed.

## GPT-2 small, first look

![gpt2 metric curves](docs/results/gpt2_metrics.png)

| layer 0 (wte + wpe[1]) | L12 (post final-LN) |
|---|---|
| ![](docs/results/gpt2_L0.png) | ![](docs/results/gpt2_L12.png) |

Flipbook: [`docs/results/gpt2_flipbook.gif`](docs/results/gpt2_flipbook.gif). The L12
frame is a hollow ring: after centering, a representation dominated by one huge shared
direction leaves the tokens on a shell around it. Surface form is still visible on the
ring (purity 0.63), just smeared.

`configs/gpt2.yaml`. Tied embeddings, so the unembed frame *is* layer 0 again: kNN
overlap 0.999 and CKA 1.00 between them, which is the sanity check passing, not a
finding. What is a finding:

- **The middle is a plateau, the top is a cliff.** Consecutive-layer overlap sits at
  0.68–0.78 from L1 all the way to L10, then L11->L12 collapses to 0.08 (CKA 0.37).
  Pythia's worst consecutive step was 0.36. Hooking the pre-LN residual (2026-09-24)
  says it's the final LayerNorm, specifically its gain: block 12 alone keeps 0.40 of
  neighbours, the LN gain then drops it to 0.09. Details in FINDINGS section 8.
- **Ethayarajh's GPT-2 anisotropy curve, reproduced almost point for point.** 0.72 at
  L0 (that's the added positional embedding; the bare `wte` alone is 0.27, see the
  unembed point), dipping to ~0.6 through L4–L7, then climbing to 0.98 at L12. His
  figure 1 shows 0.6 flat through layers 2–8 and ~0.98 at 12.
- **Surface form again never fades** (purity 0.63–0.77 everywhere vs 0.25 shuffled),
  and base/byte tokens again change far less per layer than merged tokens, with no
  difference between frequency quantiles among the merged ones.

**Timing.** extract 18 s on an RTX 5060, metrics ~30 s, AlignedUMAP on 10k tokens x 8
frames **48 minutes** when two runs share the CPU, 19 minutes for GPT-2's 14 frames
running alone. The flipbook is the whole budget; `viz.method: stacked_umap` is the fast
knob if you want a quick loop.
