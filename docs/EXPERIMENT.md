# Experiment design: how does the vocab's geometry drift across layers?

## The question

A language model starts with a fixed embedding for every token (a dictionary).
Each layer rewrites that vector. By the last layer the vector is supposed to encode
"what comes next", not "what this token is". So the question is: **at what layers,
and how fast, does the vocabulary reorganize?** And does the reorganization look
like surface-form → meaning → prediction, as the literature suggests?

Background reading (only two are required):

- Voita, Sennrich & Titov (2019), *The Bottom-up Evolution of Representations in the
  Transformer*. The "what to expect" paper. In left-to-right LMs, info about the
  current token fades going up and info about the *next* token builds.
- Ethayarajh (2019), *How Contextual are Contextualized Word Representations?*
  The "why your plot is a blob" paper. Everything lives in a narrow cone (anisotropy),
  and it gets narrower in upper layers. Center before doing anything.
- Nice-to-have: Cheng et al. (2024), *Emergence of a High-Dimensional Abstraction
  Phase in Language Transformers*; Viswanathan et al. (2025), *The Geometry of Tokens
  in Internal Representations of LLMs*. Intrinsic-dimension and neighborhood-overlap
  curves across layers, with shuffled-text and random-init controls.

## Manufacturing a "vocab embedding" for layer L

There isn't one. Two ways to fake it:

**v0 (this repo): context-free.** Feed `[BOS, tok]` for every `tok` in the vocab,
take the residual stream at position 1 after layer L. With only BOS to attend to,
this is mostly "what the MLPs do to a token in isolation". Cheap: ~50k forward
passes of length 2.

**v1 (later): corpus-averaged.** Run a corpus, average each token's residual over all
its occurrences. This is the "what does the model *usually* think this token means"
version, and it's where polysemy shows up (`bank` sits between money and river).
Slower, needs a corpus, and has a frequency-coverage problem for rare tokens.

Layer indexing: HF `hidden_states` has `n_layers + 1` entries. Index 0 is the
embedding output (= `embed_in[tok]`, possibly plus positional stuff depending on
architecture; for Pythia/GPT-NeoX with rotary embeddings, index 0 *is* the raw
embedding). Last index is after the final block, and for GPT-NeoX the final LayerNorm
has already been applied. Note this in the plots.
Since 2026-09-24 extract also hooks the *input* of the final LN and stores it as an
extra frame "L(n) pre-LN" before HF's last one, so `acts.npy` has `n_layers + 2` frames.

## Bookends

Pythia has untied embeddings, so there are *two* honest vocab-sized matrices:
`embed_in` (layer 0) and `embed_out` (the unembedding). Treat `embed_out` as a
pseudo-layer "L+1" in every metric and plot. It's the cleanest apples-to-apples
comparison in the whole experiment: two matrices, same rows, one is "identity",
one is "prediction".

## Normalization (do not skip)

Per layer:

1. Subtract the mean vector across the vocab.
2. Unit-normalize each row.
3. Optionally remove the top `k` principal components (Mu & Viswanath's
   "all-but-the-top", `k` in 1–3). Make it a config flag and try both.

Residual-stream norms grow with depth and a handful of "massive activation"
dimensions can dominate. If the flipbook looks like a single blob that just rotates,
this is why.

## Token categories (`labels.py`)

Cheap heuristics from the decoded token string. Used for coloring and silhouette.
Assign the first matching category:

| category      | rule                                                    |
|---------------|---------------------------------------------------------|
| `special`     | BOS/EOS/pad/etc.                                        |
| `byte`        | undecodable / byte-fallback tokens                      |
| `nonascii`    | any char outside ASCII                                  |
| `digit`       | all digits (after stripping a leading space)            |
| `punct`       | all punctuation / whitespace                            |
| `space_cap`   | leading space + capitalized word                        |
| `space_lower` | leading space + lowercase word                          |
| `cap`         | no leading space, capitalized                           |
| `lower`       | no leading space, lowercase (usually a word *piece*)    |
| `mixed`       | everything else                                         |

Also compute a frequency bin (from the tokenizer's merge rank as a proxy, or from a
corpus count if we have one). Frequency is famously one of the top PCs of embeddings.

## Metrics (`metrics.py`)

All computed per layer (and for the unembed pseudo-layer), on normalized acts.

1. **kNN neighborhood overlap between consecutive layers.** For each token, its
   k nearest neighbors (cosine, k=10) at layer L and at layer L+1; Jaccard of the two
   sets; average over vocab. This is *the* headline metric: it answers "how much did
   the neighborhood structure reorganize at this layer" with no projection involved.
   Also compute overlap of every layer vs layer 0 and vs unembed, for the drift curve.
2. **Linear CKA between all pairs of layers.** Gives an `(L+2) x (L+2)` similarity
   matrix; plot as a heatmap. Block structure = phases.
3. **Silhouette score** of the category labels, per layer. Tells us whether surface
   form is a good clustering at layer 0 and stops being one later.
4. **k-means ARI** between consecutive layers (k=20, fixed seed). Cheap cross-check
   on (1).
5. Optional: intrinsic dimension per layer (TwoNN). Only if the first four are done.

Use a random subsample of ~10k tokens for kNN/silhouette if the full vocab is slow.
Same subsample for every layer.

## Visualization (`viz.py`)

- `AlignedUMAP` over the list of per-layer arrays, with the identity relation
  between consecutive layers (same token index). This gives coordinates you *can*
  compare across frames.
- Fallback if AlignedUMAP is flaky: stack all layers into one `(L+2)*vocab` array,
  fit one UMAP, split back. Shared space, less pretty.
- One PNG per layer, fixed axis limits, colored by category. Stitch into a GIF.
- Bonus: pick ~30 hand-chosen tokens (`" bank"`, `" river"`, `" money"`, digits,
  punctuation, a few subword pieces) and draw their trajectories across frames.

## Controls

- **Random-init model, same config.** Whatever structure survives in the control is
  architectural, not learned. Expect layer 0 to be pure noise and every
  consecutive-layer overlap to be high (the residual stream just carries the input
  through).
- **Shuffled-labels silhouette.** Permute category labels; silhouette should drop to ~0.
- (v1) **Shuffled corpus** for the corpus-averaged version.

## What we expect to see (write down before running!)

- Layer 0: clear clusters by category. Leading-space vs not is probably the single
  biggest split. Digits and punctuation are tight islands.
- Early layers: high consecutive overlap; layer-0-vs-L overlap decays slowly.
- Middle layers: the biggest consecutive-layer drop (reorganization), silhouette on
  surface categories falls.
- Last layer / unembed: neighborhoods look like "tokens that predict similar next
  tokens". E.g. `" the"`, `" a"`, `" an"` might cluster because what follows them is
  similar, even though they were far apart earlier.
- Control model: flat, boring curves. If the control looks as interesting as the
  real model, we're measuring the architecture, not the learning.

Fill in the "what we actually saw" section in the README after the first full run,
and note anywhere the prediction was wrong. Those are the interesting bits.

## Phases

- **v0**: everything above with the context-free vocab pass. Target: one afternoon.
- **v1**: corpus-averaged activations, shuffled-corpus control, polysemy trajectories.
- **v2**: repeat across Pythia sizes (160m, 410m) and across training checkpoints
  (Pythia ships them) to see when the structure forms during training.
