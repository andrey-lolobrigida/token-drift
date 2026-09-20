# Findings, v0 (context-free vocab pass)

What we measured on 2026-09-16, with numbers. The README has the short version and
the figures; this is the long one. All numbers are on a fixed 10k-token subsample
(same subsample for every layer), k=10 for kNN, seed 0. "unembed" is the output
embedding matrix treated as a pseudo-layer after the last hidden state.

Runs: `pythia70m` (EleutherAI/pythia-70m, 6 layers, d=512, untied embeddings),
`random_init` (same architecture, fresh weights, seed 0), `gpt2` (12 layers, d=768,
tied embeddings). Raw outputs are in `runs/<name>/`, figures in `docs/results/`.

## 1. Surface form is baked in at layer 0 and never leaves

kNN category purity = fraction of a token's 10 nearest neighbours that share its
heuristic surface-form category (space-prefix, casing, digit, punct, ...). Shuffled
labels give 0.23 (pythia) / 0.25 (gpt2) at every layer.

| | L0 | min mid-model | last hidden | unembed |
|---|---|---|---|---|
| pythia70m | 0.72 | 0.66 (L5) | 0.68 | **0.88** |
| gpt2 | 0.74 | 0.63 (L1) | 0.63 | 0.74 (= L0, tied) |
| random_init | 0.23 | 0.22 | 0.23 | 0.23 |

- Digits and punctuation are islands in every frame of every trained model.
- Pythia's **unembedding is the most surface-form-organised matrix of all**. Per
  category at the unembed: digit 1.00, punct 0.94, space_lower 0.94, space_cap 0.87,
  lower 0.85, cap 0.84. At L0 the same categories are 1.00 / 0.92 / 0.83 / 0.63 /
  0.63 / 0.41. The "no leading space, capitalized" bucket in particular is a mess at
  L0 and clean at the unembed.
- Silhouette on the same labels is ~0 everywhere (pythia 0.00 to 0.02, gpt2 0.01 to
  0.03, control -0.01 to -0.02). It's the wrong tool in 512-d: it wants compact convex
  clusters and these are ribbons. Purity is the number to use.

## 2. Reorganisation happens at the ends, not the middle

Consecutive-layer kNN overlap (mean Jaccard of neighbour sets; 1 = nothing moved).

| transition | pythia70m | random_init | gpt2 |
|---|---|---|---|
| L0 -> L1 | 0.36 | 0.06 | 0.42 |
| middle layers | 0.48 / 0.56 / 0.49 / 0.36 | 0.20 -> 0.48 rising | 0.68 to 0.78, flat |
| L(n-1) -> L(n) post-LN | 0.42 | 0.48 | **0.08** |
| last hidden -> unembed | 0.13 | 0.00 | 0.05 |

- Both trained models have a **stable middle block**. In CKA terms pythia L1 to L4 sit at
  0.81 to 0.93 pairwise; gpt2 L1 to L10 at 0.95 to 0.98 consecutive. The prediction
  in EXPERIMENT.md ("middle layers: the biggest consecutive-layer drop") was wrong.
- GPT-2's last step is a cliff. L11 -> L12 overlap 0.08, CKA 0.37, k-means ARI 0.13.
  Pythia's worst step other than the unembed is 0.36. We can't yet say whether the
  cliff is the last block or the final LayerNorm (see OPEN_QUESTIONS).
- In both trained models the k-means ARI curve (k=20) sits on top of kNN overlap
  (pythia 0.40 / 0.49 / 0.56 / 0.51 / 0.46 / 0.42 / 0.19 vs overlap 0.36 / 0.48 / 0.56 /
  0.49 / 0.36 / 0.42 / 0.13), so the drift curve isn't a k=10 artefact. It does *not*
  track in the control; see section 3.

Drift from layer 0 (kNN overlap vs L0): pythia decays 0.36 -> 0.20 over six layers;
gpt2 0.42 -> 0.24 over eleven, then 0.05 at the post-LN layer. Slow and steady, then
the LN.

## 3. The random-init control is not boring in the way we expected

Prediction: "layer 0 is pure noise and every consecutive-layer overlap is high
(residual just carries the input through)". Half right.

- Purity is exactly at chance everywhere (0.23). Category structure is learned. Good.
- But a random first layer **scrambles** neighbourhoods almost completely (L0 -> L1
  overlap 0.06), and then consecutive overlap **rises** with depth (0.20, 0.31, 0.38,
  0.44, 0.48). CKA consecutive rises 0.64 -> 0.96. Each further random layer perturbs an
  accumulated residual proportionally less.
- So "upper layers look stable" is partly the architecture. Pythia's mid-layer overlap
  (0.48 to 0.56) is not dramatically above the control's L3 to L5 (0.38 to 0.48). What
  training clearly buys is the first step (0.36 vs 0.06) and the drift-from-L0 curve
  (0.20 to 0.36 vs 0.02 to 0.06).
- The random unembed shares nothing with anything (overlap 0.00, CKA 0.03 to 0.05), as
  it should.
- kNN overlap and k-means ARI **disagree** here, and that's the useful part. Overlap
  climbs to 0.48 by L5 -> L6 but ARI stays at chance the whole way (0.00, 0.02, 0.04,
  0.06, 0.06, 0.08). Local neighbourhoods carry over from one random layer to the next,
  but there is no coarse structure for k-means to latch onto, so each layer gets an
  arbitrary 20-way carving and consecutive carvings don't agree. Local persistence
  without global structure. In the trained models both persist (ARI 0.4 to 0.7), which
  is a label-free way of saying the cluster structure is learned. Added 2026-09-18.

## 4. Anisotropy (Ethayarajh 2019) reproduces, with two additions

Mean cosine between random token pairs on the *raw* (uncentered) activations.
0 = isotropic, 1 = everything in one cone.

| | L0 | mid | last hidden (post-LN) | unembed |
|---|---|---|---|---|
| pythia70m | 0.01 | 0.54 to 0.72 | **0.96** | 0.92 |
| gpt2 | 0.72 | 0.59 to 0.73 | **0.98** | 0.27 |
| random_init | 0.00 | 0.19 to 0.41 | 0.43 | 0.00 |

- GPT-2 matches Ethayarajh's figure 1 almost point for point: ~0.6 through the
  middle, near 1.0 at layer 12. He didn't have Pythia; it has the same shape.
- **Layer 0 differs by position encoding.** Pythia (rotary) starts isotropic. GPT-2
  adds `wpe[1]` to every token at layer 0, hence 0.72 there; the bare `wte` (the
  unembed point, since weights are tied) is 0.27.
- **Random init is anisotropic too and grows with depth.** Ethayarajh read anisotropy
  as a by-product of contextualization. The control says a good part is
  architectural. Suspected mechanism in our setup: position 1 attends to BOS, the BOS
  value contribution is identical for every token, and the residual accumulates it.
- Pythia's final LayerNorm is violent: mean row norm 14 at L5 -> 438 at L6, and the top
  PC explains 45% of L6's centered variance vs 11% at L5 (the triangle line in the
  anisotropy panel). A massive-activation dimension. The
  unembed's 0.92 is a mean offset, not a variance direction (top PC 5%), so centering
  handles it; the L6 one survives centering as the hollow-ring look in the flipbook.
- **The L6 offset is the final LayerNorm's bias, by name** (added 2026-09-18). The
  mean L6 vector has cosine 0.995 with `final_layer_norm.bias` (|bias| = 160 against
  normalized rows of ~23), and the offset accounts for 97.4% of each row's squared
  norm. So the 0.96 anisotropy is a rigid translation of the whole cloud and
  centering undoes it exactly. What centering *doesn't* undo is the gain side: the
  three coordinates with the largest mean (169, 181, 336) have LN gains of 17 to 20
  vs a median of 11, and after centering the top PC still carries 39% of the
  variance. That's the massive dimension: a per-axis stretch, not an offset.
- **Top-PC share is now a curve** (added 2026-09-20): `top_pc_share` in metrics.json,
  triangles in the anisotropy panel, computed on the same 10k subsample as everything
  else. Pythia: 0.01 at L0, 0.07 to 0.12 through L1 to L5, 0.45 at L6, 0.04 at the
  unembed (0.43 at L6 on the full vocab). GPT-2: 0.02 to 0.07 through L0 to L11, 0.57
  at L12, 0.02 at the unembed. Random init never exceeds 0.02. The 39% above was a
  one-off check on 09-18; the panel is the number of record. Two things the pair of
  lines says at a glance: GPT-2's 0.72 mean-cos at L0 is pure offset (share 0.02), so
  `wpe[1]` shifts the cloud without reshaping it; and the last-layer stretch is
  *bigger* in GPT-2 than in Pythia even though Pythia's offset is the larger one.
- Caveat on the metric: centered mean-cosine is ~0 for *any* centered cloud, ball or
  cigar (a synthetic 512-d cigar with one axis 20x wider reads -0.004 with a 44%
  top-PC share). Mean cosine only detects a shared offset; top-PC variance share is
  the shape check. Both are needed to read a layer.

## 5. Voita et al. (2019): the testable part does not reproduce context-free

Their figure 4b: frequent tokens change more per layer in a trained LM. Frequency here
is BPE merge rank, which for both tokenizers is exactly token-id order.

- Across the five merged-token quantile bins, per-layer change differs by <= 0.03 in
  every run, and rarer tokens change marginally *more*. No effect.
- The one bin that stands out is base/byte tokens (bin 0): they change **less** through
  the stable middle (pythia 0.41 to 0.47 vs 0.44 to 0.52 for merged tokens; gpt2 0.13
  to 0.20 vs 0.22 to 0.28). Those are the digit and punctuation islands.
- Reading: Voita's frequency effect comes from frequent tokens receiving more
  *contextual* updating. There is no context in v0, so there is no effect. This is a
  clean negative, and it's the strongest argument for the corpus-averaged v1.
- Their current-token-information-fades claim is untestable in v0 by construction
  (every layer is a function of the token alone). Their "is/are/was/were merge in
  upper LM layers" is the same phenomenon as our `" the"`/`" a"`/`" an"` convergence
  at the unembed (see `runs/pythia70m/viz/trajectories.png`).

## 6. Bookends: embed vs unembed (Pythia only, untied)

- kNN overlap between the input embedding and the unembedding: 0.22. CKA 0.46.
- The final hidden state is *less* like the unembed than the input embedding is
  (overlap 0.13 vs 0.22, CKA 0.37 vs 0.46). See OPEN_QUESTIONS.
- GPT-2's unembed frame is layer 0 exactly (overlap 0.999, CKA 1.00). That's the tied
  weights, and it's a sanity check on the pipeline, not a finding.

## 7. Things about the method itself

- Centering is non-negotiable. Without it the raw anisotropy numbers above mean every
  upper layer is one cone and cosine kNN measures "how far along the shared direction".
- kNN purity beats silhouette in high dimension for this kind of question.
- AlignedUMAP on 10k tokens: 19 min for 14 frames alone on 12 cores, 48 min for 8
  frames when two runs shared the CPU. Everything else is about a minute.
- The viz subsample is the metrics subsample (10k) plus the trajectory tokens forced in,
  not the full 50k vocab.
