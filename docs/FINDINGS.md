# Findings

What we measured, with numbers: v0 (context-free vocab pass, sections 1-10, from
2026-09-16), v1 milestone A (corpus-averaged, 11), milestone B (virtues and vices, 12) and
v2 (training checkpoints, 13). The README has the short version and the figures; this is
the long one. Paths under `runs/` are local outputs (`runs/` isn't committed): rerun the
config or script named next to them to regenerate.

Sections 1-10 (v0): All numbers are on a fixed 10k-token subsample
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
| of which: L(n-1) -> L(n) pre-LN (block) | 0.24 | 0.49 | 0.40 |
| of which: pre-LN -> post-LN (the LN) | 0.29 | 0.91 | **0.09** |
| last hidden -> unembed | 0.13 | 0.00 | 0.05 |

- Both trained models have a **stable middle block**. In CKA terms pythia L1 to L4 sit at
  0.81 to 0.93 pairwise; gpt2 L1 to L10 at 0.95 to 0.98 consecutive. The prediction
  in EXPERIMENT.md ("middle layers: the biggest consecutive-layer drop") was wrong.
- GPT-2's last step is a cliff. L11 -> L12 overlap 0.08, CKA 0.37, k-means ARI 0.13.
  Pythia's worst step other than the unembed is 0.36. **The cliff is the final
  LayerNorm's gain, not the last block** (section 8, added 2026-09-24).
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
  centering undoes it exactly. What centering *doesn't* undo: after centering the
  top PC still carries 39% of the variance. ~~That's the massive dimension: a per-axis
  stretch from a few large LN gains~~ **Wrong, corrected 2026-09-24 (section 8):** the
  pre-LN residual already has a 65% top PC, Pythia's gains only span 1.8x max/median,
  and multiplying by the gain moves top-PC share 0.43 -> 0.45. The direction comes
  from block 6. It's also not a single coordinate (none holds >1% of the variance);
  it's a spread-out direction almost parallel to the mean (|cos| 0.99).
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

> Corrected in 11.2 (2026-10-03): merge-rank bins hid a local gradient (rare tokens'
> kNN change more), and this section compared our local kNN metric to Voita's global
> PWCCA one. On a global metric the context-free "no effect" mostly stands.

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
- The pre-LN L6 frame is further still (overlap 0.08), so the LN is not what's hiding
  an embed/unembed-style alignment (Q3 candidate (a) is out). Added 2026-09-24.
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

## 8. Q1: the last step, block vs final LayerNorm (2026-09-24)

`extract` now hooks the input of the final LN (`final_layer_norm` / `ln_f`) and stores
it as an extra frame "L(n) pre-LN" just before HF's post-LN one. Every other frame is
bit-identical to the 09-16 runs (checked metric by metric). Table in section 2.

To see *which part* of the LN does what, `final_ln.npz` holds its gain and bias, and a
throwaway script applied the LN one piece at a time to the pre-LN frame: normalize
(per-token mean/std), then x gain, then + bias. kNN overlap with the pre-LN frame after
each piece, and top-PC share (10k subsample, our usual center -> unit-norm pipeline):

| | normalize | x gain | + bias (= HF post-LN) |
|---|---|---|---|
| pythia70m overlap | 0.30 | 0.29 | 0.29 |
| pythia70m top PC (pre-LN: 0.65) | 0.43 | 0.45 | 0.45 |
| gpt2 overlap | 0.50 | **0.09** | 0.09 |
| gpt2 top PC (pre-LN: 0.27) | 0.07 | **0.57** | 0.57 |

**GPT-2: the cliff is the LN gain.** Block 12 is an ordinary step (0.40, same range as
L10 -> L11 at 0.40 to 0.68). Then the gain vector reshapes everything: it spans 13.9x
max/median, coordinate 496 has gain 17.4 (median 1.25) *and* a pre-LN mean of 112
(other coords ~1), while coords 481 and 373 get gains of 0.04, so they're basically
deleted. Multiplying by that vector alone takes overlap 0.50 -> 0.09 and creates the
0.57 top PC and the hollow ring in the flipbook. The bias does nothing to neighbourhoods
(it's a translation, centering eats it).

**Pythia: the story is block 6 plus our own normalization order, and the gain barely
matters.** Block 6 makes the amount of one shared direction (almost parallel to the
mean, |cos| 0.99) vary a lot between tokens: top-PC share 0.11 at L5 -> 0.65 pre-LN,
row-norm spread (std/mean) 8% -> 19%, and the top-PC score correlates 0.84 with row
norm. Our normalize stage centers *first* and unit-norms *after*, so that per-token
variation dominates each token's direction and swamps the token-specific part. The
LN normalizes each row by its own mean/std first, which squashes it. Hence the odd
result that L5 -> L6 post-LN (0.42) keeps more neighbours than either half-step (0.24,
0.29): the pre-LN frame is the odd one out, and mostly because of how *we* look at it.
If we unit-norm rows before centering, block 6 goes 0.24 -> 0.43 and the LN 0.29 ->
0.80. GPT-2 barely changes under the same swap (LN 0.09 -> 0.10), so its cliff is real.

Surprising, and worth keeping in mind: **"center then unit-norm" is not neutral when
row lengths vary a lot along a shared direction.** CLAUDE.md is right that centering is
mandatory, but the order relative to per-row scaling matters at exactly the layers where
something big happens. Added to OPEN_QUESTIONS under Q2.

Random init sanity check: an untrained LN has gain 1, bias 0, so it should only
rescale. Its pre -> post overlap is 0.91 (not 1.0 because the LN also subtracts each
row's own mean).

## 9. Q2: does the normalization change the story? (2026-09-24)

Two variants of the normalize stage, both trained models, everything else identical
(same extract, same 10k subsample): `drop2` = `drop_top_pcs: 2`, `rownorm` =
`row_norm_first: true` (unit-norm rows, *then* center, then unit-norm again). Raw-act
metrics (anisotropy, top-PC share) are identical across variants by construction.
Plots: `docs/results/q2_*.png`.

Consecutive kNN overlap, base / drop2 / rownorm:

| transition | pythia70m | gpt2 |
|---|---|---|
| L0 -> L1 | 0.36 / 0.41 / 0.37 | 0.42 / 0.44 / 0.42 |
| middle, range | 0.36-0.56 / 0.40-0.57 / 0.39-0.58 | 0.68-0.78 / 0.69-0.78 / 0.69-0.79 |
| L(n-1) -> pre-LN | **0.24** / 0.46 / 0.45 | 0.40 / 0.61 / 0.57 |
| pre-LN -> post-LN | **0.29** / 0.68 / 0.79 | **0.09** / 0.18 / 0.15 |
| post-LN -> unembed | 0.13 / 0.14 / 0.14 | 0.05 / 0.10 / 0.07 |

What holds:

- **The stable middle is not a normalization artefact.** Middle-layer overlap moves by
  at most ~0.05 in either model (GPT-2 L10 -> L11 is the biggest mover, 0.68 -> 0.76
  under rownorm). k-means ARI agrees.
- **GPT-2's final-LN cliff is real.** 0.09 becomes 0.18 / 0.15, still by far the worst
  step, and drift-from-L0 still collapses at post-LN (0.05 / 0.10 / 0.07).
- **Q3 survives.** Pythia's embed-vs-unembed overlap (0.22) stays above every L6 frame
  vs the unembed under every variant (0.08 to 0.15).

What doesn't:

- **Pythia's L5 -> L6 pre-LN "reshuffle" was the pipeline.** Either variant turns it
  into an ordinary step (0.46 / 0.45), and pre-LN purity comes back from 0.59 to 0.68.
  Confirms section 8: center-then-unit-norm magnifies a per-token spread along one
  big shared direction, and removing that direction (drop2) or the length differences
  (rownorm) fixes it about equally.

Costs, which is why neither becomes the default:

- **drop2 throws away real structure in GPT-2.** Middle-layer purity drops 0.77 ->
  0.66 at L6 (Pythia loses up to 0.06, at L0 and the unembed). So GPT-2's top two PCs in the middle are partly
  surface form, not just "the cone". All-but-the-top assumes the top PCs are junk,
  and here they aren't entirely.
- **rownorm breaks the translation sanity check.** GPT-2's L0 is `wte + wpe[1]` and the
  unembed is `wte`. Centering first removes the constant `wpe[1]` exactly, so L0 vs
  unembed overlap is 0.999. Normalizing rows first means a shared offset no longer
  cancels, and it drops to 0.905. Anywhere a layer is "the previous one plus a
  constant", rownorm will report change that isn't there.

Flipbook check (`gpt2_drop2`, docs/results/gpt2_drop2_L12.png): **the hollow ring at
L12 post-LN fills in**, as predicted in OPEN_QUESTIONS. With the two top PCs gone it's a
solid cloud with surface-form regions (space_lower on one side, space_cap on the other,
digits and punctuation as islands). So the ring was one or two huge shared directions
putting every token on a shell, and the token-level structure was underneath it.

Takeaway: keep center-then-unit-norm as the default, and check any surprising frame
against rownorm. If the surprise survives both, it's the model.


## 10. Q10 + Q3: does k matter? (2026-09-24)

`token-drift ksweep` re-runs the kNN metrics at k = 5, 10, 30, 100 on the same 10k
subsample as `metrics.json` (k=10 reproduces it exactly, which is checked in a test).
Figure: `docs/results/q10_ksweep.png`. Chance Jaccard for two random k-sets is
0.0003 (k=5) to 0.005 (k=100), so none of this is baseline creep.

- **Trained models: the curves barely move with k.** Pythia consecutive overlap stays
  within 0.05 of itself at every transition across a 20x range of k (L2 -> L3: 0.58 /
  0.56 / 0.56 / 0.58). GPT-2's plateau is 0.77 +- 0.02 at every k, and the L12 LN
  cliff is 0.09 to 0.11. The one exception is GPT-2's first step, L0 -> L1 (0.46 at
  k=5, 0.38 at k=100), the same direction as the control below: block 1 keeps
  the closest neighbours better than the loose ones. So "k=10 is arbitrary" doesn't matter: whatever the model
  does to a token's 5 closest neighbours, it does the same to its 100 closest.
- **The control is the one that depends on k.** random_init's consecutive overlap
  goes *up* with k at every block (L0 -> L1: 0.05 -> 0.12, L5 -> L6: 0.47 -> 0.56).
  Reading: in a random net the very nearest neighbours are close to arbitrary
  (random embeddings in 512-d are almost equidistant), so they get reshuffled, but a
  looser neighbourhood survives better. In trained models the nearest neighbours are
  as stable as the looser ones, meaning the fine-grained neighbourhoods are real
  structure, not noise. That's a nice extra difference between "learned" and "architecture".
- **Purity falls with k, as it should.** Bigger neighbourhoods reach into the
  neighbouring category. Layer ordering is preserved; Pythia's L6 pre-LN drops most
  (0.62 -> 0.47), consistent with the big shared direction from section 8. The
  shuffled baseline doesn't move (0.23 / 0.25).
- **Q3: candidate (c), "k=10 is too local", is ruled out.** Pythia embed-vs-unembed
  beats last-hidden(post-LN)-vs-unembed at every k: 0.24 vs 0.14 (k=5), 0.22 vs 0.13,
  0.21 vs 0.14, 0.20 vs 0.15 (k=100). The gap does shrink a bit (0.10 -> 0.05) but
  never flips, and CKA (no k at all) already agreed. With (a) gone as well (section 8),
  (b) is the last candidate standing: the embed and unembed were trained together and
  ended up correlated, and the context-free last state is a different object.
  Proper test needs v1: if the corpus-averaged last state moves toward the unembed,
  that's (b).

Takeaway: k=10 stays. The one number that depends on k is the control's overlap
level, and that makes "trained minus control" a bit smaller at large k.

## 11. v1 milestone A: corpus-averaged vocab (2026-10-03)

v1 swaps "the token alone after BOS" for "the token as the model usually sees it": run
pile-10k (15.1M tokens, windows of 2048) through the model and, per layer, average each
token's residual over all its occurrences (unit-norm each occurrence first, `unit_mean`).
Tokens seen fewer than 20 times are left out (39,887 of ~50k remain). Runs:
`pythia70m_corpus` (main), `_shuf` (tokens shuffled inside each window: same counts, no
word order), `_rawmean` (same extract, plain average), `random_init_corpus`, and the two
`v0v1_*` comparisons (v0 vs v1 on the same 10k tokens).

Headline numbers (the README's v1 summary; per-layer values in each run's `metrics.json`):

| | L0 | L1-L5 | L6 pre / post-LN | unembed |
|---|---|---|---|---|
| purity, corpus | 0.73 | 0.71-0.85 (peak L3) | 0.69 / 0.69 | 0.89 |
| purity, shuffled | 0.73 | 0.75-0.82 | 0.70 / 0.70 | 0.89 |
| kNN vs L0, corpus | 1 | 0.42 (L1) -> 0.18 (L5) | 0.20 / 0.20 | 0.25 |
| self-sim adjusted, corpus | 0.93 | 0.62 / 0.54 / **0.42 / 0.40 / 0.36** | 0.03 / 0.01 | |
| self-sim adjusted, shuffled | 0.93 | 0.62 / 0.54 / **0.35 / 0.31 / 0.20** | 0.01 / 0.01 | |
| anisotropy (raw), corpus | 0.00 | 0.32-0.46 (L4 0.42, L5 0.45) | 0.95 / 0.97 | 0.92 |
| anisotropy (raw), shuffled | 0.00 | 0.35-0.74 (L4 **0.62**, L5 **0.74**) | 0.97 / 0.98 | 0.92 |
| anisotropy (raw), random init | 0.00 | 0.09 -> 0.23 | 0.26 / 0.26 | 0.00 |

So the shuffled control tracks the real corpus on purity and kNN, and only separates on
self-similarity from L3 on and on anisotropy at L4-L5 (bold). Random-init anisotropy is
about half of v0's (0.19 -> 0.43, section 3) but still grows with depth, so v0's
shared-BOS mechanism explains part of the control's cone, not all of it.

Heads-up on reading the kNN numbers: they're Jaccard, not "share of neighbours". At k=10,
Jaccard 0.51 ~ 6.8 of 10 neighbours shared, 0.42 ~ 6, 0.29 ~ 4.5, 0.18 ~ 3, 0.13 ~ 2.3,
0.06 ~ 1.

### 11.1 Q6: does current-token information fade with depth once there's context?

**Narrowed, not answered.** Neighbourhoods inherited from the embedding fade with depth in
context just as they did context-free (~6 of 10 kept at L1, ~3 at L5), and real word order
erases slightly more than a shuffled corpus does. But this is neighbourhood persistence,
not token recoverability, so Q6 is narrowed rather than answered.

`knn_vs_first` (overlap with the layer-0 neighbours), from `v0v1_pythia70m__pythia70m_corpus/v0v1.json`
and each run's `metrics.json`:

```
               L0    L1    L2    L3    L4    L5   L6pre L6post unemb
v0 trained    1.00  0.37  0.34  0.33  0.28  0.21  0.13  0.21  0.25
v1 trained    1.00  0.42  0.36  0.29  0.23  0.18  0.20  0.20  0.25
v1 shuffled   1.00  0.44  0.38  0.35  0.28  0.21  0.21  0.21  0.25
```

- Same shape in v0 and v1, so v0's fade wasn't an artefact of having no context.
- v1 starts above v0 (L1-L2) and ends below it (L3-L5). The shuffled run holds on to more
  of the embedding neighbourhood from L3 on (0.35 vs 0.29 at L3), so it's word order, not
  just "having neighbours", that erases a bit more of the token. Small effect, but in
  Voita's direction.
- What this does *not* measure: a vector can get entirely new neighbours and still let a
  linear probe read off the token perfectly. Voita measured recoverability (mutual
  information with the token id); we measured neighbourhood persistence. Cheap next test:
  is the nearest L0 embedding to a token's layer-L vector its own (kNN-to-own-embedding)?

### 11.2 Q7: does the frequency effect appear once there's context?

**Verdict (2026-10-03): not reproduced, and the answer depends on the metric.**

- Local (kNN, per token): rare tokens' neighbourhoods change *more* per layer, the
  opposite direction from Voita. It's there context-free too; merge-rank binning hid it.
- Global (CKA inside each bin, our stand-in for Voita's PWCCA): mostly no effect. Where
  there is one it points Voita's way (frequent change more), at the edges, strongest in
  v0's block 6.
- Context doesn't create the effect, on either metric. Voita's effect is per occurrence;
  averaging over the corpus may wash it out (Q14) -> milestone B.

First draft of this section said "backwards Voita". It was the local metric only; the
global check below is what changed the verdict. Both metrics can be true at once: a
bin's cloud keeps its overall shape while the points inside swap neighbours.

**Local:** neighbourhood change (1 - Jaccard), most frequent -> rarest of 5 equal-count
corpus-count bins, same 10k tokens (`v0v1_*/v0v1.json`, `knn_change_by_freq`):

```
              v0 (no context)    v1 (context)     random-init v1
L0 > L1        0.57 -> 0.66      0.51 -> 0.61     0.94 -> 0.94
L1 > L2        0.44 -> 0.53      0.41 -> 0.53     0.80 -> 0.81
L2 > L3        0.37 -> 0.44      0.43 -> 0.52     0.71 -> 0.71
L3 > L4        0.45 -> 0.53      0.56 -> 0.64     0.64 -> 0.63
L4 > L5        0.58 -> 0.65      0.60 -> 0.66     0.57 -> 0.58
L5 > L6pre     0.73 -> 0.77      0.57 -> 0.62     0.53 -> 0.53
```

v0 on the same 10k tokens binned by *merge rank* instead (one-off CPU check, not saved by
the pipeline): frequent merges -> rarest merges 0.61 -> 0.65 (L0>L1), 0.48 -> 0.52 (L1>L2);
gap ~0.04 everywhere. So merge rank hides most of the gradient. Hunch (untested) for why
rare tokens change more: fewer gradient updates -> less settled embedding neighbourhoods,
which get reshuffled (cf. section 10).

**Global:** 1 - linear CKA computed inside each bin, same tokens and bins
(`v0v1_*/v0v1.json`, `cka_change_by_freq`). This is the one comparable to Voita, who
measures a group's change with PWCCA:

```
              v0 (no context)            v1 (context)               random-init v0 / v1
L0 > L1       0.35 0.32 0.32 0.30 0.31   0.30 0.29 0.28 0.27 0.27   0.32 / 0.34 flat
L1 > L2       0.09 0.10 0.10 0.11 0.11   0.08 0.10 0.10 0.11 0.11   0.15 / 0.16 flat
L2 > L3       0.06 0.07 0.06 0.06 0.07   0.11 0.11 0.11 0.11 0.11   0.08 / 0.09 flat
L3 > L4       0.10 0.10 0.09 0.10 0.10   0.22 0.25 0.24 0.25 0.25   0.06 / 0.07 flat
L4 > L5       0.22 0.23 0.23 0.24 0.24   0.28 0.30 0.31 0.31 0.30   0.04 / 0.05 flat
L5 > L6pre    0.64 0.58 0.55 0.51 0.51   0.25 0.27 0.27 0.29 0.28   0.03 / 0.04 flat
```

- Most rows are flat to within 0.03. The L0>L1 rows lean Voita's way (frequent change
  more) in both trained runs, and L1>L2 leans slightly the other way.
- Surprise: the biggest frequency gradient on either metric is v0's block 6, 0.64 -> 0.51,
  frequent tokens changing most. That's the same block as the L6 stretch (Q17), and it's
  gone in v1. Not chased; noted for Q17.
- Random init is flat on both metrics, so the gradients are learned, not something
  built into the geometry.
- Caveats: CKA is not PWCCA (both compare whole point clouds, but PWCCA weights each
  matched direction by how much of the representation it accounts for; CKA doesn't
  match directions at all). And our bins are Pile-10k counts, not
  counts from Voita's training data.

### 11.3 Q13 (surprise): the final LayerNorm barely moves v1 neighbourhoods

**Verdict (2026-10-03): answered. It's section 8 again, not context.** The LN didn't do
less in v1. Our `unit_mean` averaging had already done its job (wiping out per-token
length differences) before we looked at the pre-LN frame.

Block 6 makes row lengths vary a lot along one shared direction (section 8). Center-first
normalization turns that length spread into direction differences, which scrambles the
pre-LN neighbourhoods, so the LN step *looks* big. `unit_mean` unit-norms every occurrence
before averaging, so the spread never reaches the mean vectors. Same 10k tokens, k=10
(`scripts/q13_ln_step.py`):

```
                    row-norm spread (std/mean)   L6pre > L6post overlap   L5 > L6post
                    L5     L6pre   L6post        center 1st  unit-norm 1st  (either)
v0 (token alone)    0.07   0.19    0.05          0.30        0.81           0.44 / 0.45
v1 raw_mean         0.13   0.18    0.05          0.38        0.76           0.38 / 0.38
v1 unit_mean        0.11   0.02    0.01          0.82        0.83           0.39 / 0.39
```

- Unit-norm first takes v0 and raw_mean to 0.81 / 0.76, the same as unit_mean. No context
  needed: v0 has no context at all.
- The step that skips the pre-LN frame (L5 > L6post) doesn't care about the order in any
  variant. Only the pre-LN frame depends on how we look at it.
- Same story for the L6pre point in `knn_vs_first`: 0.12 (v0) and 0.14 (raw_mean) with
  center-first, 0.20 in v0 unit-norm-first (`pythia70m_rownorm`) and in unit_mean, flat
  with L5 and L6post (both 0.20). The earlier draft called unit_mean's 0.20 a "bump"; it's the other way
  round. unit_mean is the clean frame and the 0.12/0.14 *dip* is the artefact.
- Q13's hunch (a few huge-norm occurrences dominate the raw mean) isn't needed to explain
  any of this. It might still be true; checking needs per-occurrence norms -> milestone B.

### 11.4 Q15 (surprise): v0 and v1 neighbourhoods drift apart with depth

**Verdict (2026-10-03): answered.** v0 (token alone after BOS) and v1 (corpus-averaged)
really do drift apart with depth, but the trained model stays above the random-init
control all the way, about +0.1 at the top. How much v0's deep layers tell you depends
on the token: reasonably faithful for tokens that stand on their own (digits,
punctuation, whole words), close to meaningless for word fragments.

Cross overlap = Jaccard of a token's 10 neighbours in v0 vs v1, same frame, same 10k
tokens (`scripts/q15_cross.py`). The first draft used the pipeline's center-first
frames, which put the gap at L6pre at +0.03. That was the 11.3 normalization artefact:

```
                        L0    L1    L2    L3    L4    L5   L6pre L6post
trained, center 1st    1.00  0.51  0.42  0.32  0.23  0.20  0.13  0.21
trained, unit-norm 1st 1.00  0.53  0.43  0.33  0.24  0.20  0.22  0.22
random-init (either)   1.00  0.29  0.20  0.16  0.13  0.11  0.10  0.10
trained - control       -   +.24  +.23  +.17  +.11  +.09  +.12  +.12
```

The gap shrinks through the middle, then holds at about +0.1 from L4 on. The unembed
frame (1.00 in both) is the same matrix in v0 and v1, so it says nothing.

By category, trained, unit-norm first, L1 -> L6post (random-init is 0.28-0.31 -> 0.09-0.12
for every category, and flat across frequency bins, so the ordering below is learned):

```
digit      0.71 -> 0.32    punct      0.64 -> 0.28    byte   0.60 -> 0.30   (n=77)
space_cap  0.58 -> 0.26    space_low  0.57 -> 0.23    cap    0.51 -> 0.22
mixed      0.47 -> 0.15    lower      0.37 -> 0.14
```

- `lower` (no leading space, all lowercase) is mostly word fragments: `ing`, `tion`,
  `ably`. They're worst, and by the top layers barely above random-init (0.14 vs 0.10).
  Hunch, untested: "[BOS] ing" is an input the model essentially never sees (a suffix
  doesn't start a document), so v0 asks it about something off-distribution. Test: their
  cross overlap vs how often each one occurs right after a document boundary.
- Frequency matters much less than category: the five corpus-count bins spread only
  ~0.05 (frequent keep a bit more, 0.57 -> 0.24 vs 0.50 -> 0.19 for the rarest).
- Random init is not a usable baseline for `knn_vs_first`: it drops to 0.06 (~1 of 10)
  at L1 in both v0 and v1, because random embeddings have near-arbitrary nearest
  neighbours (same reason as section 10's "control depends on k"). That's why the
  trained-minus-control comparison above uses cross overlap.

### 11.5 Q3 (b): why the embedding is closer to the unembed than the last state is

**Verdict (2026-10-03): answered, with one caveat.** The last hidden state and the unembed
encode two different similarities. The last state groups tokens by *what follows them*
(right context); the unembed groups them by *what comes before them* (left context),
because its row for `t` is used when `t` is the thing being predicted. Those two
similarities barely overlap, so the two matrices disagree. The embedding carries a bit of
both, so it ends up closer to the unembed. The naive "the last state should look like the
unembed" mixes up "predicting `t`" with "being `t`".

**Planned test (spec): null, and that's the right null.** Context doesn't move the last
state toward the unembed: `knn_vs_last` at L6post is 0.15 in v0, v1 unit_mean and
raw_mean (shuffled 0.16), CKA 0.40 / 0.40 / 0.39 / 0.44. Under (b) that's expected:
averaging over real contexts gives "what typically follows `t`", which still isn't "`t`
as a target". (L6pre jumps between 0.10 and 0.16 across runs: the 11.3 normalization
artefact again.)

**Sharper test: model-free bigram vectors.** From the Pile-10k stream, a successor
vector (which tokens follow `t`) and a predecessor vector (which tokens come before `t`)
per token, PPMI-weighted and SVD'd to 256-d, then k=10 kNN overlap with model frames on
the metrics tokens (`scripts/q3b_bigrams.py`, takes the SVD dim as an argument). Chance
~0.0005. Most frequent fifth of tokens (bigram vectors are noisy for rare ones), and all:

```
                     most frequent bin            all tokens
                     successors  predecessors    successors  predecessors
embed (L0)             0.10        0.08            0.06        0.05
v0 last (L6post)       0.10        0.06            0.06        0.04
v1 last (L6post)       0.16        0.08            0.09        0.05
unembed                0.09        0.12            0.05        0.07
successors vs predecessors: 0.04
```

- The crossing (last state -> successors, unembed -> predecessors) holds at SVD dim 100,
  256 and 500 (numbers move by <= 0.01) and in every frequency bin with enough signal; it
  fades toward the rarest bin as everything goes to ~0.02.
- Context matters on the successor side: v1's last state matches successors much better
  than v0's (0.16 vs 0.10). Averaging over real contexts gives the last state the "what
  typically follows" meaning that the token-alone version only half has.
- Caveats: absolute overlaps are small (<= 0.16, though ~300x chance); bigrams are a
  one-token context, so they cover only a slice of the neighbourhood structure. And this
  explains why the last state is *far* from the unembed, not fully why the embed is as
  *close* as it is (0.22 at k=10). "The embed carries both similarities" is a reading,
  not a test. The other half of (b) (untied matrices getting correlated through
  training) is untested.

## 12. v1 milestone B: do virtues sit between their vices? (2026-10-04)

Q16, Andrey's hunch from Aristotle's doctrine of the mean: a virtue word's point lands between
its two vices' points (courage between cowardice and rashness). Milestone B tests it on
per-occurrence vectors instead of A's whole-Pile averages.

Setup (spec: `docs/superpowers/specs/2026-10-03-v1-milestone-b-design.md`):

- Occurrences come from 10 public-domain moral-philosophy books (Chase's Aristotle, Aquinas's
  Summa I-II and II-II virtue treatises, Republic, Mill, Kant, Seneca, Epictetus, Hume, Smith;
  1.98M tokens) and 3 shards of the deduplicated Pile (345M tokens), kept apart as two
  *groups*.
- For each hit of a word, one window ending at it (up to 2048 tokens of its own document);
  the vector is the residual at the word's last token piece. At most 1000 occurrences per
  word per group; a word needs >= 20 in a group to count there.
- A word's point = `unit_mean` of its occurrences, per group and frame. 41 triples
  (deficiency / virtue / excess) in two sets: *classical* (Chase + Aquinas wording) and
  *everyday* (modern words for the same concepts). 252,855 windows in all.
- Runs: `pythia70m_probe`, `random_init_probe` (same windows, random weights). Checks:
  `scripts/q16_*.py`, outputs saved in `runs/q16_checks.txt`.

How to read the numbers:

- *t, d*: where the virtue projects on the vice-vice line (0 and 1 = the vices) and how far
  off the line it sits, both in vice-vice lengths. A perfect line has d = 0; in 512 dims a
  random triangle is near-equilateral, d ~ 0.87.
- *Null percentile*: share of 20 frequency-matched null words that sit at least as close to
  the vice-vice segment as the virtue. *Beats null* = closer than all 20.
- *Best of three*: the virtue's null percentile is strictly lower than either vice's would be
  in the middle (the role swap). If the middle word didn't matter, ~1/3 of triples pass.

Usable triples: classical 11 in books / 10 in the Pile, everyday 0 in books / 18 in the Pile.
6 of the 11 classical book triples are left out of every verdict below (12.4).

### 12.1 Q16 headline: is the virtue the middle one?

**Verdict (2026-10-04): not supported as stated. A weak lean in classical words in the Pile, nothing
in everyday words, too little book data to say.**

Best of three per frame, over the triples free of the shared-piece artefact:

```
                        L0  L1  L2  L3  L4  L5  L6pre L6post
classical Pile  (10) trained   2   5   4   4   6   6   3    4
                     random    1   2   1   2   1   1   2    2
everyday Pile   (18) trained   3   7   5   9   4   3   3    3
                     random   11   6   9   5   7   7   7    7
classical books  (5) trained   1   1   2   3   3   2   0    0
                     random    0   1   1   0   1   1   1    1
```

- Classical/Pile is the only place the trained model clearly beats its control (6/10 vs
  1/10 at L4-L5). How surprising is 6/10? The binomial p (0.08) assumes independent triples,
  and they aren't (12.4). Against 20 random-weight draws at L0, 6/10 or more happened 2 times
  in 20: about the 90th percentile. Suggestive, not more.
- Everyday/Pile sits at chance in the trained model (3-9 of 18, chance 6).
- Books: 5 clean triples, 0-3 pass (random 0-1). Nothing to say yet.
- Equal counts don't change this (12.4): subsampling every word to 20 occurrences moves the
  trained counts by 1-2.

### 12.2 Do the three words lie on one line?

**Verdict (2026-10-04): no. They form near-random triangles; the trained model's are a bit flatter
than chance.**

Over the 33 clean triple x group cases:

```
                 median virtue d   virtue d below all 20 nulls   best-of-three winners' median d
trained L0           0.87                  9/33                          0.81
trained L4-L5        0.85-0.86             8-10/33                       0.71
random, any layer    0.85-0.89             0-1/33                        0.82-0.84
```

- Chance of beating all 20 nulls on d is 1 in 21, ~1.6 of 33. The trained model gets 8-10,
  already at L0 (so partly in the embeddings), random weights 0-1.
- But the median virtue is 0.85 vice-lengths off the line, the random-triangle value. Even
  the winners sit at d ~ 0.71: a squashed triangle, not a line.
- So "best of three" doesn't mean "between". Example, covetous / liberal / prodigal, books,
  L3: all three placements have t ~ 0.5, d ~ 0.85 (an almost perfect equilateral triangle);
  the virtue "wins" only because the null words near the two vices happen to sit closer to
  their lines.

### 12.3 Is there one "too little -> too much" direction across triples?

**Verdict (2026-10-04): no.**

Mean cosine between deficiency -> excess arrows of triple pairs that share no word and no
last token piece, against frequency-matched null arrows (`scripts/q16_direction.py`):

- Classical, Pile (44 pairs): -0.007 to -0.025 at every frame (p 0.6-0.85), random weights
  the same.
- Classical, books (10 pairs): one frame at p 0.05 (L3, +0.07) out of 8.
- Everyday, Pile (148 pairs): +0.02 to +0.05, p <= 0.05 at several frames. Random weights:
  +0.01 to +0.03, also p <= 0.05 at L4-L6. So it's the words' form, not learning; and cosine
  0.05 means the arrows are still nearly perpendicular.
- Shared word pieces weren't the cause (dropping piece-sharing pairs removed 2 of 150). Hunch,
  untested: the vices' regular suffixes (-ity, -ness, -less, -ful) give the arrows a shared
  morphological component that the frequency-matched nulls don't have (Q20).

### 12.4 Things about the method (each one would have changed a verdict)

1. **Words sharing a last token piece are stuck together at every layer.** The vector is read
   at the last piece, so temper|ance and intemper|ance start as the same point. We expected
   the early layers to pull them apart by L1-L3; they don't. Distance rank among all word
   pairs (0% = the closest pair; `scripts/q16_shared_piece.py`):

   ```
                                   L0   L1   L2    L3   L4   L5   L6pre L6post
   liberality / prodigality (bk)   0.0  0.0  0.0   0.2  0.0  0.0  0.1   0.1
   pusillanimity / magnanimity     0.0  0.0  0.0   0.0  0.1  0.0  0.0   0.0
   temperance / intemperance (Pile) 0.0 0.6  1.4  28.7  4.0  6.1  8.5   8.5
   related, unshared pairs (bk)   14.7 46.8 13.2  58.9  9.0  5.0  5.5   6.1
   ```

   So the 6 affected book triples (pleasure noun, both liberality/prodigality rows, all three
   honour rows) "beat the null" at every layer in both models and never win best of three.
   They're left out above. Random weights put related pairs at 50-65% (a typical pair); the
   trained model at 5-15%: learned meaning, cleanly. Why they never separate: Q18.
2. **One random-init draw is not a control.** At L0 a point is just an embedding row, so 20
   random-weight draws cost nothing (`scripts/q16_random_seeds.py`). Best of three: everyday
   2-11 of 18 (mean 5.3, sd 2.5), classical 0-7 of 10 (mean 2.8). Our run's seed drew 11 of 18,
   the top of the range: that was surprise "random init beats chance, p 0.01". Triples share
   words (pusillanimity in 3, stingy in 3, irritable and arrogant in 2), so their verdicts
   move together and binomial p-values are too optimistic.
3. **Unequal occurrence counts don't drive the result.** Every word subsampled to 20
   occurrences, 5 seeds (`scripts/q16_equal_n.py`): trained counts move by 1-2, random by 0-1.
   Random barely moves because a random model barely uses context: its median self-similarity
   is 0.98-1.00 at every layer (trained: 0.67-0.73 at L3-L5).
4. **Per-occurrence (t, d) live on a different scale from the points.** An occurrence is a
   unit vector; a point is an average of unit vectors, inside the sphere. Single occurrences
   get seg 0.9-1.7, far above the points' values, so the `q16_occ_*` histograms can't be read
   against the point-level numbers. The `occ` command's ranking *within* a word is fine.
5. **Surprise, not chased:** in the trained model, related-but-unshared pairs look like typical
   pairs at L1 and L3 (47%, 59%) and close at L2 and L4+ (5-15%). Odd layers out. Same
   alternation in the Pile (47% L1, 34% L3). Unexplained (OPEN_QUESTIONS Q19).

### 12.5 What's left

- More book text is what Q16 needs most: 5 clean classical triples is too few.
- ~~GPT-2~~ Ran 2026-10-04 (`runs/gpt2_probe`), parked as OPEN_QUESTIONS Q21: evidence too
  thin to write up. Two corrections to what we expected: GPT-2's BPE splits the shared-piece
  words *exactly* like Pythia's (temper|ance, prodig|ality, magn|anim|ity), so it's no test of
  12.4.1 by different pieces; and 5 of our 10 books are in PG-19 (Republic, Mill, Seneca,
  Epictetus, Hume), so Pythia may have read them (Chase, Summa, Kant, Smith aren't).
- A better random control: several seeds of the full run, or at least of L0 (cheap).
- B2 (polysemy, Q8): the occurrences are on disk, untouched.

## 13. v2: Pythia-70m training checkpoints (2026-10-04)

Every run so far looked at a finished model. EleutherAI saved Pythia during training (each
checkpoint is a Hub branch, a *revision*: `step0` .. `step143000`), so v2 runs the v0 pipeline
(token alone, `[BOS, tok]`) on 12 of them and asks *when* the v0 structure shows up.

Setup (spec: `docs/superpowers/specs/2026-10-04-v2-checkpoints-design.md`):

- Revisions: step 0, 1, 8, 64, 128, 256, 512, 1000, 4000, 16000, 64000, 143000. Log-spaced;
  128 and 256 were added after the first look, because everything happened between 64 and 1000.
- Run: `runs/pythia70m_ckpt` (`configs/pythia70m_ckpt.yaml`), one step folder per revision,
  then `timeline/` across them. Sanity checks: `scripts/v2_sanity.py`. ~45 min in all,
  flipbooks included.
- *Half-way step*: the first step where a curve has covered half the distance from its step-0
  value to its final value. Fixed before looking at any curve, so "when" isn't picked by eye.
- *With final*: kNN overlap (local: does each token have its final 10 neighbours yet?) and
  linear CKA (global: is the whole cloud in its final shape?) against step 143000's same frame.
- *Row drift*: ||W_t[i] - W_0[i]|| / ||W_0[i]||, median per merge-rank bin. The checkpoints are
  stored in float16, so a change below 2^-11 ~ 4.9e-4 can't even be recorded.
- Training (from Pythia's own `pythia-70m.yml`, checked 2026-10-04): Adam, lr 1e-3 cosine to
  1e-4, 1% warmup, **weight decay 0.1**.

### 13.1 Sanity checks

1. `step143000` vs `runs/pythia70m`: max difference 0.0 on every curve. Revision loading works.
2. `step0` vs `runs/random_init` (HF's init, seed 0): close, not identical. Max gaps: CKA 0.029,
   anisotropy 0.029, consecutive kNN overlap 0.012, purity 0.003. So `random_init` was a fair
   stand-in for "the model before training".
3. Never-trained tokens. The spec expected the lowest-drift embed rows to be tokens the Pile
   never has. Wrong expectation, right tokens: a row that never gets a gradient doesn't stay put,
   weight decay shrinks it toward 0, so its drift goes to 1 (trained rows end near 1.5, see
   13.5). Found instead by "same direction as init, length collapsed" (cos > 0.99, norm < 1%):
   **214 of 50,277 embed rows**, every one with cos 1.000 and the same shrink factor, 6.1e-4.
   - Pythia's schedule with weight decay alone predicts 3.9e-4 (e^-7.86 vs the observed
     e^-7.40, within 6% in the exponent). So these rows got exactly zero gradient for 143k steps.
   - Who they are: `<|padding|>`, one broken byte, and ~200 runs of spaces / newlines. Those
     whitespace tokens are **unreachable**: the tokenizer's added whitespace tokens (ids 50254+,
     e.g. 9 spaces = 50269) are matched before BPE runs, so BPE token 2286 (also 9 spaces) can
     never come out, in any context. 64 of the 214 sit in merge-rank bin 1, "most frequent":
     merge rank isn't frequency for them.

### 13.2 Nothing happens until step 64

Steps 0, 1 and 8 are identical in every curve. Warmup: the first updates are tiny, and at step 1
every bin's median drift is exactly 0 (below float16 resolution); at step 8 it's 3e-4 to 7e-4,
right at it. Real movement starts at 64, and most of the structure in this section arrives
between 128 and 1000, i.e. in the first 0.7% of training.

### 13.3 When does surface-form clustering appear?

**Verdict (2026-10-04): by step 512-1000, and the embedding is the last frame to get it, not
the first.**

kNN purity minus shuffled-label purity:

```
step          0    64   128   256   512  1000  4000 16000 64000 143000
L0 (embed)  0.00 0.00  0.01  0.03  0.10  0.30  0.48  0.48  0.47  0.49
L3          0.00 0.01  0.06  0.18  0.35  0.48  0.51  0.51  0.54  0.54
unembed     0.00 0.00  0.00  0.27  0.66  0.69  0.67  0.64  0.59  0.65
```

Half-way: L3 and unembed at 512, L0 at 1000.

- L3 starts first (128), the unembed starts later but jumps hardest (0 -> 0.66 between 128 and
  512, the step-256 and step-512 pages of `flipbook_unembed.gif` show the blob splitting into
  space-lower / space-cap / bare-lower / punctuation / digit islands), and the embedding trails
  (0.10 at 512).
- We expected the reverse: v0 calls L0 "surface form baked in" (FINDINGS 1). It is baked in at
  the end, but it's learned, and learned after the layers that read from it. Our guess: the
  unembed gets a gradient at every position for every token (13.5), the embed only from the
  tokens in the batch, and through every layer above it.
- After 4000 nothing moves much: L0 sits at 0.47-0.49 for the last 139k steps.

### 13.4 When does the stable middle block form?

**Verdict (2026-10-04): half-way by step 512, peak at 4000, then it partly comes apart.**

kNN overlap between consecutive layers:

```
step          0    64   128   256   512  1000  4000 16000 64000 143000
L1 -> L2    0.19 0.18  0.17  0.18  0.27  0.45  0.60  0.54  0.50  0.48
L2 -> L3    0.30 0.30  0.26  0.30  0.41  0.56  0.62  0.60  0.59  0.56
L3 -> L4    0.38 0.44  0.40  0.42  0.49  0.58  0.62  0.60  0.56  0.49
L4 -> L5    0.43 0.55  0.51  0.49  0.53  0.61  0.54  0.46  0.41  0.36
mean        0.33 0.37  0.33  0.35  0.42  0.55  0.59  0.55  0.51  0.47
```

- At init the block is already 0.2-0.4, rising with depth: each random layer perturbs the
  accumulated residual a bit less (FINDINGS 3).
- It firms up between 256 and 4000, then loosens from the top down. L4 -> L5 ends at 0.36, *below*
  its step-0 value. FINDINGS 3 already said the finished middle block is barely above the
  control; the timeline shows why: training builds a tighter block by step 4000 and the last
  97% of training mostly takes it apart again, L5 first.

### 13.5 The drift prediction

The prediction, written down before the run: unembed rows get a gradient at every position
(softmax pushes every wrong token down), embed rows only when their token is in the batch. So
the unembed should move early and evenly across bins, and the embed's rare bins should lag far
behind.

**Verdict (2026-10-04): half held. The unembed moves earlier and further; the embed's rare bins
lag only a little, and only early.**

Median drift, most frequent bin (1) / rarest bin (5):

```
step            8       64      128     512    4000   143000
embed     4e-4/3e-4  .023/.013 .090/.055 .42/.35 1.82/1.86 1.48/1.49
unembed   6e-4/7e-4  .038/.044 .110/.177 .85/.82 2.14/2.24 4.25/4.21
```

- Unembed ~2x the embed up to step 512: held.
- Rare embed rows lag, about 0.6x the frequent ones at steps 64-128; gone by 4000. Not "far
  behind". Our guess (untested): Adam. Plain SGD moves a weight by lr x gradient, so a row hit
  once in a while would crawl; Adam scales each weight's step by its own gradient history, so any
  row that gets *some* gradient takes roughly full-size steps. Only zero-gradient rows stay
  behind, and those are the 214 unreachable ones (13.1).
- In the unembed the rare bin moves *faster* early (0.177 vs 0.110 at 128).
- By the end every trained embed row has turned almost perpendicular to its init (median cos
  0.011) and grown a little (median norm ratio 1.12): drift ~1.5 is what "rotated away, same
  length" looks like (sqrt 2 = 1.41).
- The unembed's jump from 2.2 to 4.2 after step 64000 is one shared vector (13.6), not the
  rows moving on their own: with each matrix's mean row removed, its final drift is 1.5, like
  the embed's.

### 13.6 Surprises (not smoothed over)

1. **A shared direction shows up late in the top of the model.** Between 64000 and 143000:
   anisotropy L6 pre-LN 0.69 -> 0.93, post-LN 0.71 -> 0.96, unembed 0.04 -> 0.92. For the
   unembed it's a single mean vector: |mean row| goes 0.20 -> 2.48 while each row's own part
   (centered norm) shrinks 1.03 -> 0.72. Adding one vector to every unembed row adds the same
   number to every logit, which softmax ignores: the loss can't see this direction, so its
   gradient there is zero, and it still grew 12x. Q22.
2. **L6 pre-LN is scrambled only at the very end.** Its silhouette is -0.02 at 64000 and -0.27
   at 143000 (every other frame: -0.03 to +0.02). That's FINDINGS 8 / 11.3's
   "center-then-unit-norm scrambles L6 pre-LN", and now we know it's a late-training thing,
   arriving with surprise 1.
3. **An early cone, gone again.** L6 anisotropy jumps 0.46 -> 0.89 at step 64, falls back to
   0.42 by step 1000, then climbs to 0.93-0.96 late (surprise 1). Two separate cones.
4. **Local structure is still reshuffling at the end.** kNN overlap with final at step 64000 is
   only 0.38-0.48; CKA with final is 0.70-0.84. Half-way for kNN is "143000" for every frame,
   which only says the curve jumps at the last point. The global shape settles long before
   each token's 10 nearest neighbours do.
5. **Global arrival order:** CKA-with-final half-way at 1000 for the unembed, 4000 for L3,
   16000 for L0. The input end of the model settles last, again (cf. 13.3).
6. **Unreachable tokens (13.1)** were a tokenizer fact we'd never have found from the finished
   model alone: in it, those rows just look like small vectors.

### 13.7 What's left

- Q22 (why a shared unembed direction grows where the loss has no gradient), Q23 (is the
  embed-last order Adam, or the depth?), Q24 (does the top-down loosening of the middle block
  continue in bigger models?).
- A random-init sweep of seeds wasn't needed here (step 0 matches `random_init` to 0.03).
- Out of scope, one config change each: Pythia-160m / 410m checkpoints (does the order of
  arrival hold with size?), and more revisions between 64000 and 143000 to see when surprise 1
  starts.
