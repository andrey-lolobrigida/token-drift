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
| post-LN -> unembed | 0.14 / 0.14 / 0.14 | 0.05 / 0.10 / 0.07 |

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

### 11.y Q15 (surprise): v0 and v1 neighbourhoods drift apart with depth

TODO (write-up in progress). Noted so far:
- random-init is not a usable baseline for `knn_vs_first`: it drops to 0.06 (~1 of 10) at
  L1 in both v0 and v1, because random embeddings have near-arbitrary nearest neighbours
  (same reason as section 10's "control depends on k"). "Trained minus control" has to come
  from the cross overlap instead:

```
                    L0    L1    L2    L3    L4    L5   L6pre L6post unemb
cross, trained     1.00  0.51  0.42  0.32  0.23  0.20  0.13  0.21  1.00
cross, random-init 1.00  0.29  0.20  0.16  0.13  0.11  0.10  0.10  1.00
trained - control   -   +0.22 +0.22 +0.16 +0.10 +0.09 +0.03 +0.11   -
```
