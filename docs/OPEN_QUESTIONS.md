# Open questions after v0, v1 (milestones A and B) and v2

Ranked roughly by how cheap they are to answer over how much they'd change the story.
Each one says what we'd do. Findings these refer to are in FINDINGS.md.

## Cheap, from v0 (most answered, see each one)

### Q1. Is the last-layer cliff the final block or the final LayerNorm?

GPT-2 L11 -> L12 kNN overlap is 0.08; Pythia L5 -> L6 is 0.42 but L6 has a 41%-variance
massive dimension and a mean norm 30x L5's. HF `hidden_states[-1]` is post final-LN
for both, so we're blind to the pre-LN residual.

Do: register a forward hook on the last block's output (before `ln_f` / `final_layer_norm`)
and save it as an extra frame "L(n) pre-LN". Then the flipbook shows block vs LN
separately. Also record the LN gain vector; if a few gains are huge, that's the
massive-activation dimension by name.

**Partly answered** (FINDINGS section 4): the Pythia L6 mean offset *is*
the final-LN bias (cos 0.995).

**Answered** (FINDINGS section 8). GPT-2: the cliff is the LN *gain* (block
12 keeps 0.40 of neighbours, the gain alone drops it to 0.09; coordinate 496 has gain
17.4 vs median 1.25). Pythia: no single cliff. Block 6 makes one shared direction vary
a lot in size between tokens (top PC 65% pre-LN), and the LN mostly squashes that back.
Most of the apparent reshuffle comes from our center-then-unit-norm order (see Q2).
The earlier "massive dimension = large LN gains" reading was wrong for Pythia.

### Q2. Does `drop_top_pcs` change the story?

**Answered** (FINDINGS section 9). Ran `drop_top_pcs: 2` and a new
`row_norm_first: true` for both models. The stable middle, GPT-2's LN cliff and Q3 all
survive. Pythia's L5 -> L6 pre-LN dip doesn't (0.24 -> 0.46): that was the pipeline.
drop2 costs GPT-2 surface-form purity in the middle (0.77 -> 0.66); rownorm breaks the
exact removal of shared offsets (GPT-2 L0 vs unembed 0.999 -> 0.905). Default stays.

### Q3. Why is Pythia's input embedding closer to the unembed than the last hidden state is?

> **Answered in FINDINGS 11.5:** (b). Last state ~ "what follows t", unembed ~
> "what t follows" (model-free bigram test); the two barely overlap, the embed has a bit
> of both. Still open: why the embed is as close as it is to the unembed.

kNN overlap embed-vs-unembed 0.22, last-hidden-vs-unembed 0.13; CKA 0.46 vs 0.37.
Naive story says the last hidden state should be *the* thing aligned with the unembed.

Candidates: (a) it's the LN massive dimension again (**ruled out**: the
pre-LN frame is even further from the unembed, overlap 0.08 vs 0.13); (b) untied
embed/unembed in Pythia still end up correlated through training, and the context-free
last state encodes "what follows this token in isolation", which is a different object
from "this token as a next-token target"; (c) k=10 is too local; check overlap at k=50
and the CKA, which already agrees though.

Do: a k sweep (10, 30, 100). Also re-check after the Q2 normalization-order variant.

**(c) ruled out** (FINDINGS section 10): embed beats last hidden state at
every k from 5 to 100 (gap 0.10 -> 0.05, never flips). (b) is what's left; test it in
v1 by checking whether the corpus-averaged last state moves toward the unembed.

### Q4. How much of the random-init anisotropy is the BOS-attention artefact?

Suspected: position 1 attends to BOS and gets the same value vector every time.

Do: two variants of the control. (i) Zero the attention output at position 1 (MLP-only
path) and re-measure anisotropy. (ii) Feed `[tok]` alone with no BOS (single position,
attends only to itself). If anisotropy in the control drops to ~0, it's the artefact.
Same variants on the trained model would tell how much of *its* mid-layer anisotropy
is the same thing.

### Q5. Is the stable middle block learned or architectural?

Pythia mid-layer consecutive overlap 0.48 to 0.56 vs control 0.38 to 0.48. GPT-2's
0.68 to 0.78 has no control yet.

Do: `random_init` variant of GPT-2 (just a config). Then compare
trained-minus-control per transition rather than raw curves.

## Needs v1 (corpus-averaged activations)

### Q6. Does current-token information actually fade with depth in these models?

> **Narrowed in FINDINGS 11.1:** neighbourhoods inherited from the embedding fade
> with depth in context too (same shape as v0), and real word order erases a bit more
> than a shuffled corpus. But that's persistence, not recoverability. Next test:
> kNN-to-own-embedding (is a token's nearest L0 row its own?), then a linear probe.

Voita's headline. Untestable context-free. v1 gives each token a representation
averaged over real contexts; then "how well can you recover the token id from layer L"
becomes a real question (kNN-to-own-embedding, or a linear probe).

### Q7. Does the frequency effect appear once there's context?

> **Answered in FINDINGS 11.2:** not reproduced, metric-dependent. Local (kNN):
> rare tokens change more, context-free too (merge-rank binning hid it). Global (CKA per
> bin): mostly flat, faint Voita direction at L0>L1, biggest gradient in v0 block 6 (Q17).
> Voita's effect is per occurrence, so it gets one more go in milestone B (Q14).

Voita fig. 4b didn't reproduce in v0. If it appears in v1, the reading "it's contextual
updating" is confirmed. Frequency bins are already computed and saved
(`extract/freq_bins.npy`), so this is free once v1 activations exist.

### Q8. Polysemy trajectories

`" bank"` between `" river"` and `" money"` is the demo. Context-free there's nothing to
see (it's a single vector). v1 with per-context activations, or at least
per-sense-cluster averages, is the version where this means something.

### Q9. Ethayarajh's other measures

> **Milestone B note:** `runs/pythia70m_probe/metrics/q16.json` has raw (not
> baseline-adjusted) per-word self-sim (~560 word x group points): trained median 1.00 / 0.94 / 0.88 /
> 0.73 / 0.67 / 0.68 at L0-L5, then 0.98 / 0.99 at L6 pre / post-LN; random init 0.98-1.00
> everywhere. The L6 jump back up is the anisotropy cone, which A's adjusted numbers remove.
> Intra-sentence similarity and max explainable variance still need every token of a
> sentence, which B's windows don't keep. Still open.

> **Deferred to after milestone B (Andrey).** Intra-sentence similarity and max
> explainable variance need per-occurrence vectors. Self-similarity is already computed
> in milestone A but not written up: pythia70m_corpus adjusted self-sim
> 0.93 / 0.62 / 0.54 / 0.42 / 0.40 / 0.36 at L0-L5, then **0.03 at L6pre** and 0.01 post-LN
> (random-init 0.99 -> 0.72, no collapse). Same last-layer collapse Ethayarajh saw in
> GPT-2. Is that block 6 again (Q17)? Write up together with B.

Self-similarity across contexts, intra-sentence similarity, maximum explainable
variance. All need multiple contexts per token. His GPT-2 self-similarity drops to ~0
at layer 12, which is the same layer where our neighbourhoods collapse; worth
checking whether those are the same event.

## Method questions

### Q10. Is k=10 the right neighbourhood size?

Everything above uses k=10 on a 10k subsample. The curves are consistent across
kNN overlap, k-means ARI and CKA, which is reassuring, but the absolute overlap numbers
will move with k.

Do: k in {5, 10, 30, 100} on the existing normalized arrays. Cheap.

**Answered** (FINDINGS section 10, `token-drift ksweep`). Trained curves move
by <= 0.05 across k (except GPT-2 L0 -> L1, 0.46 -> 0.38); only the random-init control rises with k. k=10 stays.

### Q11. Should the surface-form categories be finer?

`mixed` (3.2k tokens in Pythia) is a grab bag: "3D", "USA", "Hello!", subword tails.
`lower` (11.6k) mixes real words with word-piece tails like "ing". Purity of 0.72
might hide a much higher number for a cleaner labelling.

Do: split `lower` into "starts a word after a space in a corpus" vs "continuation", and
give ALL-CAPS its own bucket. Needs a small corpus for the first one, or a heuristic
on the merge list.

### Q12. Is the viz subsample biasing the flipbook?

10k of 50k, sampled uniformly by token id, so rare tokens dominate (ids are merge
order). Purity and overlap are computed on the same subsample, so the numbers are
internally consistent, but the pictures over-represent rare tokens.

Do: a frequency-stratified subsample, or a run with `subsample: 50277` for metrics
only (kNN on 50k x 512 is fine; skip UMAP).

## Surprises from the v1 milestone A runs

Found during Task 8 (the real runs). Q13 and Q15 are answered in FINDINGS 11.3 / 11.4;
Q14, Q16 and Q17 are still open. Runs they refer to:
`pythia70m_corpus`, `_shuf`, `_rawmean`, `random_init_corpus`, the two `v0v1_*` dirs and
`runs/compare_v1.png`.

### Q13. Why does the final LayerNorm barely move v1 neighbourhoods?

> **Answered in FINDINGS 11.3:** section 8's normalization-order effect. Unit-
> norming each occurrence before averaging already removes block 6's length spread; v0
> unit-norm-first gives the same 0.81. The heavy-tail hunch below wasn't needed (still
> untested, milestone B).

kNN overlap L6 (pre-LN) -> L6 (post-LN): **0.82** in v1 (unit_mean), **0.29** in v0,
**0.38** in the raw_mean variant. Every other transition agrees between unit_mean and
raw_mean to within ~0.04, so it's specifically the pre-LN frame that depends on how we
average. Top-PC share at L6 pre-LN is 0.43 (unit_mean) vs 0.70 (raw_mean). Hunch: a few
huge-norm occurrences (or the massive dims from FINDINGS §8) dominate a raw average, and
unit-norming each occurrence first damps them. How we'd check: per-occurrence norm
distribution at L6 pre-LN for a handful of tokens; is it heavy-tailed? Which positions /
which dims carry the tail? This bears on FINDINGS §8's "Pythia = block 6 plus our
center-then-unit-norm order".

> **Milestone B hint:** `runs/pythia70m_probe/extract/occ.npy` keeps raw (not unit-normed)
> per-occurrence residuals, so the heavy-tail check is a few lines: L6pre norms per word.

### Q14. Why does the shuffled corpus look almost the same as the real one?

Shuffling tokens inside each window (same counts, no word order) leaves kNN overlap,
purity and ARI curves nearly unchanged. It only separates on adjusted self-sim from L3 on
(L5: 0.36 real vs 0.20 shuffled) and on anisotropy at L4-L5 (0.42/0.45 real vs
0.62/0.74 shuffled). Reading to test: averaging over hundreds of contexts washes out what
word order does to the *vocab-level* geometry, so the context signal lives in the spread
of occurrences, not in their mean. If true, milestone B (per-occurrence) is where context
should show up, and it's a caveat on every v1 curve. Cheap probe: compare the
per-token *variance* across occurrences (not the mean) between real and shuffled.

### Q15. What does the v0 -> v1 cross-overlap decay mean?

> **Answered in FINDINGS 11.4:** trained stays ~+0.1 above random-init to the top
> (the +0.03 at L6pre was the 11.3 normalization artefact). v0's deep layers are fairly
> faithful for standalone tokens (digits, punct, words) and near-useless for word
> fragments. Untested hunch: fragments after BOS are off-distribution.

Jaccard overlap of a token's 10 neighbours between v0 (token alone after BOS) and v1
(corpus-averaged): 0.996 at L0, 0.51 at L1, falling to **0.13** at L6 pre-LN, 1.0 at the
unembed. Random-init does it too (0.29 at L1 -> 0.10), so some of the decay is just
"any context changes the vector", not learned. Question: how much of the trained decay
is above that control, and is v0's deep-layer picture meaningful at all? Probably the
headline for Q6 once written up.

### Q16. Do virtue words land between their vices? (Andrey's hunch) -> milestone B

> **Answered in FINDINGS 12 (Pythia-70m): not supported as stated.** Weak lean in
> classical words in the Pile (virtue best of three in 6/10 triples at L4-L5 vs 1/10 for
> random weights; ~90th percentile of random draws), chance for everyday words, too few clean
> book triples to judge. The three words form near-random triangles, not lines (virtue
> d ~ 0.85 vice-lengths off the line; trained a bit flatter than nulls), and there's no shared
> deficiency -> excess direction across triples. GPT-2 next (Andrey). New surprises: Q18-Q20.

Decided: do it in milestone B, as a **mathematical test in 512-d, not a
picture** (UMAP distances between clusters don't mean much, see CLAUDE.md).

- Unit of test: triples (deficiency vice, virtue, excess vice), e.g. cowardice / courage /
  rashness. Most of Aristotle's words are multi-token in Pythia's tokenizer (cowardice,
  rashness, temperance, magnificence, flattery...), so use milestone B's last-piece
  convention; in milestone A only ~3 loose single-token triples survive (humble/proud/vain,
  fear/courage/reckless, lazy/ambitious/greedy).
- Per triple and layer: t = where the virtue projects on the vice->vice segment
  (0.5 = dead centre, outside [0, 1] = not between), and d = distance off that line
  relative to the segment length. Three points always lie in a plane, so a
  (t, d) plot is an *exact* 2-D picture of the triple, no projection distortion.
- Null: the same (t, d) for random triples of frequency-matched words. "Between" only
  counts if it beats that.
- Probe text (Nicomachean Ethics) vs Pile occurrences separately: in the Pile the ethical
  senses are rare (" vice" is 93% "vice president / versa"; in `group_vice.png` " vice"
  sits apart from the other vices at every layer).

### Q17. What in block 6 makes the stretched direction, and what does its size encode?

Q1 found *where*: block 6, not the final LN. Pre-LN top PC is 65%, nearly parallel to
the mean (|cos| 0.99), spread over many coordinates, and its per-token amount correlates
0.84 with row norm. Not *why*. Effective rank (throwaway check, v0 run, 10k
subsample) says the rest of the L6 cloud is ordinary: erank 13 with PC1, **245 without
it**, same as L1-L5 (160-245). So L6 = a normal middle-layer cloud plus one spike.

Do: hook block 6's attention output and MLP output separately; project each onto the L6
top PC; see which one carries the per-token variation. If it's the MLP, find the few
neurons whose output weights align with it. Then correlate the per-token amount with
merge rank / corpus count and with next-token entropy (is it a "how sure am I" knob?).

Side puzzle: the corpus *pilot* run has L6 pre-LN top PC 0.17, the full corpus run 0.43.
Same model, same code. Too few occurrences of the big-norm tokens in the pilot? Unchecked.

## Surprises from the v1 milestone B runs

Runs: `pythia70m_probe`, `random_init_probe`; checks in `scripts/q16_*.py`, outputs in
`runs/q16_checks.txt`.

### Q18. Why do words sharing a last token piece never come apart?

temper|ance / intemper|ance, liber|ality / prodig|ality, pusillanim|ity / magnanim|ity are
the closest pair of all at every layer in books (FINDINGS 12.4.1); we expected the early
layers to merge each word's pieces into its last position and separate them by L1-L3. Is it
the 70m model, the window context (books), or are the pairs just close in meaning? Do: the
same distances in GPT-2 (bigger model; NOT different pieces, its BPE splits these words
exactly like Pythia's, checked); and kNN-to-own-embedding of the *first* piece
at the last position (does the last position know which word it ends?).

### Q19. Why do related pairs look like strangers at L1 and L3 only?

Trained model: related-but-unshared triple pairs (cowardice / courage...) rank at 47% (L1)
and 59% (L3) among all pairs in books, 5-15% at L2 and L4+; Pile 47% / 34%. Odd layers out.
Do: is it the points' norms (unit_mean shrinks with low self-sim) rather than directions?
Rerun the ranking with cosine distance; then look for a massive dimension at L1 / L3.

### Q20. Is the everyday "shared direction" morphology?

Deficiency -> excess arrows of everyday triples align slightly (cosine +0.02-0.05) in both the
trained and the random model (FINDINGS 12.3). Hunch: the vices' regular suffixes. Do: rebuild
the null from pool words with the same suffix as each vice.

### Q21. GPT-2 probe: the virtue isn't between, the two vices group together? (parked)

Parked (Andrey): suggestive, not evidence. 10-18 triples that share words; a
difference of 2-3 triples is noise-sized. Run: `runs/gpt2_probe` (`configs/gpt2_probe.yaml`).
Raw geometry, no null words (`scripts/q16_raw_middle.py`, outputs in each run's
`metrics/q16_raw_middle.txt`); counts = triples, chance 1/3:

```
                              Pythia L1-L5   Pythia L6    GPT-2 L1-L11
everyday (18): virtue middle      2-4           3            2-4
everyday: vice-vice closest       6-9 (L0: 9)   8-9          8-12 (L0: 5, peak 12 at L6-L8)
classical/Pile (10): middle
  defic / virtue / excess       1-3 / 5-6 / 2-3  5 / 2 / 3   5-6 / 1-4 / 0-3 (L1: 5/5/0)
```

- Hunch: in everyday words a good/bad (valence) axis beats Aristotle's too-little/too-much
  one; the two bad words land together. Built by GPT-2's layers (L0 at chance), already in
  Pythia's embeddings.
- Pythia's middle layers are the only place the virtue sits in the middle (agrees with
  FINDINGS 12.1); its L6 switches to GPT-2's arrangement (deficiency next to the virtue).
- The null-based tally (q16.json) made GPT-2's "deficiency in the middle" look much stronger
  (7-9/10, 8-12/18); raw geometry halves it. Per-role null percentiles exaggerate role skews.
- Not a length artefact: unit-normed points give the same counts, and deficiency words'
  self-sim is >= the others'.
- Control: 20 random seeds at L0 only (`scripts/q16_random_seeds_l0.py`, rebuilds GPT-2's
  wte + wpe[pos] per occurrence): everyday best-of-three 5.4 +- 2.0 / 18, classical 4.1 +- 1.6 / 10.

Do (if ever): a sentiment direction (e.g. good-minus-bad word means) and how much of each
triple's spread lies along it; many more triples; per-triple look at the classical set.

## Surprises from the v2 checkpoint run

Run: `runs/pythia70m_ckpt` (12 Pythia-70m revisions, step0 .. step143000); sanity checks in
`scripts/v2_sanity.py`. Write-up: FINDINGS 13.

### Q22. Why does a shared unembed direction grow where the loss has no gradient?

Between step 64000 and 143000 the unembed's mean row grows 0.20 -> 2.48 while each row's own
part shrinks 1.03 -> 0.72 (anisotropy 0.04 -> 0.92). A vector added to every unembed row adds
the same logit to every token, which softmax ignores, so the summed gradient along it is exactly
zero; weight decay (0.1) should shrink it. L6 pre/post-LN grows a cone at the same time (0.69 ->
0.93-0.96), and L6 pre-LN's silhouette drops to -0.27. Hunch: Adam. It rescales each weight's
step by its own gradient history, so per-row steps no longer sum to zero across the vocab, and
the always-pushed-down rare rows could drift together along minus the mean hidden state.
Do: more revisions between 64000 and 143000 (when does it start? with the lr decay?); the cosine
between the unembed mean row and the mean L6 post-LN state; same check on Pythia-160m.

### Q23. Why does the embedding cluster by surface form last?

Purity minus shuffled, half-way: L3 and unembed at step 512, L0 at 1000 (0.10 at 512 vs 0.35 /
0.66). CKA-with-final half-way: unembed 1000, L3 4000, L0 16000. Is it the gradient path (the
embed only learns through every layer above it, and only from tokens in the batch), or does
the unembed's structure get copied down? Do: purity of L0 vs unembed per merge-rank bin over
steps 128-1000 (if the embed's rare bins lag more there, it's the batch; if all bins lag
together, it's the path); same on Pythia-160m.

### Q24. Does late training take the middle block apart in bigger models too?

Mean consecutive overlap L1 -> L5: 0.33 at init, 0.59 at step 4000, 0.47 at the end; L4 -> L5
ends at 0.36, below its init value (0.43). Is that a 70m thing (too few layers, the top one
specializes for output)? Do: the same timeline on Pythia-160m (12 layers): one config with
`model: EleutherAI/pythia-160m`.

## Next-phase candidates (v1 / v2 from EXPERIMENT.md)

- ~~v1 milestone A: corpus-averaged activations, shuffled-corpus control~~ Done
  (FINDINGS 11). Q3, Q7, Q13, Q15 answered, Q6 narrowed.
- ~~v1 milestone B: Q16~~ Done (FINDINGS 12). GPT-2 probe run done and parked
  (Q21). B2 (Q8 polysemy) can run on the existing occ.npy.
  Originally inherited: Q8 (polysemy), Q9 (Ethayarajh's other measures,
  incl. the L6pre self-sim collapse), Q14 (is context in the spread, not the mean?),
  Q16 (virtue between vices), Q7 per occurrence, and Q13's untested heavy-tail hunch.
- Cheap, no new milestone needed: Q6's kNN-to-own-embedding, Q17 (block 6 hooks),
  11.4's "fragments after BOS are off-distribution" hunch, Q4, Q5, Q11, Q12.
- ~~v2: Pythia training checkpoints for "when does the surface-form structure and the
  stable middle block form during training"~~ Done (FINDINGS 13): surface form
  half-way by step 512 (L3, unembed) / 1000 (L0); the middle block half-way by 512, peaks at
  4000 and then loosens. Raised Q22-Q24.
- v2 for size: Pythia 160m / 410m, finished models or their checkpoints. Just configs and time.
- ~~Pre-LN frames (Q1) should probably become a default part of extract before v1~~
  Done: extract always emits the pre-LN frame now.
- Before v1, check any surprising frame against `row_norm_first` too (FINDINGS 9).
