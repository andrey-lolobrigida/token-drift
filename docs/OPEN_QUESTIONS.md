# Open questions after v0

Ranked roughly by how cheap they are to answer over how much they'd change the story.
Each one says what we'd do. Findings these refer to are in FINDINGS.md.

## Cheap, do before v1

### Q1. Is the last-layer cliff the final block or the final LayerNorm?

GPT-2 L11 -> L12 kNN overlap is 0.08; Pythia L5 -> L6 is 0.42 but L6 has a 41%-variance
massive dimension and a mean norm 30x L5's. HF `hidden_states[-1]` is post final-LN
for both, so we're blind to the pre-LN residual.

Do: register a forward hook on the last block's output (before `ln_f` / `final_layer_norm`)
and save it as an extra frame "L(n) pre-LN". Then the flipbook shows block vs LN
separately. Also record the LN gain vector; if a few gains are huge, that's the
massive-activation dimension by name.

**Partly answered 2026-09-18** (FINDINGS section 4): the Pythia L6 mean offset *is*
the final-LN bias (cos 0.995).

**Answered 2026-09-24** (FINDINGS section 8). GPT-2: the cliff is the LN *gain* (block
12 keeps 0.40 of neighbours, the gain alone drops it to 0.09; coordinate 496 has gain
17.4 vs median 1.25). Pythia: no single cliff. Block 6 makes one shared direction vary
a lot in size between tokens (top PC 65% pre-LN), and the LN mostly squashes that back.
Most of the apparent reshuffle comes from our center-then-unit-norm order (see Q2).
The 09-18 "massive dimension = large LN gains" reading was wrong for Pythia.

### Q2. Does `drop_top_pcs` change the story?

**Answered 2026-09-24** (FINDINGS section 9). Ran `drop_top_pcs: 2` and a new
`row_norm_first: true` for both models. The stable middle, GPT-2's LN cliff and Q3 all
survive. Pythia's L5 -> L6 pre-LN dip doesn't (0.24 -> 0.46): that was the pipeline.
drop2 costs GPT-2 surface-form purity in the middle (0.77 -> 0.66); rownorm breaks the
exact removal of shared offsets (GPT-2 L0 vs unembed 0.999 -> 0.905). Default stays.

### Q3. Why is Pythia's input embedding closer to the unembed than the last hidden state is?

kNN overlap embed-vs-unembed 0.22, last-hidden-vs-unembed 0.13; CKA 0.46 vs 0.37.
Naive story says the last hidden state should be *the* thing aligned with the unembed.

Candidates: (a) it's the LN massive dimension again (**ruled out 2026-09-24**: the
pre-LN frame is even further from the unembed, overlap 0.08 vs 0.13); (b) untied
embed/unembed in Pythia still end up correlated through training, and the context-free
last state encodes "what follows this token in isolation", which is a different object
from "this token as a next-token target"; (c) k=10 is too local; check overlap at k=50
and the CKA, which already agrees though.

Do: a k sweep (10, 30, 100). Also re-check after the Q2 normalization-order variant.

**(c) ruled out 2026-09-24** (FINDINGS section 10): embed beats last hidden state at
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

Voita's headline. Untestable context-free. v1 gives each token a representation
averaged over real contexts; then "how well can you recover the token id from layer L"
becomes a real question (kNN-to-own-embedding, or a linear probe).

### Q7. Does the frequency effect appear once there's context?

Voita fig. 4b didn't reproduce in v0. If it appears in v1, the reading "it's contextual
updating" is confirmed. Frequency bins are already computed and saved
(`extract/freq_bins.npy`), so this is free once v1 activations exist.

### Q8. Polysemy trajectories

`" bank"` between `" river"` and `" money"` is the demo. Context-free there's nothing to
see (it's a single vector). v1 with per-context activations, or at least
per-sense-cluster averages, is the version where this means something.

### Q9. Ethayarajh's other measures

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

**Answered 2026-09-24** (FINDINGS section 10, `token-drift ksweep`). Trained curves move
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

## Surprises from the v1 milestone A runs (2026-09-26), to chase next session

Found during Task 8 (the real runs). Nothing here is written up in FINDINGS yet; that's
Task 9, and these should be looked at before (or while) doing it. Runs they refer to:
`pythia70m_corpus`, `_shuf`, `_rawmean`, `random_init_corpus`, the two `v0v1_*` dirs and
`runs/compare_v1.png`.

### Q13. Why does the final LayerNorm barely move v1 neighbourhoods?

> **Answered 2026-10-03, FINDINGS 11.3:** section 8's normalization-order effect. Unit-
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

> **Answered 2026-10-03, FINDINGS 11.4:** trained stays ~+0.1 above random-init to the top
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

Decided 2026-09-26: do it in milestone B, as a **mathematical test in 512-d, not a
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
0.84 with row norm. Not *why*. Effective rank (throwaway check 2026-10-03, v0 run, 10k
subsample) says the rest of the L6 cloud is ordinary: erank 13 with PC1, **245 without
it**, same as L1-L5 (160-245). So L6 = a normal middle-layer cloud plus one spike.

Do: hook block 6's attention output and MLP output separately; project each onto the L6
top PC; see which one carries the per-token variation. If it's the MLP, find the few
neurons whose output weights align with it. Then correlate the per-token amount with
merge rank / corpus count and with next-token entropy (is it a "how sure am I" knob?).

Side puzzle: the corpus *pilot* run has L6 pre-LN top PC 0.17, the full corpus run 0.43.
Same model, same code. Too few occurrences of the big-norm tokens in the pilot? Unchecked.

## Next-phase candidates (v1 / v2 from EXPERIMENT.md)

- v1: corpus-averaged activations over a few million tokens of the Pile (Pythia's
  training data), shuffled-corpus control, polysemy trajectories. Unlocks Q6 to Q9.
- v2: Pythia 160m / 410m for size, and Pythia training checkpoints for "when does the
  surface-form structure and the stable middle block form during training". The
  pipeline needs nothing new for either; just configs and time.
- ~~Pre-LN frames (Q1) should probably become a default part of extract before v1~~
  Done 2026-09-24: extract always emits the pre-LN frame now.
- Before v1, check any surprising frame against `row_norm_first` too (FINDINGS 9).
