# Glossary

Terms as they came up, in order. Append, don't reorganize.

## 2026-09-18

**kNN** — k-nearest-neighbors. For a point, the k other points closest to it (here
k=10, cosine distance after normalization). The most local view of geometry: who a
token's immediate neighbors are, ignoring where clusters sit relative to each other.
Came up in: `src/token_drift/metrics.py`.

**kNN purity** — for each token, the fraction of its k neighbors that share its
label, averaged over tokens. 1.0 = fully self-sorted by label. Only sees local
neighborhoods. Came up in: `knn_purity()` in `src/token_drift/metrics.py`, third
panel of `metrics.png`.

**surface-form category** — heuristic label from the token *string* (leading space,
casing, digit, punctuation, ...). Ten buckets, same at every layer since they don't
depend on the model. Came up in: `src/token_drift/labels.py`.

**shuffled-label baseline** — recompute a metric after randomly permuting the labels
across tokens; gives the value expected from no structure. Not 1/n_categories,
because buckets are unequal in size (0.23 for pythia). Read curves as distance above
this floor. Came up in: the dotted line on the purity panel.

**silhouette** — textbook cluster-quality score: how close a point is to its own
cluster vs the nearest other cluster. Wants compact blob-shaped clusters; reads ~0
here because the categories are interleaved ribbons in 512-d, even though local
neighborhoods are pure. Came up in: why the purity panel doesn't show it.

**kNN overlap** — for each token, Jaccard overlap between its k-neighbor set at layer
L and at layer L+1 (or vs L0), averaged. 1 = nobody moved. Unsupervised: no labels
involved. Came up in: `knn_consecutive` / `knn_vs_first` in `metrics.json`.

**k-means** — carve a point cloud into k groups: pick k centers, assign each point to
its nearest, move centers to their group means, repeat. Prefers round blobs.
`n_init=3` = three random restarts, keep the best. Here k=20. Came up in:
`kmeans_ari()` in `src/token_drift/metrics.py`.

**partition** — a split of the token set into disjoint groups. Cluster ids are
arbitrary names, so two partitions can't be compared by matching ids.

**Rand index / ARI** — agreement between two partitions, measured over *pairs* of
points (do both partitions put the pair together, or both apart?) so cluster naming
doesn't matter. ARI = Adjusted Rand Index, chance-corrected: 0 = random agreement,
1 = identical grouping. Came up in: fourth panel of `metrics.png`, "k-means ARI,
consecutive layers".

**cosine similarity** — cosine of the angle between two vectors: 1 = same direction,
0 = perpendicular, -1 = opposite. Ignores length. The distance every kNN metric here
uses (after unit-norm, dot product = cosine).

**anisotropy / isotropic** — iso = same, tropos = direction. An isotropic point cloud
looks the same in every direction (a ball around the origin); an anisotropic one has
a preferred direction, in the extreme a single cone. Measured here as mean cosine
between random token pairs on *raw* activations (Ethayarajh 2019). Pythia L6 = 0.96:
the mean vector's norm is 433 vs a typical row's 439, i.e. nearly all of every token
vector is one shared direction. Came up in: `anisotropy()` in
`src/token_drift/metrics.py`, fifth panel of `metrics.png`.

**centering** — subtract the mean vector from every row so the cloud sits around the
origin. Removes a shared *offset* so cosine measures the token-specific part. First
step of the normalize stage. Does not remove a shared high-*variance* direction;
that's what `drop_top_pcs` is for. Came up in: `src/token_drift/normalize.py`.

**massive-activation dimension** — a single coordinate (or direction) carrying a
huge share of the variance at one layer; Pythia's post-LN L6 has one at 41% of the
variance. Survives centering. Came up in: FINDINGS.md section 4.

**translation** — shifting every point by the same vector; the cloud's shape is
untouched. Centering undoes a translation exactly. Pythia's L6 anisotropy is one:
the shared offset is the final LayerNorm's bias (cosine 0.995 with it, 97% of each
vector's squared norm). Came up while asking whether mean cos ~0 after centering
means "the model just moved the cloud".

**LayerNorm (gain, bias)** — rescale a vector to zero mean / unit variance across its
coordinates, then multiply each coordinate by a learned *gain* and add a learned
*bias*. Bias = a translation of the whole cloud; gain = a per-axis stretch, which
centering does not remove. Came up in: `final_layer_norm` of pythia-70m, the source
of the L6 offset and (via a few large gains) the massive dimension.

**principal component (PC)** — the direction along which a centered cloud has the
most variance; the 2nd PC is the widest direction perpendicular to it, and so on
(from PCA / SVD). "Top-PC variance share" is the shape check that mean cosine can't
do: a centered cigar has mean cos 0 but top-PC share 0.44. Came up in: the
`drop_top_pcs` option in `normalize.py`, and the L6 = 39% number.

**BPE (byte-pair encoding)** — how the tokenizer builds its vocab: start from single
bytes, repeatedly merge the most frequent adjacent pair into a new token, record each
merge in order. Came up in: `vocab_freq_ranks()` in `src/token_drift/extract.py`.

**merge rank** — position in the BPE merge list of the merge that created a token.
Early = the pair was common in training text, so it's a corpus-free frequency proxy.
Coarse: a late-merged token can still be common. For both tokenizers here it equals
token-id order. Base tokens (single bytes, never merged) have rank -1.

**base / byte tokens** — the ~256 single-byte tokens plus a few unmergeable entries.
No merge rank, so they get their own bin 0 in the frequency panel. Mostly digits and
punctuation, the islands that barely move between layers.

**frequency bins** — merged tokens split into 5 equal-count quantile bins by merge
rank (1 = most frequent, 5 = rarest), plus bin 0 for base tokens. Saved at extract
as `freq_bins.npy`, so v1 can reuse them.

**neighborhood change** — 1 minus the Jaccard overlap of a token's k-neighbor set
across a transition. The kNN overlap panel upside down, per token, so it can be
grouped. Came up in: `knn_change_by_bin()`, sixth panel of `metrics.png`.

**Voita et al. 2019** — "The Bottom-up Evolution of Representations in the
Transformer". Fig. 4b: frequent tokens change more per layer in a trained LM,
attributed to contextual updating. Does not reproduce context-free (v0), which is
the argument for corpus-averaged v1 (Q7).

## 2026-09-24

**hook (forward pre-hook)** — a callback PyTorch runs whenever a given module executes.
A *pre*-hook fires just before, and sees the module's input. We put one on the final
LayerNorm to grab the residual HF never returns. Came up in: `extract_activations()`
in `src/token_drift/extract.py`.

**pre-LN / post-LN frame** — the last block's output before vs after the final
LayerNorm. HF's `hidden_states[-1]` is post-LN; the pre-LN one comes from the hook.
Splitting them tells "what the block did" apart from "what the LN did" (Q1).

**LN normalize step** — the first half of LayerNorm, before gain and bias: subtract
each row's *own* mean over its coordinates, divide by its own std. Every row comes out
the same length. Not the same as our `normalize` stage, which subtracts the mean
*across tokens* first and then unit-norms. Came up in: FINDINGS section 8.

**row-norm spread (coefficient of variation)** — std of the row lengths divided by
their mean. 8% = all tokens roughly the same length, 19% = lengths vary a lot. Pythia
jumps 8% -> 19% going into the pre-LN L6 frame.

**normalization order** — center-then-unit-norm (our pipeline) vs unit-norm-then-center.
Same thing when rows have similar lengths; very different when one shared direction
has a token-dependent size, because centering turns those size differences into
direction differences. Came up in: Pythia's pre-LN frame, Q2.

Correction to **massive-activation dimension** and **LayerNorm** above (09-18 entries):
for Pythia the big L6 direction is *not* made by large LN gains. It's already in the
pre-LN residual (65% top PC) and the gain barely changes it. For GPT-2 the gain *does*
make it: coordinate 496 has gain 17.4 vs median 1.25.

**all-but-the-top** — Mu & Viswanath (2018): center, then project out the top few
principal components, on the theory that they're shared "cone" directions and not
meaning. Our `drop_top_pcs`. Q2 found it's not free: in GPT-2's middle layers the top
two PCs carry some surface-form signal (purity 0.77 -> 0.66).

**robustness check / variant run** — re-running the same measurement with one choice
changed (here: the normalization) to see whether a result is about the model or about
the choice. A finding that survives every variant is about the model. Came up in: Q2,
`configs/*_drop2.yaml`, `configs/*_rownorm.yaml`.

**k sweep** — re-running the same kNN metrics at several neighbourhood sizes
(k = 5, 10, 30, 100) to see whether a result depends on the arbitrary choice of k.
If the curve *shape* holds across k, the finding isn't about k. Came up in: Q10,
`knn_sweep()` in `src/token_drift/metrics.py`, `token-drift ksweep`.

**chance overlap** — the Jaccard overlap two *random* k-neighbour sets would have
by luck: they share about k²/(n-1) tokens on average. It grows with k, so it's the
floor to compare against when k changes. At n=10k it's 0.0003 (k=5) to 0.005 (k=100),
i.e. negligible. Came up in: `chance_overlap` in `ksweep.json`.

**corpus-averaged activation** — v1's way of manufacturing a vocab matrix: run real
text through the model and, per layer, average each token's residual over all the
places it occurred. "What the model *usually* thinks this token is", vs v0's
"this token alone after BOS". Came up in: v1 design.

**running sum** — keep a per-token total (and a count) and add every new occurrence
into it, dividing at the end, so individual occurrences never have to be stored.
~0.9 GB for 9 frames x 50k tokens x 512 dims in float32. Came up in: v1 design.

**min-count threshold** — a token only enters v1 metrics if it occurred at least N
times; an average over 3 contexts mostly reflects which 3 sentences it landed in.
At N=20, 15M Pile tokens cover 40k of 50k vocab entries. Came up in: v1 corpus probe.

**Zipf's law** — word frequency falls off roughly as 1/rank: a few tokens are
everywhere, most are rare. Why the rare tail needs a lot more corpus to cover.

**the Pile / pile-10k** — the 800 GB, 22-source text mix Pythia was trained on (web,
PubMed, GitHub, law, books, ...). `NeelNanda/pile-10k` is its first 10k documents,
15.4M Pythia tokens. Came up in: v1 corpus probe.

**PG-19 / Books3** — two book subsets of the Pile: PG-19 = 28.6k public-domain
Project Gutenberg books from before 1919; Books3 = copyrighted books, now pulled from
most mirrors. The Nicomachean Ethics (Gutenberg #8438) is *not* in PG-19.

**background corpus vs probe corpus** — v1's two-corpus split. Background = a
representative Pile slice, averaged into a vocab-wide matrix (milestone A). Probe =
small targeted texts (e.g. the Ethics) where every occurrence of a few tracked words
is kept individually, tagged by source (milestone B, polysemy).

**per-occurrence storage** — saving one vector per occurrence (per layer) instead of
folding it into a running sum. ~9 KB per occurrence, so only for a tracked word list.

**multi-token word / last-piece convention** — words like " temperance" are split
into several tokens, so they have no vocab row. To track them per occurrence, take
the residual at the word's *last* piece: that's the first position where the model
has read the whole word. Came up in: virtue/vice tokenization check.

**self-similarity (Ethayarajh 2019)** — for one token, the average cosine between its
vectors across all the places it occurs. 1 = context never changes it; low = every
occurrence looks different. v1 gets it for free from the running sum of unit vectors:
||sum||² = n + (sum of all pairwise cosines). Came up in: v1 plan, extract corpus mode.

**attention sink** — a position the model dumps attention onto when it has nowhere
useful to look (position 0, every `<|endoftext|>`, sometimes a random ordinary token).
Its residual norm balloons (~120-170 vs ~12), which is why v1 skips early positions
and EOS, and averages unit vectors. Came up in: v1 plan, `min_context`.

**eligible token** — a token with corpus count >= `min_count` (20). Only these enter
the v1 metrics subsample. Came up in: v1 plan, metrics stage.

**cross overlap** — kNN overlap between two *runs* at the same frame, on the same tokens
(v0 frame f vs v1 frame f), instead of between two frames of one run. Came up in: v1
plan, `v0v1` command.

**TDD (test-driven development)** — write the test first, watch it fail, then write
the code that makes it pass. The plan's steps follow that loop. Came up in: v1 plan.

## 2026-09-25

**logits** — the raw next-token scores the model outputs: one number per vocab entry (50,304 for
Pythia) at every position, before softmax turns them into probabilities. Computing them is the
unembed matmul; we never use them in corpus mode, which is why extract now skips them.
Came up in: the final review of `extract_corpus_means` (logits were ~60% of CPU time).

**base_model** — in HF transformers, the model *without* its output head. `GPTNeoXForCausalLM`
= `base_model` (embeddings + blocks + final LN) + `embed_out` (the unembed). Calling
`model.base_model(...)` gives the same hidden states, minus the logits.
Came up in: the corpus-extract speed fix.

**adjusted self-similarity** — self-similarity minus the anisotropy baseline (mean cosine between
two random occurrences of *any* tokens). Raw self-sim can look huge just because everything
in a layer points the same way; subtracting the baseline asks "more alike than chance?".
Came up in: the v1 pilot run (L6: self-sim 0.97, baseline 0.94, adjusted 0.03).

**pilot run** — a small, cheap run of the full pipeline (here 1.5M of 15.4M tokens) done before
the expensive one, to catch crashes and check sanity numbers first.
Came up in: Task 8 of the v1 plan.

**kernel module (NVIDIA driver)** — the part of the GPU driver that lives inside the Linux kernel.
It's built for one exact kernel version, so after a kernel update the GPU disappears until the
matching module package is installed (or you boot the older kernel).
Came up in: `nvidia-smi` failing on kernel 7.0.0-34.

## 2026-09-26

**unit_mean vs raw_mean** — two ways to turn many occurrences of a token into one vector.
unit_mean unit-norms every occurrence first and then averages (each context gets one equal vote);
raw_mean averages the raw vectors, so a few huge-norm occurrences can dominate. Same extract, only
`normalize.source` differs. Came up in: Task 8 step 6, where they disagreed only at L6 (pre-LN).

**shuffled-corpus control** — same corpus, same token counts, but tokens shuffled inside each
window, so every token keeps its frequency but loses its real context. If a curve looks the same
on shuffled text, that curve isn't measuring anything word order does.
Came up in: `configs/pythia70m_corpus_shuf.yaml`.

**CUDA** — NVIDIA's API for running general computation on the GPU; `torch.cuda.is_available()`
is the "is the GPU usable" check. Came up in: resuming Task 8 after the driver fix.

## 2026-10-03

**Jaccard overlap** — size of the intersection of two sets divided by the size of their
union. For two k=10 neighbour sets sharing s tokens: s / (20 - s). Runs lower than "share of
neighbours" (s/10): 5 of 10 shared is Jaccard 0.33, not 0.5. Every kNN overlap number in
this repo is Jaccard. Came up while writing FINDINGS §11.1 (Q6).

**linear probe** — the simplest possible readout trained on top of frozen activations: one
weight matrix (plus softmax), fit to predict some property, e.g. "which token is this?".
High accuracy = the information is there *and* readable along some direction. Kept linear
on purpose: a powerful probe could compute the answer itself, and then you'd be measuring
the probe, not the model. Came up in Q6: it measures recoverability, which kNN overlap doesn't.
Catch: a probe shows the info is *readable*, not that the model *uses* it (that's a causal
question, e.g. ablation). Standard read: Belinkov (2022), "Probing Classifiers: Promises,
Shortcomings, and Advances", Computational Linguistics 48(1), arXiv 2102.12452.

**neighbourhood persistence vs recoverability** — two different meanings of "the token is
still there at layer L". Persistence: the vector still has the same nearest neighbours as at
L0 (what `knn_vs_first` measures). Recoverability: you can still read off which token it is
(what a probe or Voita's mutual information measures). A vector can lose the first and keep
the second. Came up in FINDINGS §11.1.

**kNN-to-own-embedding** — a cheap recoverability test: take a token's layer-L vector, find
the closest row of the L0 embedding matrix; is it the token's own? Works without training
because the residual stream is one shared space across layers. Not computed yet. Came up in Q6.

**effective rank** — how many directions a centered cloud really uses, as one number. Take
the PC variance shares p_i and compute exp(entropy) = exp(-sum p_i log p_i): a round ball in
512-d scores ~512, a perfect cigar scores 1. Top-PC share only looks at the *first* axis, so
it can't tell "one mild stretch" from "a pancake spread over 20 directions"; effective rank
can. Not computed yet. Came up reading the mid-layer top-PC values (0.07 to 0.12).

**residual stream** — the 512-d vector each position carries through the model. Every block
*reads* from it and *adds* its output back (x <- x + attn(x), then x <- x + mlp(x)); nothing
overwrites it. That's why a 64-d bottleneck inside a block can't cap the rank of the stream:
the old 512-d vector is still there underneath. Came up asking whether the first projection
caps effective rank.

**attention head (W_Q, W_K, W_V, W_O)** — one of the parallel attention units in a block.
Pythia-70m has 8 per block, each working in 64 dims: W_Q/W_K/W_V squeeze the 512-d stream
down to 64 (query, key, value), W_O maps the value back up to 512 to add to the stream.
8 x 64 = 512, so the heads together are as wide as the stream. Came up in the same question.

**head subspace / principal angles** — a head's output is W_O_h @ z with W_O_h a 512 x 64 slice,
so it can only land in the 64-d span of those columns: a tilted 64-d "sheet" in 512-d, not
coordinates 1-64. To ask how much two such sheets overlap, take orthonormal bases A, B and
the singular values of A^T B: those are the cosines of the *principal angles* between the
subspaces. Mean cos^2 = 0 orthogonal, 1 identical, 64/512 = 0.125 for two random sheets.
Came up checking Pythia's W_O: heads in block 0 overlap 0.23, nearly 2x random.

**privileged basis** — whether the individual coordinates of a vector space mean anything. The
residual stream has *no* privileged basis: every write goes through a dense matrix, so you
could rotate the whole stream and retrain to the same model. Elementwise things (MLP
nonlinearity, LN gain, Adam) can break this, which is how "massive activation" coordinates
happen at all. Came up asking whether heads own slots 1-64, 65-128, ...

**CKA (centered kernel alignment)** — a 0-to-1 score for how similar two point clouds look
as a whole, for the same points in two spaces (e.g. the same tokens at layer 3 and layer 4).
1 = same shape up to rotation and uniform scaling. It's global: it sees the overall layout,
not who each point's neighbours are. `linear_cka` in metrics.py; the `cka.png` heatmaps.
Came up in Q7 as the stand-in for Voita's PWCCA.

**PWCCA (projection-weighted canonical correlation analysis)** — Voita's "amount of change"
metric. CCA finds pairs of directions, one in each space, along which the two point clouds
line up best; PWCCA averages those correlations, weighting each by how much of the
representation it accounts for. Same idea as CKA (whole-cloud similarity), different
maths. Came up when checking what Voita's fig. 4 actually measures (FINDINGS 11.2).

**local vs global metric** — local metrics look at each point's own surroundings (our kNN
overlap, 1 - Jaccard); global metrics look at the whole cloud's shape (CKA, PWCCA). They can
disagree: a cloud can keep its overall shape while the points inside swap neighbours.
That's exactly what happened with rare tokens in Q7.

**off-distribution (out-of-distribution) input** — an input unlike anything the model saw in
training. Its response is still a vector, but there's no reason it reflects how the model
treats that token in real use. Came up in Q15: "[BOS] ing" (a word fragment starting a
document) is the likely reason v0's deep layers are near-useless for fragments.

**bigram; successor / predecessor vector** — a bigram is a pair of adjacent tokens. Count
every pair in the corpus, and each token gets two model-free descriptions: its successor
vector (how often each token comes *after* it) and its predecessor vector (how often each
token comes *before* it). Came up in Q3 (b), FINDINGS 11.5.

**PPMI (positive pointwise mutual information)** — reweights co-occurrence counts by "how much
more often than chance": log P(a, b) / (P(a) P(b)), negatives clipped to 0. Without it every
token's vector is mostly " the" and ",". The standard trick in distributional semantics
(words are similar if they appear in similar contexts). Came up in Q3 (b).

**right context vs left context similarity** — two tokens are right-context similar if
the same things follow them (" the" and " a"), left-context similar if the same things
come before them (" cat" and " dog" after " the"). The last hidden state follows the first,
the unembed the second (FINDINGS 11.5).
