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
