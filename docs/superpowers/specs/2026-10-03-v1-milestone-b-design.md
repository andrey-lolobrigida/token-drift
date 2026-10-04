# v1 milestone B: virtue between vices, per occurrence (design)

Status: design approved in chat 2026-10-03, spec awaiting review.
Scope: milestone B = Q16 (Andrey's "virtue words land between their vices") as the headline,
on per-occurrence vectors from targeted retrieval. Polysemy (Q8) words are extracted in the
same pass but analysed later ("B2", its own short design). GPT-2 and Pythia-160m are
follow-up configs, decided after the Pythia result.

## Why

Milestone A averaged every token over the whole Pile. That can't test Q16: in the Pile
" virtue" is 57% "by virtue of", " vice" is 93% "vice president / versa", and most of
Aristotle's vocabulary is too rare to average (temperance 26 hits in 120M tokens). Three
things are needed that A doesn't have:

1. **Texts where the ethical senses live**: Aristotle (Chase), Aquinas's Summa (the
   virtue treatises), and a shelf of other public-domain moral philosophy.
2. **Enough occurrences of rare words** without running the model over billions of
   tokens: find the hits first, then run only a window ending at each hit.
3. **Per-occurrence vectors**, so a word isn't just one mean point: we can ask which of its
   occurrences sit between the vices and read their text.

The test itself has to beat a trap measured on A's vectors (2026-10-03, throwaway): in
512 dimensions three random words form a near-equilateral triangle, so a random "virtue"
lands at t = 0.50 (dead centre) with d = 0.87 (= sqrt(3)/2) at every frame. Loose triples
like humble / proud / vain give t = 0.34 to 0.64, i.e. exactly chance. A raw t means
nothing; everything is reported against a null and a role-swap control.

## Decisions (and why)

| decision | choice | why |
|---|---|---|
| headline | Q16; Q8 words stored, not analysed | Andrey's pick; Q8 is then an analysis pass with no model rerun |
| triple sets | two, reported side by side: **classical** (Chase + Aquinas wording) and **everyday** (modern words, same concepts) | if betweenness holds in one and not the other, that's a finding about wording vs concept |
| slot = ? | every word form is its own point; triples are built **within a part of speech** (abstract nouns, adjectives) | a pooled family blurs "cowardice" (quality), "coward" (person), "cowardly" (adjective); a single headword makes the result hinge on one pick |
| occurrence source | **targeted retrieval**: find hits by token match, run a window ending at each hit | Pythia is causal, so a hit's residual depends only on tokens before it; no need to run the rest of the document |
| books | Chase's Ethics, Summa I-II qq. 49-89 + II-II qq. 1-170, Republic (Jowett), Utilitarianism, Kant's Groundwork (Abbott), Seneca (L'Estrange), Epictetus (Long), Hume's moral Enquiry, Smith's Moral Sentiments | ~2.07M Pythia tokens; the Summa alone is 1.14M and carries most of the classical set. Cicero's De Officiis has no English plain-text Gutenberg edition |
| Pile | 3 shards of `EleutherAI/the_pile_deduplicated` (~360M tokens) | one shard = ~121M tokens; at 3 shards nearly every everyday triple clears 20 hits per word (counts below) |
| context | everything before the hit in the same document, capped at the model's max context (2048 Pythia), >= 32 tokens or the hit is dropped; a book is one document | same position spread as A (32-2047), so B points are comparable to A's averages; positions < 32 are attention sinks (A's `min_context`) |
| document start | one `<\|endoftext\|>` before the first token | A packs docs with EOS between them, which is how Pythia was trained; a doc start always follows an EOS |
| case | lowercase, space-prefixed forms only (`" courage"`) | capitalized / sentence-initial forms are different tokens with different embeddings; costs book hits (Chase capitalizes virtues), reported in the count report |
| word boundary | the token after the hit must not continue the word (no letter, digit, hyphen or apostrophe at its start) | else `" courage"` matches inside "courageous" and `" self"` inside "self-mastery" |
| multi-token words | vector at the last piece (A's convention) | decided 2026-09-24; caveat: multi-word names like "greatness of soul" / "littleness of soul" end on the same piece, so those are left out |
| cap | 1000 occurrences per word per source group, seeded reservoir sampling | bounds disk; uniform over a stream of unknown length |
| point per word | `unit_mean` of its occurrences in a source group, per frame | same as A; unit-norming each occurrence first tames L6pre's length spread (FINDINGS 11.3) |
| normalize stage | **skipped** in probe mode | t, d and segment distance are ratios of differences: unchanged by a shared offset (the anisotropy cone) or a uniform scale; the CLAUDE.md rule exists because kNN / CKA are not like that |
| min count | 20 per word per source group, else the triple is reported *missing* for that group | an average over 3 contexts mostly says which 3 sentences (A's reasoning) |
| null | per triple, keep the real vice pair, swap the virtue for the 20 pool words nearest in log-count (same part of speech, same source group) | fixing the pair keeps each pair's own geometry; frequency matters for vector norms and neighbourhoods (Q7) |
| role swap | the same test with each vice in the middle | separates "the mean is between" from "three related words cluster"; Aristotle's claim = the virtue beats both swaps |
| controls | random-init Pythia on the same windows | "learned or architectural?", as always |
| model-agnostic | tokenizer, EOS id and max context come from the model, never hardcoded | GPT-2 (1024 context, different tokenizer) and Pythia-160m become config-only follow-ups |

Compute (measured 2026-10-03, RTX 5060, fp32): Pythia-70m forward 0.42M tokens/s at length
2048. Worst case ~100k hits x 2048 tokens = 200M tokens = ~8 min. Tokenizing 360M Pile
tokens is the slow part (CPU, a few minutes per shard).

## Counts behind the word table

Throwaway count, 2026-10-03: regex approximation of the matching rules (lowercase,
space before, no letter / hyphen after), books = the 9 texts above, Pile = shard 0 only
(~121M tokens; the real run has ~3x). B = all three words >= 20 in books, P = >= 20 in the
shard. `[n]` = Pythia pieces.

Classical set, rows that survive somewhere:

| concept | pos | deficiency / mean / excess | books | pile shard |
|---|---|---|---|---|
| fear | noun | cowardice / courage / rashness | 44 / 118 / 32 B | 72 / 944 / 3 |
| fear | noun | fear / courage / fearlessness | 1188 / 118 / 11 | 4633 / 944 / 28 P |
| fear | noun | timidity / fortitude / daring | 15 / 602 / 114 | 19 / 93 / 313 (P at 3 shards) |
| fear | adj | cowardly / brave / rash | 14 / 127 / 27 | 127 / 837 / 226 P |
| pleasure | noun | insensibility / temperance / intemperance | 40 / 508 / 95 B | 3 / 26 / 5 |
| pleasure | adj | insensible / temperate / intemperate | 23 / 55 / 54 B | 21 / 119 / 10 |
| giving | noun | covetousness / liberality / prodigality | 224 / 174 / 49 B | 8 / 13 / 6 |
| giving | noun | avarice / liberality / prodigality | 56 / 174 / 49 B | 27 / 13 / 6 |
| giving | adj | covetous / liberal / prodigal | 60 / 79 / 44 B | 13 / 1308 / 29 |
| giving | adj | stingy / liberal / prodigal | 3 / 79 / 44 | 59 / 1308 / 29 P |
| spending | noun | meanness / magnificence / vulgarity | 57 / 124 / 2 | 40 / 44 / 50 P |
| honour | noun | pusillanimity / magnanimity / presumption | 42 / 253 / 131 B | 2 / 13 / 762 |
| honour | noun | pusillanimity / magnanimity / vainglory | 42 / 253 / 128 B | 2 / 13 / 4 |
| honour | noun | pusillanimity / magnanimity / vanity | 42 / 253 / 74 B | 2 / 13 / 213 |
| anger | noun | insensibility / meekness / anger | 40 / 89 / 541 B | 3 / 11 / 1556 |
| anger | adj | spiritless / meek / passionate | 2 / 23 / 42 | 0 / 59 / 1055 |
| truth | noun | irony / truthfulness / boasting | 36 / 6 / 61 | 483 / 57 / 146 P |
| shame | adj | shameless / modest / bashful | 9 / 31 / 1 | 87 / 800 / 26 P |
| neighbour | noun | spite / indignation / envy | 25 / 72 / 205 B | 817 / 88 / 269 P |

Dropped for counts (0 to 9 hits): wit (boorishness / wittiness / buffoonery, Chase's
clownishness / easy-pleasantry), company nouns (quarrelsomeness), Chase's paltriness /
munificence, angerlessness / passionateness, braggadocio, shamefacedness / bashfulness,
pusillanimous, self-mastery / self-indulgence (Chase capitalizes "Self-Mastery": 66 hits
case-insensitive, 29 lowercase). "insensibility" doubles as Aquinas's deficiency of
meekness; that row is kept but flagged. Also flagged: fear / courage / fearlessness, where
"fear" is the passion, not the vice (the vice is cowardice); kept because the Pile has it.

Everyday set, all rows (the books barely have these words; the Pile carries them):

| concept | pos | deficiency / mean / excess | pile shard |
|---|---|---|---|
| fear | noun | cowardice / bravery / recklessness | 72 / 189 / 68 P |
| fear | adj | cowardly / brave / reckless | 127 / 837 / 413 P |
| money | noun | stinginess / generosity / extravagance | 9 / 326 / 48 (P at 3 shards) |
| money | adj | stingy / generous / wasteful | 59 / 987 / 107 P |
| money | adj | stingy / generous / extravagant | 59 / 987 / 149 P |
| pleasure | noun | asceticism / moderation / indulgence | 31 / 213 / 112 P |
| pleasure | adj | ascetic / moderate / indulgent | 39 / 1044 / 82 P |
| self | noun | insecurity / confidence / arrogance | 184 / 3000 / 222 P |
| self | adj | insecure / confident / arrogant | 234 / 1668 / 304 P |
| self | adj | timid / confident / arrogant | 129 / 1668 / 304 P |
| anger | noun | apathy / patience / irritability | 77 / 859 / 45 P |
| anger | adj | apathetic / patient / irritable | 36 / 4813 / 96 P |
| anger | adj | passive / calm / irritable | 618 / 1494 / 96 P |
| truth | noun | secrecy / honesty / boastfulness | 259 / 461 / 0 |
| truth | adj | secretive / honest / boastful | 138 / 2390 / 14 (P at 3 shards) |
| wit | adj | boring / witty / silly | 970 / 191 / 1043 P |
| company | noun | hostility / friendliness / obsequiousness | 264 / 66 / 1 |
| company | adj | hostile / friendly / obsequious | 599 / 2817 / 16 (P at 3 shards) |
| drive | noun | laziness / ambition / ruthlessness | 91 / 453 / 29 P |
| drive | adj | lazy / ambitious / ruthless | 893 / 687 / 261 P |
| shame | noun | shyness / modesty / shamelessness | 40 / 103 / 3 |
| shame | adj | shy / modest / shameless | 680 / 800 / 87 P |

Sense caveats visible already: Pile " patient" is mostly medical, " friendly" mostly
"user-friendly", " liberal" mostly political, " moderate" mostly "moderate (amount)",
" reserved" / " reserve" mostly "all rights reserved" / Federal Reserve (left out).
That's the point of reporting books and Pile separately, not a bug.

These tables are the **first draft of `configs/probe_words.yaml`** (Andrey edits it).
The pipeline re-counts with real token matching and reports every missing triple.

## Config

New `configs/pythia70m_probe.yaml`; `random_init_probe.yaml` differs by
`random_init: true` and `run_name`.

```yaml
run_name: pythia70m_probe
model: EleutherAI/pythia-70m
random_init: false
seed: 0
device: auto

probe:
  words: configs/probe_words.yaml
  cache_dir: ~/.cache/token-drift      # Gutenberg downloads; outside the repo
  books:                               # start/end = marker lines (first exact match);
    - {name: chase_ethics, gutenberg: 8438}          # without them: the Gutenberg body
    - {name: summa_1_2, gutenberg: 17897, start: "QUESTION 49", end: "QUESTION 90"}
    - {name: summa_2_2, gutenberg: 18755,
       start: "TREATISE ON THE THEOLOGICAL VIRTUES (QQ. 1-46)",
       end: "TREATISE ON GRATUITOUS GRACES (QQ. 171-182)"}
    - {name: republic, gutenberg: 1497}
    - {name: utilitarianism, gutenberg: 11224}
    - {name: kant_groundwork, gutenberg: 5682}
    - {name: seneca_morals, gutenberg: 56075}
    - {name: epictetus, gutenberg: 10661}
    - {name: hume_morals, gutenberg: 4320}
    - {name: smith_sentiments, gutenberg: 67363}
  pile:
    source: EleutherAI/the_pile_deduplicated
    text_field: text
    shards: 3                          # ~121M tokens each; files in sorted order
  cap: 1000                            # per word per source group (books | pile)
  min_context: 32                      # same as A: positions < 32 are attention sinks
  max_context: null                    # null = model.config.max_position_embeddings

extract:
  mode: probe                          # vocab (v0) | corpus (A) | probe (B)
  batch_size: 16                       # windows per forward, sorted by length
  dtype_on_disk: float16

metrics:
  min_count: 20                        # per word per source group
  null_pool: 50                        # words per part of speech
  null_k: 20                           # nearest-in-log-count pool words per triple
  between_pct: 5                       # "beats the null" = closer than 95% of null words

viz:
  occ_frames: [0, 3, 6]                # frames for the per-occurrence histograms
```

`configs/probe_words.yaml`:

```yaml
sets:
  classical:
    fear:
      noun: [[cowardice, courage, rashness], [fear, courage, fearlessness],
             [timidity, fortitude, daring]]
      adj:  [[cowardly, brave, rash]]
    # ... the rest of the classical table above
  everyday:
    fear:
      noun: [[cowardice, bravery, recklessness]]
      adj:  [[cowardly, brave, reckless]]
    # ... the rest of the everyday table above
polysemy: [bank, bat, spring, bass, match, light, cell, plant, mouse, python, apple, java,
           pitch, current, table, net, tie, seal, mole, trunk, bark, date, fair, kind, saw,
           lead, right, left]        # A's trajectory group; B2 analyses them
null_exclude: []                     # junk the suffix heuristic lets through; add and rerun
```

Words are written bare; the code prepends the space. A word that appears in several triples
(courage, liberality, insensibility) is extracted once.

## Data flow

```
probe_corpus/    books: download (cached) -> marker slice -> tokenize -> one doc each
                 pile: stream shards -> tokenize -> uint16 ids in RAM (~0.7 GB) + doc offsets
                 pass 1: count whole-word hits of every lowercase space-prefixed single-token
                         word per source group -> pick the null pool (seeded)
                 pass 2: hits of probe + null + polysemy words -> boundary check ->
                         context >= 32 -> reservoir (cap per word per group)
                 -> windows_tokens.npy (uint16, flat), windows_offsets.npy (int64),
                    occ_meta.parquet, null_pool.json, counts.json, books.json (sha256s)
extract/         ragged windows -> right-pad, sorted by length -> residual at each window's
                 last real position, 8 frames -> occ.npy (8, n_occ, d) float16
                 (+ embed.npy, unembed.npy, final_ln.npz as in A)
normalize/       skipped (see decisions)
metrics/         points = unit_mean per word x source group x frame -> t, d, segment
                 distance, null percentiles, role swaps, per-occurrence (t, d), per-word
                 self-similarity -> q16.json, count_report.md
viz/             q16_summary.png, q16_<set>_<concept>_<pos>.png, q16_occ_<...>.png
```

The `all` command runs `probe_corpus` instead of `corpus` when the config has a `probe:`
block. A config with both `corpus:` and `probe:` is an error.

## Components

### `probe.py` (new)

Pure functions, arrays in and out; file and network I/O stays in `cli.py` except the
Gutenberg fetch helper.

- `fetch_gutenberg(id, cache_dir) -> str`: download `pg{id}.txt` once, keep it in the cache.
- `gutenberg_body(text, start=None, end=None) -> str`: cut between the `*** START` /
  `*** END` lines, then between the first exact `start` line and the first `end` line
  after it (start inclusive, end exclusive). A missing marker raises, naming it.
- `word_token_ids(tokenizer, words) -> dict[str, tuple[int, ...]]`: ids of `" " + word`.
- `boundary_ok(next_ids, tokenizer) -> bool mask`: the token after the hit must not start
  with a letter, digit, hyphen or apostrophe (a token that begins with a space or
  punctuation, or EOS, is fine). Decided on the decoded token string, cached per id.
- `find_hits(ids, doc_offsets, patterns) -> (word_idx, doc, pos)`: every position where a
  pattern's full id sequence ends, inside one document, passing the boundary check.
  Vectorized per pattern length (numpy sliding comparison), not a Python loop over tokens.
- `count_single_token_words(ids, doc_offsets, candidate_ids) -> counts`: pass 1, for the
  null pool.
- `pick_null_pool(counts_books, counts_pile, tokens, probe_words, exclude, n, seed)`:
  candidates = single-token lowercase alphabetic words with >= `min_count` in **both**
  groups, not in the word file; part of speech by suffix (abstract noun: -ness, -ity,
  -ence, -ance, -ion, -ment, -ism, -ship, -dom; adjective: -ous, -ful, -ive, -ent, -ant,
  -less, -able, -ible, -al, -ic, -ish). Per part of speech, `n` words picked so the
  pool's book counts look like the probe words' book counts (books are the binding
  group): for each probe word of that pos, the unused candidate nearest in log book-count,
  then a seeded random fill up to `n`. Junk goes in `null_exclude`.
- `reservoir(keys, cap, seed) -> kept indices`: seeded reservoir sampling per key
  (key = word x source group). Deterministic for a given seed and hit order.
- `cut_windows(ids, doc_offsets, hits, min_context, max_context, eos_id)`: window = EOS (if
  the context reaches the doc start) + up to `max_context - 1` preceding tokens + the hit's
  pieces, ending at the hit's last piece. Drops hits with < `min_context` tokens before
  the last piece. Returns flat tokens + offsets.
- `snippet(tokenizer, window, n_chars=150) -> str`: decoded tail of the window, for
  `occ_meta.parquet`.

`occ_meta.parquet` columns: `word`, `role_tags` (which sets / concepts / slots use it, or
`null` / `polysemy`), `group` (books | pile), `source` (book name or `pile:<shard>`),
`doc`, `position` (index of the last piece = context length), `n_pieces`, `snippet`.

`counts.json`: per word x group: hits found, hits passing min_context, kept after cap,
capitalized / sentence-initial hits seen but skipped (for the count report).

### `extract.py`: probe mode (new function)

`extract_probe(model, tokens, offsets, *, batch_size, device) -> (n_frames, n_occ, d) float16`.
Same frames and same LN pre-hook as `extract_corpus_means`; `model.base_model` (no logits).
Windows are sorted by length and right-padded per batch (pad id = EOS); the vector is read
at each window's last real position. Right-padding is safe because attention is causal:
tokens after the hit can't change it, and positions before it don't shift. A test checks
this rather than trusting it (batched-and-padded vs one-at-a-time, `allclose`).
Results go back in the original window order.

### `betweenness.py` (new)

All pure numpy on float32 arrays; one frame at a time.

- `unit_mean(occ, word_idx, group_idx, min_count) -> points (n_words, n_groups, d), counts`.
- `segment_stats(a, v, b) -> t, d, seg`:
  - t = (v - a)·(b - a) / |b - a|^2 (where v projects on the line; 0 = a, 1 = b)
  - d = |v - a - t(b - a)| / |b - a| (distance off the line)
  - seg = distance from v to the closest point of segment [a, b], divided by |b - a|
    (= d when 0 <= t <= 1, else distance to the nearer end). Headline score: small only
    when v is between a and b.
  Broadcasts over many v at once (the null).
- `null_percentile(a, b, v, null_vs) -> pct`: % of null words with seg <= the real one's
  (0 = closer than every null word).
- `role_swap(a, v, b, null_for_each_role) -> pct per placement`: mean-in-the-middle,
  deficiency-in-the-middle, excess-in-the-middle, each with its own null (swap the
  middle word for its role's nearest-count pool words).
- `triple_verdict(pcts, between_pct) -> {"beats_null": bool, "best_of_three": bool}`:
  beats_null = the virtue-in-the-middle pct < `between_pct`; best_of_three = its pct is
  strictly lower than both swapped placements' pcts (ties count as not best).
- `set_summary(verdicts) -> per frame: n triples, n beats_null, n best_of_three,
  binomial p of best_of_three vs chance 1/3`.
- `occurrence_stats(occ_v, a, b) -> t, d per occurrence` (unit-normed occurrences of the
  virtue vs the two vice means; scale matches the points).
- `self_similarity` per word: reuse `extract.self_similarity` on the unit-normed sums.

### `metrics` (probe mode) in `cli.py`

Loads `occ.npy` frame by frame (upcast per frame), builds points, runs every triple of both
sets for both groups and every frame. Writes:

- `q16.json`: per set / concept / pos / triple / group / frame: t, d, seg, null pct, the
  three role-swap pcts, verdict; `set_summary` per set x group; per-word self-sim;
  `missing`: triples skipped per group with the word that fell short.
- `count_report.md`: the counts table with real token matching (replaces this spec's regex
  numbers), the null pool with each word's counts, capitalized hits skipped.

### `viz.py` (probe mode)

- `q16_summary.png`: one heatmap per group (books, pile); rows = triples grouped
  set -> concept -> pos, columns = frames, colour = virtue's null pct (sequential palette,
  low = between), a dot where the virtue is best of three, hatched = missing.
- `q16_<set>_<concept>_<pos>.png`: one panel per frame, the exact (t, d) plane: grey
  null words, filled dot = virtue, hollow = the two role swaps; vertical lines at t = 0
  and 1. Three points always lie in a plane, so there's no projection distortion. Same
  axis limits across panels (CLAUDE.md flipbook rule).
- `q16_occ_<set>_<concept>_<pos>.png`: per-occurrence t and d histograms of the virtue,
  books vs pile overlaid, at `viz.occ_frames`.

### `cli.py`

New `probe_corpus` stage; `extract`, `metrics`, `viz` dispatch on `extract.mode: probe`;
`all` picks the stage list from the config. A `token-drift occ` helper prints the most and
least between occurrences of a word for a triple and frame, with snippets (reading
the senses is half the point).

## Where each open question gets answered

| question | where | what we'll know |
|---|---|---|
| Q16 virtue between vices | `q16.json`, summary + triple figures | per set (classical / everyday) and group (books / pile): how many triples beat the null, how many have the virtue best of three, at which layers |
| Q8 polysemy | B2 (stored here) | nothing yet; occurrences are on disk |
| Q9 self-sim | `q16.json` per-word self-sim | raw numbers only; intra-sentence sim and max explainable variance need every token, out of scope |
| Q13 heavy-tail hunch | follow-up script | `occ.npy` keeps raw (not unit-normed) residuals, so the L6pre norm distribution per word is a few lines |
| Q14 spread vs mean | deferred | needs a shuffled version of the probe windows |
| Q7 per occurrence | deferred | B's word set is too small for frequency bins; needs its own design |

## Errors

- Missing marker in a book: raise with the book name and marker.
- Gutenberg download fails: raise; a cached copy is used without network. `books.json`
  records sha256 per book and the stage warns if a cached book's hash differs from the run's
  previous `books.json`.
- A word whose tokenization contains the EOS id or is empty: raise at config load.
- Token ids >= 65536 (uint16 overflow): raise; switch the dtype in that case.
- Both `corpus:` and `probe:` in one config: raise.
- A triple with a word below `min_count` in a group: not an error, listed in `missing`.
- Fewer than `null_k` pool words for a part of speech: raise, naming the pos and the
  count range (fix by widening `null_pool` or the word list).

## Testing

Tiny and fast, no network. The 200-token-vocab model fixture from A for extract; the real
Pythia tokenizer (already cached) for matching tests.

- `gutenberg_body`: header/footer cut, markers inclusive/exclusive, missing marker raises.
- `find_hits` + `boundary_ok`: " courage" found, "courageous" not, " self" not inside
  "self-mastery", multi-token word found at its last piece, no match across a document
  boundary, capitalized form not matched.
- `cut_windows`: context capped at max_context, hits under min_context dropped, EOS added
  only when the context reaches the doc start.
- `reservoir`: never more than cap, same seed = same picks, roughly uniform over 10k
  synthetic hits.
- `pick_null_pool`: suffix rules, both-groups requirement, excluded words gone, seeded.
- `extract_probe`: padded batch == one-window-at-a-time (allclose), original order restored,
  8 frames.
- `segment_stats`: midpoint -> t 0.5, d 0, seg 0; point beyond b on the line -> t > 1,
  d 0, seg > 0; unchanged under translation and uniform scaling.
- `null_percentile`, `role_swap`, `set_summary` on hand-built 2-D/3-D points.

## Order of work (one small commit each, `pytest` before each)

1. `configs/probe_words.yaml` (draft from the tables above) + config loading/validation.
2. `probe.py`: Gutenberg fetch/slice, word ids, boundary, `find_hits`, tests.
3. `probe.py`: reservoir, `cut_windows`, pass-1 counts, null pool, tests.
4. `cli.py`: `probe_corpus` stage end to end (books + Pile streaming), count report draft.
5. `extract.py`: `extract_probe` + padding test; cli dispatch.
6. `betweenness.py` + tests.
7. `cli.py`: probe metrics -> `q16.json`, `count_report.md`.
8. `viz.py`: the three figure kinds; `token-drift occ` helper.
9. Real runs: `pythia70m_probe`, `random_init_probe`. Andrey reviews the count report and
   null pool, edits the word file if needed, rerun.
10. FINDINGS section 12 written together with Andrey; OPEN_QUESTIONS Q16 verdict;
    README v1 milestone B paragraph.

## Definition of done

- `token-drift all --config configs/pythia70m_probe.yaml` and the random-init twin run end
  to end.
- `q16.json`, `count_report.md` and the three figure kinds exist for both.
- FINDINGS section 12 gives a verdict per triple set (classical / everyday) per group
  (books / pile), random-init alongside.
- `pytest` green.

## Out of scope for B

Polysemy analysis (B2); GPT-2 and Pythia-160m runs (follow-up configs); shuffled probe
windows (Q14); intra-sentence similarity and max explainable variance (Q9); per-occurrence
frequency effect (Q7); capitalized word forms; multi-word names ("greatness of soul").
