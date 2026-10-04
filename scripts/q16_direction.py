"""FINDINGS 12 / Q16: is there one shared "too little -> too much" direction across triples?

Per triple, the arrow deficiency -> excess (between the two vices' points). If Aristotle's mean is
one axis in the model, arrows of different triples point the same way: positive mean cosine.
Only pairs of triples whose vices share no word AND no last token piece count: a shared endpoint
aligns arrows for free, and so does a shared piece (wasteful / boastful both end in "ful", and
under the last-piece convention that piece's embedding rides along in both arrows).

Null: each vice swapped for one of its 20 nearest-in-log-count pool words (same pos, same group),
1000 resamples. Keeps the frequency structure (if excess words were rarer than deficiency words,
real AND null arrows would share a frequency direction) and drops the meaning.
Shared-last-piece triples are left out (FINDINGS 12: tokenizer artefact).

One-off check, not part of the pipeline. Needs runs/*_probe (probe_corpus + extract).
`uv run python scripts/q16_direction.py`, ~2 min.
"""
from itertools import combinations
from pathlib import Path

import numpy as np

from token_drift import betweenness as bt, cli, probe as pb

MIN, K, N_NULL = 20, 20, 1000


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


for run in ("pythia70m_probe", "random_init_probe"):
    rd = Path("runs") / run
    pc = rd / "probe_corpus"
    data = cli._load_probe(rd)
    names = cli._load_json(rd / "extract" / "layer_names.json")[:-1]
    words = pb.load_words(pc / "probe_words.yaml")
    found = {w: {g: c[g]["found"] for g in cli.GROUPS} for w, c in cli._load_json(pc / "counts.json").items()}
    pool = {pos: [r["word"] for r in rows] for pos, rows in cli._load_json(pc / "null_pool.json").items()}
    shared = {e["id"] for e in cli._load_json(pc / "shared_last_piece.json")}
    triples = [t for t in words["triples"] if t["id"] not in shared]
    tok = cli._load_tokenizer("EleutherAI/pythia-70m")
    last = lambda w: tok(" " + w, add_special_tokens=False)["input_ids"][-1]  # noqa: E731
    ends = [{t["words"][0], t["words"][2], last(t["words"][0]), last(t["words"][2])} for t in triples]
    tix = {t["id"]: i for i, t in enumerate(triples)}
    rng = np.random.default_rng(0)
    print(f"\n== {run}: mean cosine between deficiency->excess arrows of word- and piece-disjoint triple pairs")
    print(f"{'set/group':17} {'pairs':>5}  " + "  ".join(f"{n:>16}" for n in names))
    out = {}
    for f in range(data.occ.shape[0]):
        pts, counts, _ = bt.unit_mean(np.asarray(data.occ[f], dtype=np.float32), data.widx, data.gidx,
                                      len(data.vocab), len(cli.GROUPS))
        for g_i, g in enumerate(cli.GROUPS):
            ok = lambda w: w in data.w_index and counts[data.w_index[w], g_i] >= MIN  # noqa: E731
            P = lambda w: pts[data.w_index[w], g_i].astype(np.float64)  # noqa: E731
            present = [t for t in triples if all(ok(w) for w in t["words"])]
            for set_name in ("classical", "everyday", "both"):
                ts = [t for t in present if set_name in ("both", t["set"])]
                pairs = [(i, j) for i, j in combinations(range(len(ts)), 2)
                         if not set(ts[i]["words"]) & set(ts[j]["words"])
                         and not ends[tix[ts[i]["id"]]] & ends[tix[ts[j]["id"]]]]
                if len(pairs) < 3:
                    continue
                I, J = np.array(pairs).T
                real = unit(np.stack([P(t["words"][2]) - P(t["words"][0]) for t in ts]))
                stat = float(np.mean(np.sum(real[I] * real[J], 1)))
                # frequency-matched stand-ins for each triple's two vices
                cand = []
                for t in ts:
                    usable = {w: found[w][g] for w in pool[t["pos"]] if ok(w)}
                    cand.append([bt.nearest_in_log_count(found[t["words"][k]][g], usable, K) for k in (0, 2)])
                null = np.empty(N_NULL)
                for r in range(N_NULL):
                    arr = []
                    for de, ex_ in cand:
                        a = rng.choice(de)
                        b = rng.choice([w for w in ex_ if w != a])
                        arr.append(P(b) - P(a))
                    arr = unit(np.stack(arr))
                    null[r] = np.mean(np.sum(arr[I] * arr[J], 1))
                p = float(np.mean(null >= stat))
                out.setdefault((set_name, g, len(pairs)), []).append(f"{stat:+.3f} ({null.mean():+.3f}) p{p:.2f}")
    for (s, g, n), cells in sorted(out.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"{s + '/' + g:17} {n:5}  " + "  ".join(f"{c:>16}" for c in cells))
    print("cells: real mean cosine (null mean) p = share of null resamples >= real")
