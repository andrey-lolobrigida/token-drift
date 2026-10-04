"""Q16 random-weight seed spread at L0, rebuilt per occurrence. Works for Pythia and GPT-2.

q16_random_seeds.py took each word's lowercase last piece as its L0 point. Two things that
shortcut misses: capitalized hits ("Courage" at a sentence start is a different token), and
GPT-2's learned positions (its L0 is wte[tok] + wpe[pos], so every occurrence differs by where
the word sits in its window). Here each occurrence is rebuilt from its window's real last token
and position, then averaged exactly like the pipeline does (bt.unit_mean). Only needs
probe_corpus, no extract.

`uv run python scripts/q16_random_seeds_l0.py runs/gpt2_probe gpt2 [n_seeds]`, a few min on CPU.
With seed "trained" it uses the real weights and, if extract/occ.npy exists, checks the rebuild
against occ[0] (they should agree to float16 precision).
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from token_drift import betweenness as bt, cli, extract as ex, probe as pb

rd, model_name = Path(sys.argv[1]), sys.argv[2]
n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 20
pc = rd / "probe_corpus"
words = pb.load_words(pc / "probe_words.yaml")
found = {w: {g: c[g]["found"] for g in cli.GROUPS} for w, c in cli._load_json(pc / "counts.json").items()}
pool = {pos: [r["word"] for r in rows] for pos, rows in cli._load_json(pc / "null_pool.json").items()}
meta = pq.read_table(pc / "occ_meta.parquet", columns=["word", "group"]).to_pydict()
vocab = sorted(set(meta["word"]))
w_index = {w: i for i, w in enumerate(vocab)}
widx = np.array([w_index[w] for w in meta["word"]])
gidx = np.array([cli.GROUPS.index(g) for g in meta["group"]])
wt, wo = np.load(pc / "windows_tokens.npy"), np.load(pc / "windows_offsets.npy")
last_tok = wt[wo[1:] - 1].astype(np.int64)  # every window ends at the word's last piece
last_pos = np.diff(wo) - 1                    # position id the model saw for that token
assert len(last_tok) == len(widx)


def l0_points(model):
    E = model.get_input_embeddings().weight.detach().float().numpy()
    occ = E[last_tok]
    wpe = getattr(getattr(model, "transformer", None), "wpe", None)  # GPT-2 only; Pythia is rotary
    if wpe is not None:
        occ = occ + wpe.weight.detach().float().numpy()[last_pos]
    points, counts, _ = bt.unit_mean(occ, widx, gidx, len(vocab), len(cli.GROUPS))
    return occ, points, counts


def q16(points, counts):
    entries, _ = bt.run_q16([points], counts, w_index, words["triples"], pool, found,
                            groups=cli.GROUPS, min_count=20, null_k=20, between_pct=5)
    s = bt.summarize(entries, cli.GROUPS, 1)
    return {k: (s[k]["pile"][0]["best_of_three"], s[k]["pile"][0]["n"]) for k in ("everyday", "classical")}


model, _ = ex.build_model(model_name, random_init=False, seed=0, device="cpu")
occ, points, counts = l0_points(model)
if (rd / "extract" / "occ.npy").exists():
    ref = np.load(rd / "extract" / "occ.npy", mmap_mode="r")[0]
    i = np.random.default_rng(0).choice(len(widx), 2000, replace=False)
    err = np.abs(np.asarray(ref[np.sort(i)], np.float32) - occ[np.sort(i)]).max()
    print(f"rebuild check vs extract occ[0], 2000 occurrences: max abs diff {err:.4f}")
print("trained L0 best_of_three (pile):", q16(points, counts))
del model, occ

res = []
for seed in range(n_seeds):
    model, _ = ex.build_model(model_name, random_init=True, seed=seed, device="cpu")
    res.append(q16(*l0_points(model)[1:]))
    del model
for k in ("everyday", "classical"):
    r = np.array([x[k][0] for x in res])
    n = res[0][k][1]
    print(f"random seeds 0-{n_seeds - 1}, L0 best_of_three {k}/pile (n={n}): {r.tolist()} "
          f"mean {r.mean():.1f} sd {r.std():.1f} (chance {n / 3:.1f})")
