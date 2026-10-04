"""FINDINGS 12 / Q16: how much does the random-init control's best-of-three count wobble by seed?

random_init_probe is ONE draw of random weights. At L0 a word's point is just its (last piece's)
embedding row, so other draws need no extract: build 20 random-init models, take the rows, rerun
the Q16 test at that frame. The spread of counts across seeds is the honest null for "how many
triples are best of three", since triples share words and aren't independent coin flips.

One-off check, not part of the pipeline. Needs runs/random_init_probe (probe_corpus + extract).
`uv run python scripts/q16_random_seeds.py`, ~1 min.
"""
from pathlib import Path

import numpy as np

from token_drift import betweenness as bt, cli, extract as ex, probe as pb

rd = Path("runs/random_init_probe")
pc = rd / "probe_corpus"
words = pb.load_words(pc / "probe_words.yaml")
found = {w: {g: c[g]["found"] for g in cli.GROUPS} for w, c in cli._load_json(pc / "counts.json").items()}
pool = {pos: [r["word"] for r in rows] for pos, rows in cli._load_json(pc / "null_pool.json").items()}
data = cli._load_probe(rd)
# counts decide which triples are usable; same at every frame
_, counts, _ = bt.unit_mean(np.asarray(data.occ[0], dtype=np.float32), data.widx, data.gidx, len(data.vocab), 2)
tok = cli._load_tokenizer("EleutherAI/pythia-70m")
last = [tok(" " + w, add_special_tokens=False)["input_ids"][-1] for w in data.vocab]
res = []
for seed in range(20):
    model, _ = ex.build_model("EleutherAI/pythia-70m", random_init=True, seed=seed, device="cpu")
    E = model.get_input_embeddings().weight.detach().numpy()[last]
    P = (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)
    entries, _ = bt.run_q16([np.stack([P, P], 1)], counts, data.w_index, words["triples"], pool, found,
                            groups=cli.GROUPS, min_count=20, null_k=20, between_pct=5)
    s = bt.summarize(entries, cli.GROUPS, 1)
    res.append((s["everyday"]["pile"][0]["best_of_three"], s["classical"]["pile"][0]["best_of_three"]))
r = np.array(res)
print("random-weight seeds 0-19, L0 best_of_three")
print("  everyday/pile  (n=18):", r[:, 0].tolist(), f"mean {r[:, 0].mean():.1f} sd {r[:, 0].std():.1f} (chance 6.0)")
print("  classical/pile (n=10):", r[:, 1].tolist(), f"mean {r[:, 1].mean():.1f} sd {r[:, 1].std():.1f} (chance 3.3)")
