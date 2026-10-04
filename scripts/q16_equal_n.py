"""FINDINGS 12 / Q16: do the verdicts survive giving every word the same number of occurrences?

Kept counts run from 20 to 1000, and a unit_mean point's noise shrinks like 1/sqrt(n): a rare
word's point is noisier, which pushes it off the vice-vice segment. Here every word x group is
subsampled to exactly `metrics.min_count` = 20 occurrences (5 seeds), points rebuilt, and the
whole Q16 test rerun. If the summary barely moves, unequal counts weren't driving it.

One-off check, not part of the pipeline. Needs runs/*_probe (probe_corpus + extract).
`uv run python scripts/q16_equal_n.py`, ~2 min.
"""
from pathlib import Path

import numpy as np

from token_drift import betweenness as bt, cli, probe as pb

N, SEEDS = 20, range(5)


def summary_line(summ, s, g):
    pf = summ[s][g]
    return [f"{x['best_of_three']}/{x['n']}" for x in pf], [x["beats_null"] for x in pf]


for run in ("pythia70m_probe", "random_init_probe"):
    rd = Path("runs") / run
    pc = rd / "probe_corpus"
    data = cli._load_probe(rd)
    names = cli._load_json(rd / "extract" / "layer_names.json")[:-1]
    words = pb.load_words(pc / "probe_words.yaml")
    found = {w: {g: c[g]["found"] for g in cli.GROUPS} for w, c in cli._load_json(pc / "counts.json").items()}
    pool = {pos: [r["word"] for r in rows] for pos, rows in cli._load_json(pc / "null_pool.json").items()}
    key = data.widx * len(cli.GROUPS) + data.gidx
    sels = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = [rng.choice(np.flatnonzero(key == k), size=min(N, int((key == k).sum())), replace=False)
               for k in np.unique(key)]
        sels.append(np.sort(np.concatenate(idx)))
    points = {s: [] for s in SEEDS}
    counts = {}
    for f in range(data.occ.shape[0]):
        occ_f = np.asarray(data.occ[f], dtype=np.float32)
        for s, sel in zip(SEEDS, sels):
            pts, c, _ = bt.unit_mean(occ_f[sel], data.widx[sel], data.gidx[sel], len(data.vocab), len(cli.GROUPS))
            points[s].append(pts)
            counts[s] = c
    print(f"\n== {run}: best_of_three/n per frame, every word subsampled to {N} occurrences")
    print(f"{'set/group':16} {'seed':>4} " + " ".join(f"{n:>9}" for n in names))
    full = cli._load_json(rd / "metrics" / "q16.json")["summary"]
    summ = {}
    for s in SEEDS:
        entries, _ = bt.run_q16(points[s], counts[s], data.w_index, words["triples"], pool, found,
                                groups=cli.GROUPS, min_count=N, null_k=20, between_pct=5)
        summ[s] = bt.summarize(entries, cli.GROUPS, len(names))
    for s_name in full:
        for g in cli.GROUPS:
            if all(x["n"] == 0 for x in full[s_name][g]):
                continue
            b3, _ = summary_line(full, s_name, g)
            print(f"{s_name + '/' + g:16} {'full':>4} " + " ".join(f"{x:>9}" for x in b3))
            for s in SEEDS:
                b3, _ = summary_line(summ[s], s_name, g)
                print(f"{'':16} {s:>4} " + " ".join(f"{x:>9}" for x in b3))
