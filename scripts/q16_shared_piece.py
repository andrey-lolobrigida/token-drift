"""FINDINGS 12 / Q16: do triples whose words share a last token piece ever come apart?

Under the last-piece convention " temper|ance" and " intemper|ance" start as the same vector (L0).
For each shared pair, per frame and group: where does the distance between the two words'
points rank among ALL word pairs in that group? 0% = closer than every pair, ~50% = a typical
pair. Related-but-unshared pairs from the same triples (cowardice/courage...) are the yardstick:
if shared pairs end up where related pairs sit, the tokenizer stopped mattering.

One-off check, not part of the pipeline. Needs runs/pythia70m_probe and runs/random_init_probe
(probe_corpus + extract). `uv run python scripts/q16_shared_piece.py`, ~1 min.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from token_drift import betweenness as bt, cli, probe as pb

MIN = 20  # metrics.min_count: same words the Q16 test can use


def pct_rank(d, all_d):
    return 100.0 * np.mean(all_d < d)


for run in ("pythia70m_probe", "random_init_probe"):
    rd = Path("runs") / run
    data = cli._load_probe(rd)
    names = cli._load_json(rd / "extract" / "layer_names.json")[:-1]
    words = pb.load_words(rd / "probe_corpus" / "probe_words.yaml")
    shared = {tuple(p) for e in cli._load_json(rd / "probe_corpus" / "shared_last_piece.json") for p in e["pairs"]}
    # yardstick: every within-triple pair that doesn't share a last piece
    related = {pr for t in words["triples"] for pr in combinations(t["words"], 2)} - shared - {p[::-1] for p in shared}
    rows = {}
    for f in range(data.occ.shape[0]):
        pts, counts, _ = bt.unit_mean(np.asarray(data.occ[f], dtype=np.float32), data.widx, data.gidx,
                                      len(data.vocab), len(cli.GROUPS))
        for g_i, g in enumerate(cli.GROUPS):
            ok = np.flatnonzero(counts[:, g_i] >= MIN)
            P = pts[ok, g_i].astype(np.float64)
            D = np.linalg.norm(P[:, None] - P[None], axis=-1)
            all_d = D[np.triu_indices(len(ok), 1)]
            pos = {data.vocab[i]: k for k, i in enumerate(ok)}
            for a, b in sorted(shared):
                if a in pos and b in pos:
                    rows.setdefault((g, f"{a} / {b}"), []).append(pct_rank(D[pos[a], pos[b]], all_d))
            rel = [pct_rank(D[pos[a], pos[b]], all_d) for a, b in related if a in pos and b in pos]
            rows.setdefault((g, f"related, unshared (median of {len(rel)})"), []).append(float(np.median(rel)))
    print(f"\n== {run}: distance rank among all word pairs in the group (% of pairs closer)")
    print(f"{'group':5} {'pair':40}" + "".join(f"{n:>13}" for n in names))
    for (g, label), v in sorted(rows.items()):
        print(f"{g:5} {label:40}" + "".join(f"{x:13.1f}" for x in v))
