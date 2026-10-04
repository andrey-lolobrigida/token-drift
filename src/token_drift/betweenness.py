"""Q16: does a virtue's point sit between its two vices' points? Pure numpy, one frame at a time.

The trap (spec, "Why"): in 512 dims three random words form a near-equilateral triangle, so a
random "virtue" lands at t = 0.5, d = 0.87 at every frame. A raw t means nothing; every number
here is judged against frequency-matched null words and against the role-swapped triple.
"""
from __future__ import annotations

from math import comb

import numpy as np

from token_drift.extract import self_similarity

ENDS = {0: (1, 2), 1: (0, 2), 2: (0, 1)}  # placement k: word k in the middle, the other two are the ends


def unit_mean(occ, word_idx, group_idx, n_words: int, n_groups: int):
    """One frame of occurrences -> (points, counts, self_sim), each indexed [word, group].

    Points average unit-normed occurrences (A's unit_mean): a few huge-norm occurrences
    (L6pre, FINDINGS 11.3) can't drag the point. No occurrences -> NaN point, on purpose.
    """
    occ = np.asarray(occ, dtype=np.float32)
    u = occ / np.maximum(np.linalg.norm(occ, axis=1, keepdims=True), 1e-8)
    key = np.asarray(word_idx) * n_groups + np.asarray(group_idx)
    K = n_words * n_groups
    sums = np.zeros((K, occ.shape[1]), dtype=np.float64)
    np.add.at(sums, key, u)
    counts = np.bincount(key, minlength=K)
    with np.errstate(invalid="ignore", divide="ignore"):
        points = (sums / counts[:, None]).astype(np.float32)
    ss = self_similarity((sums**2).sum(1), counts)
    shape = (n_words, n_groups)
    return points.reshape(*shape, -1), counts.reshape(shape), ss.reshape(shape)


def segment_stats(a, v, b):
    """t = where v projects on the a->b line (0 = a, 1 = b), d = distance off the line,
    seg = distance to the closest point of the *segment* [a, b] (= d when 0 <= t <= 1),
    all in units of |b - a|. Broadcasts over leading dims of v."""
    a, b, v = (np.asarray(x, dtype=np.float64) for x in (a, b, v))
    ab = b - a
    L = np.sqrt(ab @ ab)
    t = np.asarray(((v - a) @ ab) / (L * L))
    d = np.linalg.norm(v - (a + t[..., None] * ab), axis=-1) / L
    tc = np.clip(t, 0.0, 1.0)
    seg = np.linalg.norm(v - (a + tc[..., None] * ab), axis=-1) / L
    return t, d, seg


def null_percentile(a, b, v, null_vs) -> float:
    """% of null words at least as close to segment [a, b] as v (0 = closer than all of them)."""
    real = segment_stats(a, v, b)[2]
    nulls = segment_stats(a, np.asarray(null_vs), b)[2]
    return float(100.0 * np.mean(nulls <= real))


def role_swap(trip, nulls) -> list[float]:
    """Null percentile with each word of the triple in the middle (ROLES order), each against
    its own null words. Aristotle's claim = the mean (index 1) beats both swaps."""
    return [null_percentile(trip[ENDS[k][0]], trip[ENDS[k][1]], trip[k], nulls[k]) for k in range(3)]


def triple_verdict(pcts, between_pct: float) -> dict:
    return {"beats_null": bool(pcts[1] < between_pct),
            "best_of_three": bool(pcts[1] < pcts[0] and pcts[1] < pcts[2])}  # ties: not best


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return float(sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def set_summary(verdicts: list[dict]) -> dict:
    """How many triples beat the null / have the virtue best of three, and how surprising the
    best-of-three count is if the middle word were picked at random (chance = 1/3)."""
    n = len(verdicts)
    best = sum(v["best_of_three"] for v in verdicts)
    return {"n": n, "beats_null": sum(v["beats_null"] for v in verdicts), "best_of_three": best,
            "p_best_of_three": binom_sf(best, n, 1 / 3) if n else None}


def occurrence_stats(occ_v, a, b):
    """(t, d, seg) of each unit-normed occurrence of the middle word vs the two end points."""
    occ_v = np.asarray(occ_v, dtype=np.float64)
    u = occ_v / np.maximum(np.linalg.norm(occ_v, axis=1, keepdims=True), 1e-8)
    return segment_stats(a, u, b)


def nearest_in_log_count(target: int, cands: dict[str, int], k: int) -> list[str]:
    """The k candidates closest in log count (frequency moves norms and neighbourhoods, Q7)."""
    words = sorted(cands)
    lc = np.log(np.maximum([cands[w] for w in words], 1))
    order = np.argsort(np.abs(lc - np.log(max(target, 1))), kind="stable")
    return [words[i] for i in order[:k]]


def _f(x) -> float:
    return float(x)


def run_q16(points, counts, w_index, triples, pool, found, *, groups, min_count, null_k, between_pct):
    """Score every triple in every group at every frame -> (entries, missing).

    points: one (n_words, n_groups, d) array per frame. counts: occurrences behind each point.
    found[word][group]: uncapped hit count, for picking null words (kept counts all sit at the cap).
    """
    entries, missing = [], []
    for tr in triples:
        entry = {k: tr[k] for k in ("id", "set", "concept", "pos")} | {"words": list(tr["words"]), "groups": {}}
        for g_i, g in enumerate(groups):
            short = [w for w in tr["words"] if w not in w_index or counts[w_index[w], g_i] < min_count]
            if short:
                entry["groups"][g] = {"missing": short}
                missing.append({"id": tr["id"], "group": g, "short": short})
                continue
            usable = {w: found[w][g] for w in pool.get(tr["pos"], [])
                      if w in w_index and counts[w_index[w], g_i] >= min_count}
            if len(usable) < null_k:
                c = sorted(usable.values())
                raise ValueError(f"{tr['pos']} null pool has {len(usable)} usable words in {g} "
                                 f"(found counts {c[:1]}..{c[-1:]}), need null_k={null_k}; widen null_pool")
            nulls = [nearest_in_log_count(found[w][g], usable, null_k) for w in tr["words"]]
            rows = [w_index[w] for w in tr["words"]]
            nrows = [[w_index[w] for w in ns] for ns in nulls]
            e = {"counts": [int(counts[r, g_i]) for r in rows], "null_words": nulls[1],
                 **{k: [] for k in ("t", "d", "seg", "null_pct", "swap_pct", "td", "null_td",
                                    "beats_null", "best_of_three")}}
            for P in points:
                trip = P[rows, g_i].astype(np.float64)
                null_pts = [P[nr, g_i].astype(np.float64) for nr in nrows]
                pcts = role_swap(trip, null_pts)
                t, d, seg = segment_stats(trip[0], trip[1], trip[2])
                nt, nd, _ = segment_stats(trip[0], null_pts[1], trip[2])
                td = [segment_stats(trip[ENDS[k][0]], trip[k], trip[ENDS[k][1]])[:2] for k in range(3)]
                v = triple_verdict(pcts, between_pct)
                e["t"].append(_f(t))
                e["d"].append(_f(d))
                e["seg"].append(_f(seg))
                e["null_pct"].append(pcts[1])
                e["swap_pct"].append(pcts)
                e["td"].append([[_f(x), _f(y)] for x, y in td])
                e["null_td"].append([[_f(x), _f(y)] for x, y in zip(nt, nd)])
                e["beats_null"].append(v["beats_null"])
                e["best_of_three"].append(v["best_of_three"])
            entry["groups"][g] = e
        entries.append(entry)
    return entries, missing


def summarize(entries, groups, n_frames: int) -> dict:
    """{set: {group: [set_summary per frame]}} over the triples present in that group."""
    out: dict = {}
    for s in dict.fromkeys(e["set"] for e in entries):
        out[s] = {}
        for g in groups:
            present = [e["groups"][g] for e in entries if e["set"] == s and "missing" not in e["groups"][g]]
            out[s][g] = [set_summary([{"beats_null": p["beats_null"][f], "best_of_three": p["best_of_three"][f]}
                                      for p in present]) for f in range(n_frames)]
    return out
