"""Q16 maths on hand-built points: t, d, seg, nulls, role swaps, verdicts."""
import numpy as np
import pytest

from token_drift import betweenness as bt

A, B = np.array([0.0, 0, 0]), np.array([2.0, 0, 0])


def test_segment_stats_midpoint_and_beyond():
    t, d, seg = bt.segment_stats(A, np.array([1.0, 0, 0]), B)
    assert (t, d, seg) == pytest.approx((0.5, 0, 0))
    t, d, seg = bt.segment_stats(A, np.array([3.0, 0, 0]), B)  # on the line, past b
    assert t == pytest.approx(1.5) and d == pytest.approx(0) and seg == pytest.approx(0.5)
    t, d, seg = bt.segment_stats(A, np.array([1.0, 1, 0]), B)
    assert (t, d, seg) == pytest.approx((0.5, 0.5, 0.5))


def test_segment_stats_ignores_shared_offset_and_scale():
    rng = np.random.default_rng(0)
    a, v, b = rng.normal(size=(3, 8))
    shift = rng.normal(size=8)
    for x, y in zip(bt.segment_stats(a, v, b), bt.segment_stats(3 * a + shift, 3 * v + shift, 3 * b + shift)):
        assert x == pytest.approx(y)


def test_segment_stats_broadcasts_over_many_v():
    vs = np.array([[1.0, 0, 0], [3.0, 0, 0]])
    t, d, seg = bt.segment_stats(A, vs, B)
    assert t.tolist() == pytest.approx([0.5, 1.5]) and seg.tolist() == pytest.approx([0, 0.5])


def test_unit_mean_points_counts_and_self_sim():
    occ = np.array([[2.0, 0], [0, 3.0], [5.0, 0], [0, 1.0]], dtype=np.float16)
    pts, counts, ss = bt.unit_mean(occ, np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]), n_words=2, n_groups=2)
    assert counts.tolist() == [[2, 0], [1, 1]]
    np.testing.assert_allclose(pts[0, 0], [0.5, 0.5])  # mean of unit vectors, not re-normed
    assert np.isnan(pts[0, 1]).all()
    assert ss[0, 0] == pytest.approx(0.0)  # two orthogonal occurrences
    assert np.isnan(ss[1, 0])  # one occurrence: no pairs


def test_null_percentile():
    nulls = np.array([[1.0, 0.5, 0], [1.0, 1, 0], [5.0, 0, 0], [1.0, 0.05, 0]])
    # real seg 0.1; null segs 0.25, 0.5, 1.5, 0.025 -> one of four is at least as close
    assert bt.null_percentile(A, B, np.array([1.0, 0.2, 0]), nulls) == pytest.approx(25.0)


TRIP = np.array([[0.0, 0, 0], [1.0, 0.05, 0], [2.0, 0, 0]])  # deficiency, mean, excess
NULLS = np.array([[1, 0.5, 0], [1.5, 0.3, 0], [0.5, 0.4, 0], [1, -0.6, 0], [1.2, 0.2, 0.1], [0.3, 0.25, 0]])


def test_role_swap_and_verdict_mean_between():
    pcts = bt.role_swap(TRIP, [NULLS] * 3)
    assert pcts[1] == 0.0 and pcts[0] > 0 and pcts[2] > 0
    assert bt.triple_verdict(pcts, between_pct=5) == {"beats_null": True, "best_of_three": True}


def test_verdict_ties_are_not_best():
    assert bt.triple_verdict([0.0, 0.0, 50.0], 5) == {"beats_null": True, "best_of_three": False}


def test_set_summary_binomial():
    vs = [{"beats_null": True, "best_of_three": True}] * 5 + [{"beats_null": False, "best_of_three": False}]
    s = bt.set_summary(vs)
    assert (s["n"], s["beats_null"], s["best_of_three"]) == (6, 5, 5)
    assert s["p_best_of_three"] == pytest.approx(13 / 729)  # P(X >= 5), X ~ Bin(6, 1/3)
    assert bt.set_summary([])["p_best_of_three"] is None


def test_occurrence_stats_unit_norms_each_occurrence():
    a, b = np.array([1.0, 0]), np.array([0.0, 1])
    t, d, seg = bt.occurrence_stats(np.array([[10.0, 10.0], [3.0, 0.0]]), a, b)
    assert t.tolist() == pytest.approx([0.5, 0.0]) and seg[1] == pytest.approx(0.0)


def test_nearest_in_log_count():
    assert bt.nearest_in_log_count(50, {"a": 45, "b": 55, "c": 5000, "d": 60, "e": 40}, 3) == ["b", "a", "d"]


# ---------- run_q16 ----------

WORDS = ["cowardice", "courage", "rashness"] + [f"n{i}" for i in range(6)] + ["timidity"]


def _setup(found=None):
    w_index = {w: i for i, w in enumerate(WORDS)}
    P = np.zeros((len(WORDS), 1, 3), dtype=np.float32)
    P[:3, 0] = TRIP
    P[3:9, 0] = NULLS
    counts = np.full((len(WORDS), 1), 30)
    counts[w_index["timidity"], 0] = 2
    found = found or {w: {"books": 100} for w in WORDS}
    triples = [
        {"id": "c/fear/noun/a", "set": "c", "concept": "fear", "pos": "noun", "words": ("cowardice", "courage", "rashness")},
        {"id": "c/fear/noun/b", "set": "c", "concept": "fear", "pos": "noun", "words": ("courage", "cowardice", "rashness")},
        {"id": "c/fear/noun/c", "set": "c", "concept": "fear", "pos": "noun", "words": ("timidity", "courage", "rashness")},
    ]
    pool = {"noun": [f"n{i}" for i in range(6)], "adj": []}
    return [P, P], counts, w_index, triples, pool, found


def test_run_q16_verdicts_missing_and_summary():
    points, counts, w_index, triples, pool, found = _setup()
    entries, missing = bt.run_q16(points, counts, w_index, triples, pool, found, groups=("books",),
                                  min_count=20, null_k=6, between_pct=5)
    good, swapped, short = (e["groups"]["books"] for e in entries)
    assert good["beats_null"] == [True, True] and good["best_of_three"] == [True, True]
    assert good["t"][0] == pytest.approx(0.5) and len(good["null_td"][0]) == 6 and len(good["td"][0]) == 3
    assert swapped["beats_null"] == [False, False] and swapped["best_of_three"] == [False, False]
    assert short == {"missing": ["timidity"]}
    assert missing == [{"id": "c/fear/noun/c", "group": "books", "short": ["timidity"]}]
    s = bt.summarize(entries, ("books",), n_frames=2)
    assert s["c"]["books"][0]["n"] == 2 and s["c"]["books"][0]["best_of_three"] == 1


def test_run_q16_null_neighbours_use_found_counts_not_capped_counts():
    # Review Focus 4: kept counts are all capped (30 here), found counts are not
    found = {w: {"books": 50} for w in WORDS}
    found.update({"n0": {"books": 45}, "n1": {"books": 55}, "n2": {"books": 5000},
                  "n3": {"books": 9000}, "n4": {"books": 60}, "n5": {"books": 40}})
    points, counts, w_index, triples, pool, _ = _setup()
    entries, _ = bt.run_q16(points, counts, w_index, triples[:1], pool, found, groups=("books",),
                            min_count=20, null_k=3, between_pct=5)
    assert set(entries[0]["groups"]["books"]["null_words"]) == {"n0", "n1", "n4"}


def test_run_q16_too_few_null_words_raises():
    points, counts, w_index, triples, pool, found = _setup()
    with pytest.raises(ValueError, match="null pool"):
        bt.run_q16(points, counts, w_index, triples[:1], pool, found, groups=("books",),
                   min_count=20, null_k=7, between_pct=5)


# ---------- F2: degenerate triples (two words share a last token piece -> same frame-0 point) ----------

def test_segment_stats_with_a_equal_b_is_nan_without_a_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t, d, seg = bt.segment_stats(A, np.array([1.0, 0, 0]), A)
        assert np.isnan(t) and np.isnan(d) and np.isnan(seg)
        ts = bt.segment_stats(A, np.array([[1.0, 0, 0], [2.0, 0, 0]]), A)
        assert all(np.isnan(x).all() and x.shape == (2,) for x in ts)
        assert np.isnan(bt.null_percentile(A, A, np.array([1.0, 0, 0]), NULLS))


def test_run_q16_degenerate_frame_has_no_verdict_and_is_not_counted():
    points, counts, w_index, triples, pool, found = _setup()
    P0 = points[0].copy()
    P0[w_index["courage"], 0] = P0[w_index["cowardice"], 0]  # virtue == deficiency at frame 0 only
    entries, _ = bt.run_q16([P0, points[1]], counts, w_index, triples[:2], pool, found, groups=("books",),
                            min_count=20, null_k=6, between_pct=5)
    good = entries[0]["groups"]["books"]
    assert good["beats_null"] == [None, True] and good["best_of_three"] == [None, True]
    assert good["null_pct"][0] is None and good["swap_pct"][0] == [None, None, None]
    s = bt.summarize(entries, ("books",), n_frames=2)
    # triple b is the same three words in another order, so it's degenerate at frame 0 too
    assert entries[1]["groups"]["books"]["beats_null"][0] is None
    assert s["c"]["books"][0]["n"] == 0 and s["c"]["books"][1]["n"] == 2


def test_run_q16_records_null_counts_in_null_words_order():
    points, counts, w_index, triples, pool, found = _setup()
    counts = counts.copy()
    counts[w_index["n2"], 0] = 25
    entries, _ = bt.run_q16(points, counts, w_index, triples[:1], pool, found, groups=("books",),
                            min_count=20, null_k=6, between_pct=5)
    e = entries[0]["groups"]["books"]
    assert e["null_counts"] == [25 if w == "n2" else 30 for w in e["null_words"]]
