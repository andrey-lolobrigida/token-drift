"""v2 timeline maths on toy arrays."""
import numpy as np
import pytest

from token_drift import timeline as tl


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_identical_geometry_is_fully_finished():
    x = _unit(np.random.default_rng(0).normal(size=(60, 8)))
    assert tl.overlap_with_final(x, x, k=5) == pytest.approx(1.0)
    assert tl.cka_with_final(x, x) == pytest.approx(1.0)


def test_a_rotation_counts_as_finished_too():
    # both metrics only care about the shape of the cloud, not which way it faces
    rng = np.random.default_rng(1)
    x = _unit(rng.normal(size=(60, 8)))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert tl.overlap_with_final(x @ q, x, k=5) == pytest.approx(1.0)
    assert tl.cka_with_final(x @ q, x) == pytest.approx(1.0)


def test_unrelated_geometry_is_far_from_finished():
    rng = np.random.default_rng(0)
    a, b = _unit(rng.normal(size=(300, 16))), _unit(rng.normal(size=(300, 16)))
    assert tl.overlap_with_final(a, b, k=5) < 0.1  # chance is ~0.01
    assert tl.cka_with_final(a, b) < 0.3


def test_row_drift_is_zero_for_untouched_rows_and_relative_for_moved_ones():
    rng = np.random.default_rng(0)
    w0 = rng.normal(size=(10, 4)).astype(np.float32)
    wt = w0.copy()
    wt[3] = w0[3] * 1.5  # moved by half its own length
    wt[7] = 0            # shrunk to nothing: drift 1
    d = tl.row_drift(wt, w0)
    assert d.shape == (10,) and d.dtype == np.float32
    assert d[3] == pytest.approx(0.5, rel=1e-5) and d[7] == pytest.approx(1.0)
    assert (d[[0, 1, 2, 4, 5, 6, 8, 9]] == 0).all()


def test_row_drift_takes_float16_like_weights_on_disk():
    w0 = np.array([[3.0, 4.0]], dtype=np.float16)
    wt = np.array([[3.0, 4.5]], dtype=np.float16)
    assert tl.row_drift(wt, w0)[0] == pytest.approx(0.1, rel=1e-3)  # 0.5 / 5


def test_row_drift_of_a_zero_init_row_is_nan_not_inf():
    w0 = np.array([[0.0, 0.0], [3.0, 4.0]])
    wt = np.array([[1.0, 0.0], [3.0, 4.0]])
    d = tl.row_drift(wt, w0)
    assert np.isnan(d[0]) and d[1] == 0


def test_median_by_bin_hand_checked():
    values = np.array([1.0, 5.0, 3.0, 10.0, 20.0, np.nan, 7.0])
    bins = np.array([0, 0, 0, 1, 1, 2, -1])  # -1 = no bin, ignored
    out = tl.median_by_bin(values, bins, n_bins=4)
    assert out[:2] == [3.0, 15.0]
    assert np.isnan(out[2]) and np.isnan(out[3])  # bin 2 holds only a NaN, bin 3 is empty


def test_half_way_step_rising_falling_and_flat():
    steps = [0, 1, 8, 64]
    assert tl.half_way_step([0.0, 0.1, 0.6, 1.0], steps) == 8
    assert tl.half_way_step([1.0, 0.8, 0.4, 0.0], steps) == 8  # falling curves count the same way
    assert tl.half_way_step([0.0, 0.5, 0.6, 1.0], steps) == 1  # exactly half counts as reached
    assert tl.half_way_step([0.3, 0.3, 0.3, 0.3], steps) is None  # nothing happened
