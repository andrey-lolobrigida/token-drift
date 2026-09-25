"""Per-layer centering / unit-norm / all-but-the-top. Functions take arrays, return arrays."""
import numpy as np
import pytest

from token_drift.normalize import normalize_layer, normalize_all


@pytest.fixture
def x():
    rng = np.random.default_rng(0)
    # 200-token slice, d=16, deliberately off-center and anisotropic
    base = rng.normal(size=(200, 16)).astype(np.float16)
    base[:, 0] *= 5.0  # one high-variance direction, like a "massive activation" dim
    base[:, 1] += 3.0  # and an offset so centering has something to do
    return base


def test_center_removes_mean(x):
    out = normalize_layer(x, center=True, unit_norm=False, drop_top_pcs=0)
    assert out.dtype == np.float32  # upcast per layer; float16 stays on disk only
    np.testing.assert_allclose(out.mean(0), 0.0, atol=1e-5)


def test_unit_norm_gives_unit_rows(x):
    out = normalize_layer(x, center=False, unit_norm=True, drop_top_pcs=0)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_drop_top_pcs_removes_dominant_direction(x):
    out = normalize_layer(x, center=True, unit_norm=False, drop_top_pcs=1)
    # after dropping PC1 the variance along the old dominant axis should collapse
    assert out[:, 0].var() < 0.05 * x.astype(np.float32)[:, 0].var()
    # and it must not silently drop everything
    assert out.var() > 0


def test_drop_top_pcs_then_unit_norm_order(x):
    # centering -> drop PCs -> unit norm: rows are unit even after projection
    out = normalize_layer(x, center=True, unit_norm=True, drop_top_pcs=2)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_zero_row_does_not_produce_nan():
    x = np.zeros((3, 4), dtype=np.float32)
    x[0] = [1, 0, 0, 0]
    out = normalize_layer(x, center=False, unit_norm=True, drop_top_pcs=0)
    assert np.isfinite(out).all()


def test_normalize_all_is_per_layer_and_keeps_shape(x):
    stack = np.stack([x, x * 2 + 1], axis=0)  # (2, 200, 16)
    out = normalize_all(stack, center=True, unit_norm=True, drop_top_pcs=0)
    assert out.shape == stack.shape
    assert out.dtype == np.float16  # back to float16 for disk
    a = normalize_layer(x, center=True, unit_norm=True, drop_top_pcs=0)
    np.testing.assert_allclose(out[0].astype(np.float32), a, atol=2e-3)
    # layer 1 is an affine map of layer 0, and centering+unit-norm removes exactly that
    np.testing.assert_allclose(out[1].astype(np.float32), a, atol=2e-3)


def test_row_norm_first_off_is_unchanged(x):
    a = normalize_layer(x, center=True, unit_norm=True, drop_top_pcs=0)
    b = normalize_layer(x, center=True, unit_norm=True, drop_top_pcs=0, row_norm_first=False)
    np.testing.assert_array_equal(a, b)


def test_row_norm_first_still_gives_unit_rows(x):
    out = normalize_layer(x, center=True, unit_norm=True, drop_top_pcs=0, row_norm_first=True)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_row_norm_first_recovers_neighbours_hidden_by_a_varying_shared_direction():
    # Pythia's pre-LN L6 in miniature: token-specific part r plus one shared direction u
    # in a token-dependent amount a. Center-then-unit-norm turns the spread in a into
    # direction differences and loses r's neighbourhoods; normalizing rows first doesn't.
    from token_drift.metrics import knn_indices, knn_overlap

    rng = np.random.default_rng(0)
    r = rng.normal(size=(300, 32))
    u = rng.normal(size=32)
    u /= np.linalg.norm(u)
    a = rng.uniform(20, 60, size=300)
    x = (r + a[:, None] * u).astype(np.float32)
    truth = knn_indices(normalize_layer(r, center=True, unit_norm=True, drop_top_pcs=0), 10)
    kw = dict(center=True, unit_norm=True, drop_top_pcs=0)
    off = knn_overlap(truth, knn_indices(normalize_layer(x, **kw, row_norm_first=False), 10))
    on = knn_overlap(truth, knn_indices(normalize_layer(x, **kw, row_norm_first=True), 10))
    assert off < 0.3 and on > 0.6


def test_normalize_all_passes_row_norm_first(x):
    stack = np.stack([x], axis=0)
    out = normalize_all(stack, center=True, unit_norm=True, drop_top_pcs=0, row_norm_first=True)
    a = normalize_layer(x, center=True, unit_norm=True, drop_top_pcs=0, row_norm_first=True)
    np.testing.assert_allclose(out[0].astype(np.float32), a, atol=2e-3)


def test_fit_rows_centers_on_those_rows_only():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 8)) + 3
    x[:10] = 0  # ten never-seen tokens: zero rows
    fit = np.ones(50, bool)
    fit[:10] = False
    out = normalize_layer(x, center=True, unit_norm=False, drop_top_pcs=0, fit_rows=fit)
    np.testing.assert_allclose(out[fit].mean(0), 0, atol=1e-5)
    # without the mask the zero rows drag the mean toward 0 and real rows keep an offset
    out_all = normalize_layer(x, center=True, unit_norm=False, drop_top_pcs=0)
    assert np.abs(out_all[fit].mean(0)).max() > 0.3


def test_fit_rows_pcs_come_from_those_rows():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 8))
    x[:20, 0] *= 50  # junk rows stretched along axis 0; real rows aren't
    fit = np.ones(60, bool)
    fit[:20] = False
    out = normalize_layer(x, center=True, unit_norm=False, drop_top_pcs=1, fit_rows=fit)
    ref = normalize_layer(x[fit], center=True, unit_norm=False, drop_top_pcs=1)
    np.testing.assert_allclose(out[fit], ref, atol=1e-5)


def test_normalize_all_passes_fit_rows_to_every_frame():
    rng = np.random.default_rng(0)
    acts = rng.normal(size=(3, 40, 8)).astype(np.float16)
    fit = np.arange(40) >= 5
    out = normalize_all(acts, center=True, unit_norm=True, drop_top_pcs=0, fit_rows=fit)
    for i in range(3):
        ref = normalize_layer(acts[i], center=True, unit_norm=True, drop_top_pcs=0, fit_rows=fit)
        np.testing.assert_allclose(out[i].astype(np.float32), ref, atol=2e-3)
