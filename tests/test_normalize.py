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
