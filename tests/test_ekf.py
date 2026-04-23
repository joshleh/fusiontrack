"""
Unit tests for :mod:`src.ekf` predict/update behavior and basic invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import ekf


def test_get_state_length_and_get_covariance_shape() -> None:
    """State is always 4D; covariance is 4x4 and consistent with the filter state."""
    tr = ekf.KFTracker(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64))
    assert tr.get_state().shape == (4,)
    assert tr.get_covariance().shape == (4, 4)


def test_predict_advances_state_under_constant_velocity_model() -> None:
    """
    The filter mean must follow the discrete constant-velocity transition F: for ``dt=1``,
    ``x <- x + vx``, ``y <- y + vy`` when only ``predict`` is applied (mean path does not
    use Q — Q adds covariance, not mean bias, in the linear KF).
    """
    x0 = np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float64)
    tr = ekf.KFTracker(x0, dt=1.0)
    tr.predict()
    got = tr.get_state()
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(0.5)
    assert got[2] == pytest.approx(1.0)
    assert got[3] == pytest.approx(0.5)


def test_update_camera_moves_toward_measurement() -> None:
    """
    A position update with reasonable default (P, R) must reduce plain L2 error
    to the measurement (batch case from origin).
    """
    x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    tr = ekf.KFTracker(x0, dt=1.0)
    z = np.array([10.0, 5.0], dtype=np.float64)
    before = tr.get_state()[:2].copy()
    tr.update_camera(z)
    after = tr.get_state()[:2]
    assert np.linalg.norm(after - z) < np.linalg.norm(before - z)


def test_update_reduces_total_covariance_trace_after_high_initial_uncertainty() -> None:
    """
    When P starts huge relative to R, a position update *typically* shrinks the total
    trace — a healthy regression check that the KF is actually fusing, not no-op.
    (Trace is not a cost in general Kalman theory; this is a sufficient sanity test.)
    """
    x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    tr = ekf.KFTracker(x0, dt=1.0)
    tr._kf.P = 1.0e4 * np.eye(4)  # type: ignore[attr-defined]  # isolate the fusion step
    t_before = float(np.trace(tr.get_covariance()))
    tr.update_camera(np.array([1.0, 2.0], dtype=np.float64))
    t_after = float(np.trace(tr.get_covariance()))
    assert t_after < t_before


def test_repeated_update_with_same_z_monotonic_move_toward_z() -> None:
    """
    Idempotent re-measurements should continue to **reduce** the gap to a fixed
    (consistent) report — here we check the second nudge is smaller in L2 to ``z``.
    """
    tr = ekf.KFTracker(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64))
    z = np.array([3.0, 4.0], dtype=np.float64)
    tr._kf.P = 50.0 * np.eye(4)  # type: ignore[attr-defined]
    tr.update_camera(z)
    a1 = tr.get_state()[:2].copy()
    tr.update_camera(z)
    a2 = tr.get_state()[:2]
    assert np.linalg.norm(a2 - z) < np.linalg.norm(a1 - z) + 1e-9


def test_covariance_is_symmetric_positive_semidefinite() -> None:
    p = ekf.KFTracker(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)).get_covariance()
    assert np.allclose(p, p.T)
    w = np.linalg.eigvalsh(0.5 * (p + p.T))  # Hermitian for numerical safety
    assert np.min(w) >= -1e-6


def test_uncertainty_ellipse_center_matches_state() -> None:
    tr = ekf.KFTracker(np.array([3.0, -2.0, 0.0, 0.0], dtype=np.float64))
    ell = tr.get_uncertainty_ellipse()
    st = tr.get_state()
    assert ell.center[0] == pytest.approx(float(st[0]))
    assert ell.center[1] == pytest.approx(float(st[1]))
    assert ell.width > 0 and ell.height > 0
