"""
Unit tests for :mod:`src.ekf` predict/update behavior and basic invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import ekf


def test_predict_advances_state_under_constant_velocity_model() -> None:
    """
    The filter mean must follow the discrete constant-velocity transition F: for ``dt=1``,
    ``x <- x + vx``, ``y <- y + vy`` when only ``predict`` is applied (mean path does not
    use Q — Q adds covariance, not mean bias, in the linear KF).
    """
    x0 = np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float64)
    tr = ekf.EKFTracker(x0, dt=1.0)
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
    tr = ekf.EKFTracker(x0, dt=1.0)
    z = np.array([10.0, 5.0], dtype=np.float64)
    before = tr.get_state()[:2].copy()
    tr.update_camera(z)
    after = tr.get_state()[:2]
    assert np.linalg.norm(after - z) < np.linalg.norm(before - z)


def test_uncertainty_ellipse_center_matches_state() -> None:
    tr = ekf.EKFTracker(np.array([3.0, -2.0, 0.0, 0.0], dtype=np.float64))
    ell = tr.get_uncertainty_ellipse()
    st = tr.get_state()
    assert ell.center[0] == pytest.approx(float(st[0]))
    assert ell.center[1] == pytest.approx(float(st[1]))
    assert ell.width > 0 and ell.height > 0
