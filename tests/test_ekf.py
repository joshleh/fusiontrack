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


# ---------------------------------------------------------------------------
# EKFTracker tests
# ---------------------------------------------------------------------------


def test_ekf_tracker_predict_follows_cv_model() -> None:
    """EKFTracker prediction must follow the same linear F as KFTracker (shared CV dynamics)."""
    x0 = np.array([10.0, 5.0, 2.0, -1.0], dtype=np.float64)
    tr = ekf.EKFTracker(x0, dt=1.0)
    tr.predict()
    got = tr.get_state()
    assert got[0] == pytest.approx(12.0)   # x + vx*dt
    assert got[1] == pytest.approx(4.0)    # y + vy*dt
    assert got[2] == pytest.approx(2.0)    # vx unchanged
    assert got[3] == pytest.approx(-1.0)   # vy unchanged


def test_ekf_tracker_camera_update_moves_toward_z() -> None:
    """Camera (linear) update in EKFTracker must pull the state toward the measurement."""
    x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    tr = ekf.EKFTracker(x0, dt=1.0)
    z = np.array([20.0, 10.0], dtype=np.float64)
    before = tr.get_state()[:2].copy()
    tr.update_camera(z)
    after = tr.get_state()[:2]
    assert np.linalg.norm(after - z) < np.linalg.norm(before - z)


def test_ekf_tracker_radar_polar_moves_state_toward_truth() -> None:
    """
    A polar radar update must reduce the position error toward the measured target.

    Note: the state must not start at the radar origin because h(x) = [r, az] has
    an undefined Jacobian at r=0 (both rows collapse to zero, giving K=0 → no update).
    We start at an off-origin position that is still far from the truth.
    """
    import math
    # State at (30, 10) — off-origin, well away from truth at (100, 50)
    x0 = np.array([30.0, 10.0, 0.0, 0.0], dtype=np.float64)
    tr = ekf.EKFTracker(x0, dt=1.0)
    tr._kf.P = 1e4 * np.eye(4, dtype=np.float64)  # type: ignore[attr-defined]
    r_true = math.hypot(100.0, 50.0)
    az_true = math.atan2(50.0, 100.0)
    z = np.array([r_true, az_true], dtype=np.float64)
    before_err = np.linalg.norm(tr.get_state()[:2] - np.array([100.0, 50.0]))
    tr.update_radar_polar(z)
    after_err = np.linalg.norm(tr.get_state()[:2] - np.array([100.0, 50.0]))
    assert after_err < before_err


def test_ekf_tracker_radar_polar_angle_wrapping() -> None:
    """
    A target near azimuth ±π must not produce a ~2π unwrapped innovation.
    After the update, the state should remain bounded and not NaN.
    """
    # Place the filter state near (-r, 0) → az ≈ π - ε
    x0 = np.array([-50.0, 0.5, 0.0, 0.0], dtype=np.float64)
    tr = ekf.EKFTracker(x0, dt=1.0)
    # Measurement at the "other side" of π: azimuth = -π + 0.05 (equivalent to +π - 0.05)
    import math
    r_z = math.hypot(50.0, 0.5)
    az_z = -math.pi + 0.05  # just past the ±π seam
    z = np.array([r_z, az_z], dtype=np.float64)
    tr.update_radar_polar(z)
    state = tr.get_state()
    assert not np.any(np.isnan(state)), "State should not be NaN after near-pi update"
    assert np.all(np.abs(state) < 1e4), "State should not blow up due to unhandled angle wrapping"


def test_ekf_tracker_covariance_reduces_after_polar_update() -> None:
    """
    After a radar polar update with high initial P, the total covariance trace
    must shrink — same regression check as for KFTracker.
    """
    x0 = np.array([100.0, 80.0, 0.0, 0.0], dtype=np.float64)
    tr = ekf.EKFTracker(x0, dt=1.0)
    tr._kf.P = 1e4 * np.eye(4, dtype=np.float64)  # type: ignore[attr-defined]
    import math
    r_true = math.hypot(100.0, 80.0)
    az_true = math.atan2(80.0, 100.0)
    t_before = float(np.trace(tr.get_covariance()))
    tr.update_radar_polar(np.array([r_true, az_true]))
    t_after = float(np.trace(tr.get_covariance()))
    assert t_after < t_before


def test_ekf_tracker_covariance_psd_after_polar_update() -> None:
    """EKF posterior covariance must remain PSD after multiple predict+update cycles."""
    x0 = np.array([50.0, 30.0, 1.0, 0.5], dtype=np.float64)
    tr = ekf.EKFTracker(x0, dt=1.0)
    import math
    rng = np.random.default_rng(42)
    for _ in range(10):
        tr.predict()
        r = math.hypot(float(tr.get_state()[0]), float(tr.get_state()[1]))
        az = math.atan2(float(tr.get_state()[1]), float(tr.get_state()[0]))
        noise = rng.normal(0, 0.1, 2)
        tr.update_radar_polar(np.array([r + noise[0], az + noise[1]]))
    p = tr.get_covariance()
    assert np.allclose(p, p.T, atol=1e-9)
    w = np.linalg.eigvalsh(0.5 * (p + p.T))
    assert np.min(w) >= -1e-6


def test_ekf_tracker_ellipse_center_matches_state() -> None:
    """EKFTracker uncertainty ellipse center must match the state position."""
    x0 = np.array([7.0, -4.0, 0.0, 0.0], dtype=np.float64)
    tr = ekf.EKFTracker(x0)
    ell = tr.get_uncertainty_ellipse()
    st = tr.get_state()
    assert ell.center[0] == pytest.approx(float(st[0]))
    assert ell.center[1] == pytest.approx(float(st[1]))
    assert ell.width > 0 and ell.height > 0
