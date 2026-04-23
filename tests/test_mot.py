"""
Tests for :mod:`src.mot`: TrackerManager lifecycle, gating, and association.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.mot import (
    CHI2_GATE_2DOF_99,
    HITS_TO_CONFIRM,
    MAX_MISSES_BEFORE_DELETE,
    Track,
    TrackState,
    TrackerManager,
    run_mot_demo,
)
from src import ekf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _polar(x_w: float, y_w: float) -> np.ndarray:
    """World position → noiseless polar measurement."""
    return np.array([math.hypot(x_w, y_w), math.atan2(y_w, x_w)], dtype=np.float64)


def _fresh_manager(**kwargs) -> TrackerManager:
    return TrackerManager(dt=1.0, **kwargs)


# ---------------------------------------------------------------------------
# Birth and lifecycle
# ---------------------------------------------------------------------------


def test_measurement_births_tentative_track() -> None:
    """An unmatched measurement must create exactly one TENTATIVE track."""
    mgr = _fresh_manager()
    mgr.predict_all()
    mgr.update([_polar(100.0, 50.0)])
    assert len(mgr.tracks) == 1
    assert mgr.tracks[0].state == TrackState.TENTATIVE
    assert len(mgr.get_confirmed_tracks()) == 0


def test_two_hits_confirm_track() -> None:
    """
    HITS_TO_CONFIRM=2 consecutive measurements at the same position must confirm
    the track — the first hit births it (tentative), the second confirms it.
    """
    mgr = _fresh_manager(hits_to_confirm=2)
    z = _polar(200.0, 100.0)
    # Frame 1 — birth
    mgr.predict_all()
    mgr.update([z])
    assert mgr.tracks[0].state == TrackState.TENTATIVE

    # Frame 2 — confirm
    mgr.predict_all()
    mgr.update([z])
    assert mgr.tracks[0].state == TrackState.CONFIRMED
    assert len(mgr.get_confirmed_tracks()) == 1


def test_max_misses_deletes_confirmed_track() -> None:
    """
    A confirmed track receives MAX_MISSES_BEFORE_DELETE consecutive empty frames
    and must be deleted (removed from the list entirely).
    """
    mgr = _fresh_manager(hits_to_confirm=2, max_misses=3)
    z = _polar(150.0, 80.0)
    # Confirm the track first
    for _ in range(3):
        mgr.predict_all()
        mgr.update([z])
    assert mgr.tracks[0].state == TrackState.CONFIRMED

    # Now starve it
    for _ in range(3):
        mgr.predict_all()
        mgr.update([])
    assert len(mgr.tracks) == 0


def test_tentative_track_pruned_if_no_second_hit() -> None:
    """
    A tentative track that never gets a second measurement must be pruned after
    MAX_TENTATIVE_AGE frames — it was a clutter false alarm, not a target.
    """
    mgr = _fresh_manager(hits_to_confirm=2)
    mgr.predict_all()
    mgr.update([_polar(300.0, 200.0)])   # birth
    for _ in range(5):                   # no more measurements
        mgr.predict_all()
        mgr.update([])
    assert len(mgr.tracks) == 0, "Stale tentative track should have been pruned"


def test_no_measurements_all_tracks_miss() -> None:
    """Passing an empty measurement list must increment miss count on every track."""
    mgr = _fresh_manager(hits_to_confirm=2)
    z = _polar(100.0, 100.0)
    # Confirm the track
    for _ in range(3):
        mgr.predict_all()
        mgr.update([z])
    assert mgr.tracks[0].state == TrackState.CONFIRMED
    initial_misses = mgr.tracks[0].misses

    mgr.predict_all()
    mgr.update([])
    assert mgr.tracks[0].misses == initial_misses + 1


# ---------------------------------------------------------------------------
# Cost matrix and gating
# ---------------------------------------------------------------------------


def test_cost_matrix_shape() -> None:
    """Cost matrix must be (n_tracks × n_meas)."""
    mgr = _fresh_manager(hits_to_confirm=99)  # keep tracks tentative
    # Birth 2 tracks
    mgr.predict_all()
    mgr.update([_polar(100.0, 50.0), _polar(300.0, 200.0)])

    assert len(mgr.tracks) == 2
    meas = [_polar(105.0, 55.0), _polar(305.0, 205.0), _polar(400.0, 400.0)]
    mgr.predict_all()
    cost = mgr._build_cost_matrix(meas)
    assert cost.shape == (2, 3)


def test_far_measurement_gated_out() -> None:
    """
    A measurement 300 m from the track's predicted position must have Mahalanobis
    distance well above the chi-square gate and must not be assigned.
    """
    mgr = _fresh_manager(hits_to_confirm=99)
    close_z = _polar(100.0, 100.0)
    far_z = _polar(400.0, 400.0)   # ~424 m away from track

    # Birth and let the close measurement confirm
    for _ in range(3):
        mgr.predict_all()
        mgr.update([close_z])

    assert len(mgr.tracks) == 1
    mgr.predict_all()
    cost = mgr._build_cost_matrix([far_z])
    # The far measurement must be gated out (cost left at _LARGE_COST)
    from src.mot import _LARGE_COST
    assert cost[0, 0] >= _LARGE_COST, "Far measurement must not be inside the gate"


def test_track_ids_are_unique() -> None:
    """Every track in one manager must have a distinct ID."""
    mgr = _fresh_manager(hits_to_confirm=99)
    zs = [_polar(x, 50.0) for x in [100.0, 200.0, 300.0]]
    mgr.predict_all()
    mgr.update(zs)
    ids = [t.track_id for t in mgr.tracks]
    assert len(ids) == len(set(ids)), "Track IDs must be unique"


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_mot_demo_produces_confirmed_tracks() -> None:
    """
    Full 100-frame crossing demo must yield at least two confirmed tracks —
    a basic end-to-end regression that the association and lifecycle logic
    don't catastrophically fail on a realistic multi-target scenario.
    """
    res = run_mot_demo(rng=np.random.default_rng(42), n_frames=100)
    # Collect all unique track IDs that appear in history
    all_ids = set()
    for frame in res["track_history"]:
        all_ids.update(frame.keys())
    assert len(all_ids) >= 2, "At least 2 distinct tracks should be formed over 100 frames"


def test_mot_demo_track_history_length() -> None:
    """track_history must have exactly n_frames entries."""
    res = run_mot_demo(rng=np.random.default_rng(7), n_frames=50)
    assert len(res["track_history"]) == 50


def test_mahalanobis_sq_finite_for_well_conditioned_state() -> None:
    """
    For a track well away from the radar origin and a nearby measurement,
    the Mahalanobis distance must be finite and non-negative.
    """
    x0 = np.array([100.0, 80.0, 2.0, 1.0], dtype=np.float64)
    r_polar = np.diag([
        ekf.RADAR_RANGE_NOISE_STD_EKF_M ** 2,
        ekf.RADAR_AZIMUTH_NOISE_STD_EKF_RAD ** 2,
    ]).astype(np.float64)
    track = Track(x0, track_id=99, dt=1.0, r_radar_polar=r_polar)
    track.predict()   # advance to predicted state

    z_near = _polar(102.0, 81.0)
    d2 = TrackerManager._mahalanobis_sq(track, z_near)
    assert d2 is not None
    assert np.isfinite(d2)
    assert d2 >= 0.0
