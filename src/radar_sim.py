"""
Synthetic range–azimuth radar returns with missed detections, polar noise, and false alarms.

This file exists to feed the fusion engine with *physically named* error sources.
It does not model multipath, range folding, Doppler, or any clutter model beyond
a simple i.i.d. false-alarm event — those can be added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, List, Optional, Tuple

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from . import utils

# ---------------------------------------------------------------------------
# Physical / statistical sensor parameters
# ---------------------------------------------------------------------------
# 1-sigma error on range *after* the sensor's internal tracking (meters) —
# INTERVIEW CRITICAL: real systems split thermal noise, range gate straddle, etc.
RADAR_RANGE_NOISE_STD_M: Final[float] = 3.0
# 1-sigma error on *angle* in radians (0.5°) — at long range, cross-range = r * σ_θ
RADAR_AZIMUTH_NOISE_STD_RAD: Final[float] = float(np.radians(0.5))
# P(no detection) when the target is line-of-sight: models multipath / beam shape / SNR fades.
RADAR_DETECTION_MISS_PROB: Final[float] = 0.15
# P(false contact this frame) when a real target is present: ghost / clutter spike near truth.
# INTERVIEW CRITICAL: a real MHT would spawn hypotheses; the single-Bernoulli EKF
# we use is intentionally naive for pedagogy and may diverge on bad gates.
RADAR_FALSE_ALARM_PROB: Final[float] = 0.05
# When a false alarm fires, we offset truth with these meters (broadly "nearby")
# for a spurious (r, az) pair, so the EKF is stressed but not sent to a random
# field corner every time.
FALSE_ALARM_MAX_OFFSET_M: Final[float] = 40.0
# Radar origin in world frame (meters) — the sensor is co-located with the world origin
# for the demo; a moving platform would time-vary this.
RADAR_ORIGIN_X_M: Final[float] = 0.0
RADAR_ORIGIN_Y_M: Final[float] = 0.0


@dataclass
class PolarRadarReturn:
    """
    A single polar sensor frame.

    *range_m* is ground (or slant) range; *azimuth_rad* is CCW from +x, matching ``utils.polar_to_cartesian``.
    """

    range_m: float
    azimuth_rad: float
    is_from_true_target: bool
    is_false_alarm: bool


def radar_measurement_from_true_position(
    true_x: float,
    true_y: float,
    rng: Generator,
) -> Optional[PolarRadarReturn]:
    """
    Draw one radar *measurement* for a true target, or return ``None`` on miss.

    What this simulates, in order:
    1) **Missed detection** (sn-radar, beam loss): no return, so fusion must coast.
    2) **False alarm** (clutter, birds): a nearby ghost return that is not a simple
       Gaussian "second peak" in the MHT sense — for this single-object demo, we
       return *one* measurement that is intentionally biased when the FA fires.
    3) **Normal hit**: add Gaussian noise in range and in azimuth *before* the
       Cartesian hand-off in fusion.

    Each parameter in the function signature is explicit about physical meaning; see module constants.

    Parameters
    ----------
    true_x, true_y
        True target world position in meters, same frame as the KF.
    rng
        NumPy ``Generator`` for reproducible sequences.
    """
    # 1) Swerling / SNR miss — no measurement this frame
    if rng.random() < RADAR_DETECTION_MISS_PROB:
        return None
    r_true, az_true = utils.cartesian_to_polar(
        true_x, true_y, RADAR_ORIGIN_X_M, RADAR_ORIGIN_Y_M
    )
    # 2) False alarm — treat as a spurious (r, az) near the *truth* in Cartesian space, then
    # convert back to polar; keeps targets "nearby" as requested
    is_fa = rng.random() < RADAR_FALSE_ALARM_PROB
    if is_fa:
        jx, jy = false_alarm_jitter_2d(rng)
        x_g = true_x + jx
        y_g = true_y + jy
        r_noisy, az_noisy = utils.cartesian_to_polar(x_g, y_g, RADAR_ORIGIN_X_M, RADAR_ORIGIN_Y_M)
        return PolarRadarReturn(
            range_m=float(r_noisy), azimuth_rad=float(az_noisy), is_from_true_target=False, is_false_alarm=True
        )
    # 3) Standard hit — add independent Gaussian in sensors-native polar
    n_r = rng.normal(0.0, RADAR_RANGE_NOISE_STD_M)
    n_az = rng.normal(0.0, RADAR_AZIMUTH_NOISE_STD_RAD)
    r_noisy = max(1.0, r_true + n_r)  # avoid negative range at the origin
    az_noisy = az_true + n_az
    return PolarRadarReturn(
        range_m=float(r_noisy), azimuth_rad=float(az_noisy), is_from_true_target=True, is_false_alarm=False
    )


def false_alarm_jitter_2d(rng: Generator) -> Tuple[float, float]:
    """
    A bounded uniform offset in x/y for a false contact (meters), sized by
    :data:`FALSE_ALARM_MAX_OFFSET_M` so the blip is "local" in the world frame.
    """
    jx = rng.uniform(-FALSE_ALARM_MAX_OFFSET_M, FALSE_ALARM_MAX_OFFSET_M)
    jy = rng.uniform(-FALSE_ALARM_MAX_OFFSET_M, FALSE_ALARM_MAX_OFFSET_M)
    return float(jx), float(jy)


def polar_to_world_xy(
    r: PolarRadarReturn,
) -> Tuple[float, float]:
    """
    Hand polar returns to the Cartesian world for the Kalman update path.
    """
    return utils.polar_to_cartesian(
        r.range_m, r.azimuth_rad, origin_x=RADAR_ORIGIN_X_M, origin_y=RADAR_ORIGIN_Y_M
    )


@dataclass
class TrackObservations:
    """
    A batch container used by the fusion loop: per-frame *lists* to keep the door
    open for later multi-returns (not used for association yet).
    """

    polars: List[Optional[PolarRadarReturn]]


def run_radar_on_trajectory(
    true_xy: NDArray[np.float64],
    rng: Generator,
) -> List[Optional[PolarRadarReturn]]:
    """
    Run :func:`radar_measurement_from_true_position` for every row of a ``(N,2)`` path.
    """
    out: List[Optional[PolarRadarReturn]] = []
    for i in range(true_xy.shape[0]):
        x, y = float(true_xy[i, 0]), float(true_xy[i, 1])
        out.append(radar_measurement_from_true_position(x, y, rng))
    return out
