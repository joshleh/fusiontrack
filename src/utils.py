"""
Coordinate transforms and projection helpers for FusionTrack.

This module exists to keep a single, testable place for the assumptions that
turn pixels (what a camera actually measures) and polar radar returns into
the shared flat world frame used by the Kalman filter. In a real system these
mappings are calibration-heavy; here we use a simple orthographic top-down
model so the math pipeline is explicit and debuggable.
"""

from __future__ import annotations

import math
from typing import Final, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Geometric constants — orthographic "map" model (meters in world, pixels on screen)
# ---------------------------------------------------------------------------
# One pixel width/height in world space (meters). Scaling ground truth to pixels.
METERS_PER_PIXEL: Final[float] = 0.5
# Image principal point in pixels; top-left origin, u right, v down.
PRINCIPAL_U_PX: Final[float] = 400.0
PRINCIPAL_V_PX: Final[float] = 300.0


def polar_to_cartesian(
    range_m: float,
    azimuth_rad: float,
    *,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> Tuple[float, float]:
    """
    Convert a radar-style polar measurement to flat world (East-North) Cartesian.

    What this does: maps ``(range, azimuth)`` about a local origin to ``(x, y)``
    in the same 2D world frame as the EKF. Azimuth is measured from +x toward +y
    (mathematical CCW) so it matches the usual ENU-style convention in code.

    Parameters
    ----------
    range_m
        Slant or ground range from the origin to the object (meters), depending
        on your interpretation; here it is 2D ground range for a flat world.
    azimuth_rad
        Angle from the +x axis to the target, counter-clockwise (radians).
    origin_x, origin_y
        Radar/vehicle origin in world coordinates (meters). Non-zero if the
        sensor is not at the world origin; kept explicit for extensibility.
    """
    # Convert polar to Cartesian offset, then add sensor origin
    x = origin_x + range_m * math.cos(azimuth_rad)
    y = origin_y + range_m * math.sin(azimuth_rad)
    return x, y


def cartesian_to_polar(x: float, y: float, origin_x: float, origin_y: float) -> Tuple[float, float]:
    """
    Convert world Cartesian to polar about ``(origin_x, origin_y)``.

    This is the *noise-free* forward model used to synthesize range/azimuth
    from ground truth in the radar simulator, so polar noise matches geometry.
    """
    dx = x - origin_x
    dy = y - origin_y
    r = float(math.hypot(dx, dy))
    az = float(math.atan2(dy, dx))
    return r, az


def world_meters_to_pixel(u_m: float, v_m: float) -> Tuple[float, float]:
    """
    World ``(x, y)`` in meters to image pixel ``(u, v)`` for the top-down map.

    The mapping is linear with ``METERS_PER_PIXEL``, placing the world origin
    at ``(PRINCIPAL_U_PX, PRINCIPAL_V_PX)``. Increasing ``u`` is +x, increasing
    ``v`` is +y in this notebook-friendly convention (v down on screen, y
    down in the figures — consistent with the fusion plots).
    """
    u = PRINCIPAL_U_PX + u_m / METERS_PER_PIXEL
    v = PRINCIPAL_V_PX + v_m / METERS_PER_PIXEL
    return u, v


def pixel_to_world_meters(u_px: float, v_px: float) -> Tuple[float, float]:
    """
    Invert :func:`world_meters_to_pixel` to recover world coordinates from a
    noisy box center. Used before camera updates in the EKF.
    """
    u_m = (u_px - PRINCIPAL_U_PX) * METERS_PER_PIXEL
    v_m = (v_px - PRINCIPAL_V_PX) * METERS_PER_PIXEL
    return u_m, v_m


def pixel_noise_to_world_covariance(
    sigma_u_px: float,
    sigma_v_px: float,
) -> NDArray[np.float64]:
    """
    Build a 2x2 world-frame covariance from independent pixel standard deviations.

    Under the linear local mapping, scaling by ``METERS_PER_PIXEL`` takes pixel
    variance to world variance on each axis. This assumes pixel axes align with
    world axes (a reasonable local approximation for a nadir or nearly-nadir
    camera model).

    # INTERVIEW CRITICAL: Real projective cameras need a Jacobian from pixel to
    # ground, and R_world depends on altitude, lens distortion, and gimbal pose.
    """
    s = METERS_PER_PIXEL
    # Independent Gaussian noise in u and v -> diagonal covariance, scaled by s^2
    ru = (sigma_u_px * s) ** 2
    rv = (sigma_v_px * s) ** 2
    return np.array([[ru, 0.0], [0.0, rv]], dtype=np.float64)
