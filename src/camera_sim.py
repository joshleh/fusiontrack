"""
Simulated imaging detections: noisy pixel centers on a top-down orthographic map.

The camera does *not* run a real detector DNN here; we assume a perfect box
regressor and add only measurement noise and structured drop-outs to keep the
fusion story about the filter, not about learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, List, Optional

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from . import utils

# ---------------------------------------------------------------------------
# Pixel noise, miss, and occlusion schedule
# ---------------------------------------------------------------------------
# 1-sigma error on each pixel axis (independent Gaussians) — spec: ~8 px
CAMERA_CENTER_NOISE_STD_PX: Final[float] = 8.0
# P(no frame detection) when the target is visible — model label miss, score threshold
CAMERA_DETECTION_MISS_PROB: Final[float] = 0.10
# Inclusive frame index range [start, end] with *no* camera returns — simulates buildings/self-occlusion
CAMERA_OCCLUSION_START_FRAME: Final[int] = 40
CAMERA_OCCLUSION_END_FRAME: Final[int] = 60
# Synthetic full box size (pixels) — only used for a complete box record; center is what the EKF consumes
DEFAULT_BOX_WIDTH_PX: Final[float] = 48.0
DEFAULT_BOX_HEIGHT_PX: Final[float] = 36.0


@dataclass
class CameraPixelBox:
    """
    A bounding box in pixel space: center ``(u, v)`` plus size.

    *Why* this exists: portfolio code can log the full box even if the EKF only
    needs the center for a position-only measurement model.
    """

    u_center: float
    v_center: float
    width_px: float
    height_px: float


def camera_measurement_from_truth(
    true_x_m: float,
    true_y_m: float,
    frame_index: int,
    rng: Generator,
) -> Optional[CameraPixelBox]:
    """
    Draw a noisy pixel-space box from true world ``(x, y)``.

    Physical effects layered in order:
    1) **Occlusion window** — between :data:`CAMERA_OCCLUSION_START_FRAME` and
       :data:`CAMERA_OCCLUSION_END_FRAME`, we return ``None`` every frame so the
       fusion loop must use radar and coast.
    2) **Stochastic miss** — independent of occlusion, a Bernoulli miss models
       missed detections in clear view (thresholding, small target).
    3) **Gaussian center jitter** — models calibration error, discretization, and
       detector localisation noise. Applied in pixel space, then the inverse
       mapping in ``utils`` lifts to world coordinates.

    Parameters
    ----------
    true_x_m, true_y_m
        Ground-truth world position in meters of the *box center* (not an offset corner).
    frame_index
        Simulation frame, 0-based, used to gate occlusion.
    rng
        NumPy ``Generator`` for determinism in notebooks/tests.
    """
    if CAMERA_OCCLUSION_START_FRAME <= frame_index <= CAMERA_OCCLUSION_END_FRAME:
        return None
    if rng.random() < CAMERA_DETECTION_MISS_PROB:
        return None
    u0, v0 = utils.world_meters_to_pixel(true_x_m, true_y_m)
    # Independent Gaussian in each pixel direction around the true projection
    nu = float(rng.normal(0.0, CAMERA_CENTER_NOISE_STD_PX))
    nv = float(rng.normal(0.0, CAMERA_CENTER_NOISE_STD_PX))
    u, v = u0 + nu, v0 + nv
    return CameraPixelBox(
        u_center=u,
        v_center=v,
        width_px=DEFAULT_BOX_WIDTH_PX,
        height_px=DEFAULT_BOX_HEIGHT_PX,
    )


def run_camera_on_trajectory(
    true_xy: NDArray[np.float64],
    rng: Generator,
) -> List[Optional[CameraPixelBox]]:
    """
    Call :func:`camera_measurement_from_truth` for each row of a ``(N,2)`` path.
    """
    out: List[Optional[CameraPixelBox]] = []
    for k in range(int(true_xy.shape[0])):
        x, y = float(true_xy[k, 0]), float(true_xy[k, 1])
        out.append(camera_measurement_from_truth(x, y, k, rng))
    return out
