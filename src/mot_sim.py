"""
Multi-target scenario generator for FusionTrack MOT.

Produces N synthetic constant-velocity targets plus Poisson clutter,
returning shuffled per-frame polar measurement lists that the tracker
must associate without being told which return came from which target.
"""

from __future__ import annotations

import math
from typing import Final, List

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from . import radar_sim

WORLD_SIZE_M: Final[float] = 500.0


def make_cv_trajectory(
    start_xy: NDArray[np.float64],
    velocity_xy: NDArray[np.float64],
    n_frames: int,
) -> NDArray[np.float64]:
    """Straight constant-velocity path, clipped to ``[0, WORLD_SIZE_M]``."""
    t = np.arange(n_frames, dtype=np.float64)[:, None]
    xy = start_xy[None, :] + velocity_xy[None, :] * t
    return np.clip(xy, 0.0, WORLD_SIZE_M)


def make_crossing_scenario(n_frames: int = 100) -> List[NDArray[np.float64]]:
    """
    Three targets on straight CV paths that converge near the scene centre around frame 50.

    Target 1 (→): (50, 225) at 4.0 m/s east — reaches ~(245, 225) at k=49.
    Target 2 (↗): (225, 50) at (0.4, 3.4) m/s — reaches ~(245, 216) at k=49.
    Target 3 (←): (450, 290) at (-4.0, -0.5) m/s — reaches ~(254, 265) at k=49.

    Targets 1 & 2 pass within ~9 m of each other at frame 49, which is the hardest
    data-association moment: Mahalanobis gating and Hungarian assignment are stressed.
    Target 3 is ~40 m away from the pair at that moment — easily distinguishable.

    # INTERVIEW CRITICAL: at the crossing, two tracks are inside each other's chi-square
    # gate; the Hungarian algorithm makes the globally optimal assignment. JPDA or MHT
    # would maintain *distributions* over hypotheses instead of committing to one.
    """
    return [
        make_cv_trajectory(np.array([50.0, 225.0]), np.array([4.0, 0.0]), n_frames),
        make_cv_trajectory(np.array([225.0, 50.0]), np.array([0.4, 3.4]), n_frames),
        make_cv_trajectory(np.array([450.0, 290.0]), np.array([-4.0, -0.5]), n_frames),
    ]


def generate_multi_target_measurements(
    trajectories: List[NDArray[np.float64]],
    rng: Generator,
    *,
    clutter_rate: float = 0.5,
) -> List[List[radar_sim.PolarRadarReturn]]:
    """
    For every frame: draw radar returns from each target (may miss), add Poisson
    clutter, then **shuffle** — the tracker receives no ordering information.

    Parameters
    ----------
    trajectories
        List of ``(n_frames, 2)`` world-coordinate arrays.
    rng
        Shared RNG for scene determinism.
    clutter_rate
        Mean number of spurious false-alarm returns per frame (Poisson).
        A value of 0.5 means ~1 clutter blip every 2 frames on average.
    """
    n_frames = trajectories[0].shape[0]
    per_frame: List[List[radar_sim.PolarRadarReturn]] = []

    for k in range(n_frames):
        frame_meas: List[radar_sim.PolarRadarReturn] = []

        # True target returns (independent miss per target per frame)
        for traj in trajectories:
            ret = radar_sim.radar_measurement_from_true_position(
                float(traj[k, 0]), float(traj[k, 1]), rng
            )
            if ret is not None:
                frame_meas.append(ret)

        # Poisson clutter — random positions in the surveillance volume
        n_clutter = int(rng.poisson(clutter_rate))
        for _ in range(n_clutter):
            r_c = float(rng.uniform(10.0, 450.0))
            az_c = float(rng.uniform(-math.pi, math.pi))
            frame_meas.append(
                radar_sim.PolarRadarReturn(
                    range_m=r_c,
                    azimuth_rad=az_c,
                    is_from_true_target=False,
                    is_false_alarm=True,
                )
            )

        # Shuffle: real radar gives no target-ordered list
        indices = rng.permutation(len(frame_meas))
        frame_meas = [frame_meas[i] for i in indices]
        per_frame.append(frame_meas)

    return per_frame
