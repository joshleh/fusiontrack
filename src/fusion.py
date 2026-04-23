"""
End-to-end single-target fusion: synthetic trajectory, parallel simulators, three KF tracks.

This module is the *glue* the portfolio story rests on. It exists so a reviewer
can run one command and see measurement gaps, process noise, and covariance
evolution in one place without any API or deployment plumbing.
"""

from __future__ import annotations

from typing import Any, Dict, Final, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from numpy.random import default_rng
from numpy.typing import NDArray

from . import camera_sim, ekf, radar_sim, utils

# ---------------------------------------------------------------------------
# Session / script constants — all tunable simulation knobs live here
# ---------------------------------------------------------------------------
# Length of the synthetic run (frames); matches portfolio brief.
TRAJECTORY_NUM_FRAMES: Final[int] = 100
# Time between frames (seconds) — filter, radar, and camera share the same clock.
FUSION_FRAME_DT_S: Final[float] = ekf.DEFAULT_DT_S
# World box for the path (meters) — UAS in a 500m local tangent patch.
TRAJECTORY_WORLD_MIN_M: Final[float] = 0.0
TRAJECTORY_WORLD_MAX_M: Final[float] = 500.0
# Path parameter sweep for the smooth curve: controls how many "lobes" appear.
TRAJECTORY_SIN_SCALE_RAD: Final[float] = 0.12
# Initial velocity error (m/s) on top of a finite-difference trim from the first
# two ground-truth samples — *cold-start* is intentionally imperfect so early frames
# show real convergence; values like 0.1 m/s were too oracle-like.
INIT_VELOCITY_ERROR_STD: Final[float] = 2.0
# Random seed for a repeatable demo; change for new Monte-Carlo runs.
DEMO_RNG_SEED: Final[int] = 7
# Frame indices at which to draw error ellipses for clarity
ELLIPSE_FRAME_STEP: Final[int] = 10
# Shaded occlusion window for the RMSE panel (inclusive) — same as camera_sim
PLOT_OCCLUSION_START: Final[int] = camera_sim.CAMERA_OCCLUSION_START_FRAME
PLOT_OCCLUSION_END: Final[int] = camera_sim.CAMERA_OCCLUSION_END_FRAME
# Per-axis world variance for radar (m²). 2.0 m 1-σ → 4.0 m², **tighter** than
# camera: ~4 m 1-σ (from 8 px × 0.5 m/px) → 16 m² — so fusion can lean on range/azimuth
# when the camera is noisy or missing. (Real R is state-dependent after polar; this is a knob.)
# INTERVIEW: if camera and radar R were numerically equal, you would not have a *sensor*
# story — you would be testing geometry only.
RADAR_MEASUREMENT_VAR_M2: Final[float] = 2.0**2
# Camera: map pixel σ to a full 2x2 R via utils (8 px 1-σ in each direction)


def _make_smooth_curved_trajectory(n_frames: int) -> NDArray[np.float64]:
    """
    Build a 2D smooth path in ``[0, 500]`` m by sampling a parametric wave so the
    track is *not* a random walk, matching the brief for a *curved* UAV line.

    The function uses a sine phase along the x baseline plus a forward drift
    in y so the whole path stays in-bounds. This is *not* a Dubins model — it
    is a controllable, interview-friendly S-curve.
    """
    t = np.arange(n_frames, dtype=np.float64)
    # Normalized abscissa in (0,1) for the whole run
    s = t / max(n_frames - 1, 1)
    # A gentle easting sweep across most of the box
    x = 40.0 + 400.0 * s
    # Sinusoidal offset + mild northing drift, clipped to stay in world limits
    y = 80.0 + 140.0 * np.sin(TRAJECTORY_SIN_SCALE_RAD * t) + 1.1 * t
    x = np.clip(x, TRAJECTORY_WORLD_MIN_M + 5.0, TRAJECTORY_WORLD_MAX_M - 5.0)
    y = np.clip(y, TRAJECTORY_WORLD_MIN_M + 5.0, TRAJECTORY_WORLD_MAX_M - 5.0)
    return np.column_stack([x, y])


def _initial_state_from_ground_truth(
    true_xy: NDArray[np.float64], dt: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """
    Seed ``[x, y, vx, vy]`` from the first two ground-truth samples so the very
    first velocity is physically aligned with the synthetic path, plus
    :data:`INIT_VELOCITY_ERROR_STD` to simulate a non-oracle handoff from a
    first-stage tracker or operator cue.
    """
    p0 = true_xy[0, :2]
    p1 = true_xy[1, :2]
    v0 = (p1 - p0) / float(dt) + rng.normal(0.0, INIT_VELOCITY_ERROR_STD, size=2)
    return np.array([p0[0], p0[1], v0[0], v0[1]], dtype=np.float64)


def run_fusion_demo(
    *,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Run the *full* single-target fusion experiment and return a dict for plotting/JSON.

    The routine wires up three :class:`ekf.KFTracker` objects:
    *fused* (camera+radar when available), *camera-only*, and *radar-only*.
    Each time step applies ``predict()``, then zero, one, or two measurement
    updates depending on simulator output.

    The returned structure is a plain *serializable-friendly* map with NumPy
    arrays so notebooks can introspect the same data as the CLI demo.
    """
    rng = rng or default_rng(DEMO_RNG_SEED)
    true_xy = _make_smooth_curved_trajectory(TRAJECTORY_NUM_FRAMES)
    radar_list = radar_sim.run_radar_on_trajectory(true_xy, rng)
    cam_list = camera_sim.run_camera_on_trajectory(true_xy, rng)

    # R_camera: from pixel 1-σ; R_radar: tighter (see module constants above)
    r_cam = utils.pixel_noise_to_world_covariance(
        camera_sim.CAMERA_CENTER_NOISE_STD_PX,
        camera_sim.CAMERA_CENTER_NOISE_STD_PX,
    )
    r_radar = np.eye(2, dtype=np.float64) * RADAR_MEASUREMENT_VAR_M2

    x0 = _initial_state_from_ground_truth(true_xy, FUSION_FRAME_DT_S, rng)
    fused = ekf.KFTracker(
        x0, dt=FUSION_FRAME_DT_S, r_camera=r_cam, r_radar=r_radar
    )
    cam_kf = ekf.KFTracker(
        x0, dt=FUSION_FRAME_DT_S, r_camera=r_cam, r_radar=r_radar
    )
    rad_kf = ekf.KFTracker(
        x0, dt=FUSION_FRAME_DT_S, r_camera=r_cam, r_radar=r_radar
    )

    n = TRAJECTORY_NUM_FRAMES
    fused_path = np.zeros((n, 2), dtype=np.float64)
    cam_path = np.zeros((n, 2), dtype=np.float64)
    rad_path = np.zeros((n, 2), dtype=np.float64)
    trace_p = np.zeros(n, dtype=np.float64)  # scalar uncertainty proxy: tr(P)
    cam_meas_world = np.full((n, 2), np.nan, dtype=np.float64)  # for debugging/plots
    rad_meas_world = np.full((n, 2), np.nan, dtype=np.float64)
    # Precompute ellipse geometry for the trajectory plot (no second RNG pass)
    uncertainty_ellipses: List[Tuple[int, ekf.UncertaintyEllipse2D]] = []
    for k in range(n):
        # time update for all three filters first
        fused.predict()
        cam_kf.predict()
        rad_kf.predict()

        z_cam: Optional[NDArray[np.float64]] = None
        if cam_list[k] is not None:
            px = cam_list[k]
            u, v = px.u_center, px.v_center
            wx, wy = utils.pixel_to_world_meters(u, v)
            z_cam = np.array([wx, wy], dtype=np.float64)
            cam_meas_world[k, :] = z_cam
        z_rad: Optional[NDArray[np.float64]] = None
        if radar_list[k] is not None:
            zx, zy = radar_sim.polar_to_world_xy(radar_list[k])
            z_rad = np.array([zx, zy], dtype=np.float64)
            rad_meas_world[k, :] = z_rad
        # Fusion rules: feed both sensors in sequence when they exist; order: camera, radar.
        # INTERVIEW CRITICAL: two sequential KF updates at one time step is not the same as a
        # single joint update with stacked H/R unless measurements are conditionally independent
        # in the right order — fine for a tutorial; production systems use joint M-H or gating.
        if z_cam is not None and z_rad is not None:
            fused.update_camera(z_cam, r_override=r_cam)
            fused.update_radar(z_rad, r_override=r_radar)
        elif z_cam is not None:
            fused.update_camera(z_cam, r_override=r_cam)
        elif z_rad is not None:
            fused.update_radar(z_rad, r_override=r_radar)
        if z_cam is not None:
            cam_kf.update_camera(z_cam, r_override=r_cam)
        if z_rad is not None:
            rad_kf.update_radar(z_rad, r_override=r_radar)

        fused_path[k, :] = fused.get_state()[:2]
        cam_path[k, :] = cam_kf.get_state()[:2]
        rad_path[k, :] = rad_kf.get_state()[:2]
        p_full = fused.get_covariance()
        trace_p[k] = float(np.trace(p_full))
        if k % ELLIPSE_FRAME_STEP == 0:
            # Snapshot after the measurement stage so the ellipse reflects the posterior
            uncertainty_ellipses.append((k, fused.get_uncertainty_ellipse()))

    return {
        "ground_truth": true_xy,
        "fused": fused_path,
        "camera_only": cam_path,
        "radar_only": rad_path,
        "cov_trace": trace_p,
        "camera_pixel_centers": cam_list,  # list of objects / None
        "radar_polars": radar_list,
        "camera_world_measurements": cam_meas_world,
        "radar_world_measurements": rad_meas_world,
        "r_camera_2d": r_cam,
        "r_radar_2d": r_radar,
        "uncertainty_ellipses": uncertainty_ellipses,
    }


def _per_frame_l2(
    true_xy: NDArray[np.float64], est: NDArray[np.float64]
) -> NDArray[np.float64]:
    d = true_xy - est
    return np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)


def plot_results(
    res: Dict[str, Any], *, show: bool = True
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create the two *portfolio* figures: trajectory+ellipses and RMSE with occlusion.
    """
    true_xy = res["ground_truth"][:, :2]
    fused = res["fused"]
    c_only = res["camera_only"]
    r_only = res["radar_only"]
    n = true_xy.shape[0]

    fig1, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        true_xy[:, 0],
        true_xy[:, 1],
        color="0.1",
        linewidth=2.0,
        label="Ground truth",
    )
    ax.plot(
        fused[:, 0], fused[:, 1], color="C0", linewidth=1.5, label="Fused (camera+radar)"
    )
    ax.plot(
        c_only[:, 0],
        c_only[:, 1],
        color="C1",
        linestyle="--",
        linewidth=1.2,
        label="Camera-only KF",
    )
    ax.plot(
        r_only[:, 0],
        r_only[:, 1],
        color="C2",
        linestyle=":",
        linewidth=1.2,
        label="Radar-only KF",
    )
    for _frame, ell in res.get("uncertainty_ellipses", []):
        e = Ellipse(
            ell.center,
            width=ell.width,
            height=ell.height,
            angle=ell.angle_deg,
            facecolor="none",
            edgecolor="C0",
            linewidth=0.8,
            alpha=0.7,
        )
        ax.add_patch(e)
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_title("Fused track vs. single-modality filters")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE panel — per-frame L2 as instantaneous error, not a sliding mean
    err_f = _per_frame_l2(true_xy, fused)
    err_c = _per_frame_l2(true_xy, c_only)
    err_r = _per_frame_l2(true_xy, r_only)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    t = np.arange(n)
    ax2.plot(t, err_f, color="C0", label="Fused")
    ax2.plot(t, err_c, color="C1", linestyle="--", label="Camera-only")
    ax2.plot(t, err_r, color="C2", linestyle=":", label="Radar-only")
    ax2.axvspan(
        PLOT_OCCLUSION_START,
        PLOT_OCCLUSION_END,
        color="0.4",
        alpha=0.15,
        label="Camera occlusion (radar only)",
    )
    ax2.set_xlabel("Frame index k")
    ax2.set_ylabel("Position error (m) — L2 to ground truth")
    ax2.set_title("Per-frame error — fusion reduces peak error in occlusion window")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    if show and "agg" not in str(plt.get_backend()).lower():
        plt.show()
    return fig1, fig2


if __name__ == "__main__":
    out = run_fusion_demo()
    plot_results(out, show=True)
