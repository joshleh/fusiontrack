"""
End-to-end single-target fusion: synthetic trajectory, parallel simulators, four tracks.

Four parallel trackers are compared:

* **KF fused** — linear KF, radar converted to Cartesian before update (old path)
* **EKF fused** — ExtendedKalmanFilter, radar in native (r, θ) with analytic Jacobian
* **Camera-only** — linear KF, camera measurements only
* **Radar-only** — linear KF, Cartesian radar only (for degraded baseline)

The EKF vs KF comparison is the primary takeaway for the portfolio.
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

    The routine wires up four trackers:

    * ``fused`` — :class:`ekf.KFTracker`, camera + Cartesian radar (baseline)
    * ``ekf_fused`` — :class:`ekf.EKFTracker`, camera + polar radar (the new path)
    * ``cam_kf`` — :class:`ekf.KFTracker`, camera-only baseline
    * ``rad_kf`` — :class:`ekf.KFTracker`, Cartesian radar-only baseline

    The returned structure is a plain *serializable-friendly* map with NumPy
    arrays so notebooks can introspect the same data as the CLI demo.
    """
    rng = rng or default_rng(DEMO_RNG_SEED)
    true_xy = _make_smooth_curved_trajectory(TRAJECTORY_NUM_FRAMES)
    radar_list = radar_sim.run_radar_on_trajectory(true_xy, rng)
    cam_list = camera_sim.run_camera_on_trajectory(true_xy, rng)

    # R_camera: from pixel 1-σ; R_radar_cart: tighter diagonal world covariance (KF path)
    r_cam = utils.pixel_noise_to_world_covariance(
        camera_sim.CAMERA_CENTER_NOISE_STD_PX,
        camera_sim.CAMERA_CENTER_NOISE_STD_PX,
    )
    r_radar = np.eye(2, dtype=np.float64) * RADAR_MEASUREMENT_VAR_M2
    # EKF polar R: diag([σ_r², σ_θ²]) — built from physical constants, no tuning needed
    r_radar_polar = np.diag(
        [ekf.RADAR_RANGE_NOISE_STD_EKF_M ** 2, ekf.RADAR_AZIMUTH_NOISE_STD_EKF_RAD ** 2]
    ).astype(np.float64)

    x0 = _initial_state_from_ground_truth(true_xy, FUSION_FRAME_DT_S, rng)
    # Linear KF with Cartesian radar (existing baseline)
    fused = ekf.KFTracker(
        x0, dt=FUSION_FRAME_DT_S, r_camera=r_cam, r_radar=r_radar
    )
    # EKF with polar radar (new — the main upgrade)
    ekf_fused = ekf.EKFTracker(
        x0, dt=FUSION_FRAME_DT_S, r_camera=r_cam, r_radar_polar=r_radar_polar
    )
    cam_kf = ekf.KFTracker(
        x0, dt=FUSION_FRAME_DT_S, r_camera=r_cam, r_radar=r_radar
    )
    rad_kf = ekf.KFTracker(
        x0, dt=FUSION_FRAME_DT_S, r_camera=r_cam, r_radar=r_radar
    )

    n = TRAJECTORY_NUM_FRAMES
    kf_fused_path = np.zeros((n, 2), dtype=np.float64)
    ekf_fused_path = np.zeros((n, 2), dtype=np.float64)
    cam_path = np.zeros((n, 2), dtype=np.float64)
    rad_path = np.zeros((n, 2), dtype=np.float64)
    kf_trace_p = np.zeros(n, dtype=np.float64)
    ekf_trace_p = np.zeros(n, dtype=np.float64)
    cam_meas_world = np.full((n, 2), np.nan, dtype=np.float64)
    rad_meas_world = np.full((n, 2), np.nan, dtype=np.float64)
    # Ellipses stored during the loop — one set per tracker, no second RNG pass
    kf_ellipses: List[Tuple[int, ekf.UncertaintyEllipse2D]] = []
    ekf_ellipses: List[Tuple[int, ekf.UncertaintyEllipse2D]] = []

    for k in range(n):
        # Time update: all four filters advance one step
        fused.predict()
        ekf_fused.predict()
        cam_kf.predict()
        rad_kf.predict()

        # Decode camera measurement
        z_cam: Optional[NDArray[np.float64]] = None
        if cam_list[k] is not None:
            px = cam_list[k]
            u, v = px.u_center, px.v_center
            wx, wy = utils.pixel_to_world_meters(u, v)
            z_cam = np.array([wx, wy], dtype=np.float64)
            cam_meas_world[k, :] = z_cam

        # Decode radar measurement (Cartesian for KF, polar for EKF)
        z_rad_cart: Optional[NDArray[np.float64]] = None
        z_rad_polar: Optional[NDArray[np.float64]] = None
        if radar_list[k] is not None:
            ret = radar_list[k]
            zx, zy = radar_sim.polar_to_world_xy(ret)
            z_rad_cart = np.array([zx, zy], dtype=np.float64)
            z_rad_polar = np.array([ret.range_m, ret.azimuth_rad], dtype=np.float64)
            rad_meas_world[k, :] = z_rad_cart

        # --- Linear KF fused update (Cartesian radar) ---
        # INTERVIEW CRITICAL: two sequential KF updates at one time step is not the same
        # as a single joint update with stacked H/R unless measurements are conditionally
        # independent given the state — acceptable for a tutorial; production uses gating.
        if z_cam is not None:
            fused.update_camera(z_cam, r_override=r_cam)
        if z_rad_cart is not None:
            fused.update_radar(z_rad_cart, r_override=r_radar)

        # --- EKF fused update (polar radar, linear camera) ---
        if z_cam is not None:
            ekf_fused.update_camera(z_cam, r_override=r_cam)
        if z_rad_polar is not None:
            ekf_fused.update_radar_polar(z_rad_polar)

        # --- Single-modality baselines ---
        if z_cam is not None:
            cam_kf.update_camera(z_cam, r_override=r_cam)
        if z_rad_cart is not None:
            rad_kf.update_radar(z_rad_cart, r_override=r_radar)

        kf_fused_path[k, :] = fused.get_state()[:2]
        ekf_fused_path[k, :] = ekf_fused.get_state()[:2]
        cam_path[k, :] = cam_kf.get_state()[:2]
        rad_path[k, :] = rad_kf.get_state()[:2]
        kf_trace_p[k] = float(np.trace(fused.get_covariance()))
        ekf_trace_p[k] = float(np.trace(ekf_fused.get_covariance()))
        if k % ELLIPSE_FRAME_STEP == 0:
            kf_ellipses.append((k, fused.get_uncertainty_ellipse()))
            ekf_ellipses.append((k, ekf_fused.get_uncertainty_ellipse()))

    return {
        "ground_truth": true_xy,
        # KF fused (Cartesian radar) — kept for comparison
        "fused": kf_fused_path,
        "cov_trace": kf_trace_p,
        "uncertainty_ellipses": kf_ellipses,
        # EKF fused (polar radar) — new
        "ekf_fused": ekf_fused_path,
        "ekf_cov_trace": ekf_trace_p,
        "ekf_uncertainty_ellipses": ekf_ellipses,
        # Single-modality baselines
        "camera_only": cam_path,
        "radar_only": rad_path,
        # Raw measurements for debugging / notebook
        "camera_pixel_centers": cam_list,
        "radar_polars": radar_list,
        "camera_world_measurements": cam_meas_world,
        "radar_world_measurements": rad_meas_world,
        "r_camera_2d": r_cam,
        "r_radar_2d": r_radar,
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

    ekf_fused = res.get("ekf_fused")

    fig1, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        true_xy[:, 0],
        true_xy[:, 1],
        color="0.1",
        linewidth=2.0,
        label="Ground truth",
    )
    ax.plot(
        fused[:, 0], fused[:, 1], color="C0", linewidth=1.2,
        linestyle="--", label="KF fused (Cartesian radar)",
    )
    if ekf_fused is not None:
        ax.plot(
            ekf_fused[:, 0], ekf_fused[:, 1], color="C3", linewidth=1.6,
            label="EKF fused (polar radar)",
        )
    ax.plot(
        c_only[:, 0], c_only[:, 1],
        color="C1", linestyle=":", linewidth=1.0, label="Camera-only KF",
    )
    ax.plot(
        r_only[:, 0], r_only[:, 1],
        color="C2", linestyle=":", linewidth=1.0, label="Radar-only KF",
    )
    # EKF ellipses (preferred); fall back to KF ellipses
    ellipse_data = res.get("ekf_uncertainty_ellipses") or res.get("uncertainty_ellipses", [])
    ell_color = "C3" if res.get("ekf_uncertainty_ellipses") else "C0"
    for _frame, ell in ellipse_data:
        e = Ellipse(
            ell.center,
            width=ell.width,
            height=ell.height,
            angle=ell.angle_deg,
            facecolor="none",
            edgecolor=ell_color,
            linewidth=0.8,
            alpha=0.7,
        )
        ax.add_patch(e)
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_title("EKF (polar radar) vs. KF (Cartesian radar) vs. single-modality")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Error panel — per-frame L2, all four tracks
    err_kf = _per_frame_l2(true_xy, fused)
    err_c = _per_frame_l2(true_xy, c_only)
    err_r = _per_frame_l2(true_xy, r_only)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    t = np.arange(n)
    ax2.plot(t, err_kf, color="C0", linestyle="--", linewidth=1.2, label="KF fused (Cartesian)")
    if ekf_fused is not None:
        err_ekf = _per_frame_l2(true_xy, ekf_fused)
        ax2.plot(t, err_ekf, color="C3", linewidth=1.6, label="EKF fused (polar)")
    ax2.plot(t, err_c, color="C1", linestyle=":", label="Camera-only")
    ax2.plot(t, err_r, color="C2", linestyle=":", label="Radar-only")
    ax2.axvspan(
        PLOT_OCCLUSION_START,
        PLOT_OCCLUSION_END,
        color="0.4",
        alpha=0.15,
        label="Camera occlusion (radar only)",
    )
    ax2.set_xlabel("Frame index k")
    ax2.set_ylabel("Position error (m)")
    ax2.set_title("Per-frame L2 error — EKF vs. KF")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    if show and "agg" not in str(plt.get_backend()).lower():
        plt.show()
    return fig1, fig2


if __name__ == "__main__":
    out = run_fusion_demo()
    plot_results(out, show=True)
