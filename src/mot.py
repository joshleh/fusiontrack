"""
Multi-Object Tracker (MOT) for 2D radar targets.

Architecture (Global Nearest-Neighbour):
  1. Predict — propagate all tracks one time step.
  2. Gate — compute Mahalanobis distance in polar measurement space.
  3. Assign — Hungarian algorithm on the gated cost matrix.
  4. Update — EKF update for matched (track, measurement) pairs.
  5. Manage — birth from unmatched measurements, miss/death from unmatched tracks.

# INTERVIEW CRITICAL: GNN commits to one assignment hypothesis per frame.
# JPDA computes soft assignment weights; MHT maintains a tree of hypotheses.
# GNN is O(n³) per frame via Hungarian; JPDA and MHT grow exponentially in
# clutter density without pruning.  For low-clutter, moderate-target-count
# scenarios (like UAV surveillance), GNN is production-deployable.
"""

from __future__ import annotations

import math
from enum import Enum, auto
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from numpy.random import default_rng
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from . import ekf, mot_sim

# ---------------------------------------------------------------------------
# Track lifecycle constants
# ---------------------------------------------------------------------------
# Gate threshold: chi-square with 2 DOF at 99% → 9.21.
# Measurements with Mahalanobis distance² > this are not eligible for any track.
# INTERVIEW: loosening the gate reduces missed associations at the cost of more
# clutter ingestion; tightening it rejects bad measurements but risks track loss.
CHI2_GATE_2DOF_99: Final[float] = 9.210

HITS_TO_CONFIRM: Final[int] = 2
MAX_MISSES_BEFORE_DELETE: Final[int] = 3
MAX_TENTATIVE_AGE: Final[int] = 3

# Sentinel cost for gated (infeasible) assignments so Hungarian still runs cleanly.
_LARGE_COST: Final[float] = 1e9

# Ellipse snapshot cadence (frames)
ELLIPSE_FRAME_STEP: Final[int] = 10

# Max distance (m) to call a (track, GT) pair a valid match when computing metrics
MATCH_GATE_M: Final[float] = 50.0


class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    DELETED = auto()


class Track:
    """
    Single target track backed by an :class:`ekf.EKFTracker`.

    Track IDs are assigned by :class:`TrackerManager` (not a class counter)
    so they remain unique even across multiple manager instances in tests.
    """

    def __init__(
        self,
        initial_state: NDArray[np.float64],
        track_id: int,
        *,
        dt: float,
        r_radar_polar: NDArray[np.float64],
    ) -> None:
        self.track_id: int = track_id
        self.state: TrackState = TrackState.TENTATIVE
        self.hits: int = 1
        self.misses: int = 0
        self.age: int = 0
        self._tracker: ekf.EKFTracker = ekf.EKFTracker(
            initial_state, dt=dt, r_radar_polar=r_radar_polar
        )

    def predict(self) -> None:
        self._tracker.predict()
        self.age += 1

    def update(self, z_polar: NDArray[np.float64]) -> None:
        self._tracker.update_radar_polar(z_polar)
        self.hits += 1
        self.misses = 0

    def miss(self) -> None:
        self.misses += 1

    def get_position(self) -> NDArray[np.float64]:
        return self._tracker.get_state()[:2]

    def get_uncertainty_ellipse(self) -> ekf.UncertaintyEllipse2D:
        return self._tracker.get_uncertainty_ellipse()

    def compute_innovation_polar(
        self, z_polar: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Delegate to EKFTracker for gating; returns (y, S)."""
        return self._tracker.compute_innovation_polar(z_polar)


class TrackerManager:
    """
    Global Nearest-Neighbour multi-object tracker over polar radar measurements.

    Typical usage per frame::

        manager.predict_all()
        zs = [np.array([ret.range_m, ret.azimuth_rad]) for ret in frame_returns]
        manager.update(zs)
        confirmed = manager.get_confirmed_tracks()
    """

    def __init__(
        self,
        *,
        dt: float = 1.0,
        hits_to_confirm: int = HITS_TO_CONFIRM,
        max_misses: int = MAX_MISSES_BEFORE_DELETE,
        gate_chi2: float = CHI2_GATE_2DOF_99,
        r_radar_polar: Optional[NDArray[np.float64]] = None,
    ) -> None:
        self._dt = dt
        self._hits_to_confirm = hits_to_confirm
        self._max_misses = max_misses
        self._gate_chi2 = gate_chi2
        if r_radar_polar is None:
            r_radar_polar = np.diag(
                [ekf.RADAR_RANGE_NOISE_STD_EKF_M ** 2, ekf.RADAR_AZIMUTH_NOISE_STD_EKF_RAD ** 2]
            ).astype(np.float64)
        self._r_radar_polar: NDArray[np.float64] = r_radar_polar
        self.tracks: List[Track] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Per-frame API
    # ------------------------------------------------------------------

    def predict_all(self) -> None:
        for t in self.tracks:
            t.predict()

    def update(self, measurements: Sequence[NDArray[np.float64]]) -> None:
        """
        Associate measurements to existing tracks, then manage lifecycle.

        Parameters
        ----------
        measurements
            Sequence of ``[range_m, azimuth_rad]`` polar arrays (no ordering info).
        """
        measurements = list(measurements)
        n_tracks = len(self.tracks)
        n_meas = len(measurements)

        if n_tracks == 0:
            for z in measurements:
                self._birth(z)
            return
        if n_meas == 0:
            for t in self.tracks:
                t.miss()
            self._apply_lifecycle_rules()
            return

        # Build cost matrix and run Hungarian
        cost = self._build_cost_matrix(measurements)
        row_ind, col_ind = linear_sum_assignment(cost)

        # Accept only within-gate assignments
        assigned_tracks: set = set()
        assigned_meas: set = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < _LARGE_COST:
                self.tracks[r].update(measurements[c])
                assigned_tracks.add(r)
                assigned_meas.add(c)

        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.miss()

        for j, z in enumerate(measurements):
            if j not in assigned_meas:
                self._birth(z)

        self._apply_lifecycle_rules()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

    def get_all_tracks(self) -> List[Track]:
        return list(self.tracks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cost_matrix(
        self, measurements: List[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        """
        (n_tracks × n_meas) Mahalanobis cost matrix.

        Entry [i, j] = d²(track_i, meas_j) if d² ≤ gate_chi2 else _LARGE_COST.
        Gating here prevents the Hungarian from wasting capacity on infeasible pairs.
        # INTERVIEW CRITICAL: without pre-gating, a dense clutter field fills every
        # row with valid costs and the Hungarian may steal a measurement from a
        # confirmed track to serve a false-alarm-born tentative.
        """
        cost = np.full(
            (len(self.tracks), len(measurements)), fill_value=_LARGE_COST, dtype=np.float64
        )
        for i, track in enumerate(self.tracks):
            for j, z in enumerate(measurements):
                d2 = self._mahalanobis_sq(track, z)
                if d2 is not None and d2 <= self._gate_chi2:
                    cost[i, j] = d2
        return cost

    @staticmethod
    def _mahalanobis_sq(
        track: Track, z_polar: NDArray[np.float64]
    ) -> Optional[float]:
        """d² = y^T S^{-1} y in polar space (angle-normalised). None if S is singular."""
        try:
            y, S = track.compute_innovation_polar(z_polar)
            y_flat = y.ravel()
            return float(y_flat @ np.linalg.inv(S) @ y_flat)
        except np.linalg.LinAlgError:
            return None

    def _birth(self, z_polar: NDArray[np.float64]) -> None:
        """Spawn TENTATIVE track from unmatched measurement. Velocity initialised to zero."""
        r, az = float(z_polar[0]), float(z_polar[1])
        x0 = np.array([r * math.cos(az), r * math.sin(az), 0.0, 0.0], dtype=np.float64)
        t = Track(x0, self._next_id, dt=self._dt, r_radar_polar=self._r_radar_polar)
        self._next_id += 1
        self.tracks.append(t)

    def _apply_lifecycle_rules(self) -> None:
        for t in self.tracks:
            if t.state == TrackState.TENTATIVE:
                if t.hits >= self._hits_to_confirm:
                    t.state = TrackState.CONFIRMED
                elif t.age > MAX_TENTATIVE_AGE:
                    t.state = TrackState.DELETED
            elif t.state == TrackState.CONFIRMED:
                if t.misses >= self._max_misses:
                    t.state = TrackState.DELETED
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_mot_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute per-track RMSE and ID-switch count by assigning tracks to ground truth.

    For each frame, each live track is assigned to the nearest GT trajectory (within
    ``MATCH_GATE_M``) via Hungarian on the Euclidean cost matrix.  An ID switch is
    counted when a track's assigned GT trajectory changes between consecutive frames.

    Parameters
    ----------
    res
        Dict returned by :func:`run_mot_demo` (must contain ``true_trajectories``,
        ``track_history``, and ``n_frames``).

    Returns
    -------
    dict with keys ``id_switches`` (int), ``per_track_rmse`` (dict[int→float]),
    ``mean_rmse`` (float).
    """
    trajectories: List[NDArray[np.float64]] = res["true_trajectories"]
    track_history: List[Dict[int, NDArray[np.float64]]] = res["track_history"]
    n_gt = len(trajectories)

    per_frame_assignments: List[Dict[int, int]] = []

    for k, frame_state in enumerate(track_history):
        if not frame_state:
            per_frame_assignments.append({})
            continue

        track_ids = list(frame_state.keys())
        track_pos = np.array([frame_state[tid] for tid in track_ids])
        gt_pos = np.array([trajectories[j][k] for j in range(n_gt)])

        # Euclidean cost matrix (n_tracks × n_gt)
        diff = track_pos[:, None, :] - gt_pos[None, :, :]
        cost = np.sqrt((diff ** 2).sum(axis=2))

        row_ind, col_ind = linear_sum_assignment(cost)

        assignments: Dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < MATCH_GATE_M:
                assignments[track_ids[r]] = c
        per_frame_assignments.append(assignments)

    # Count ID switches: a track's GT assignment changes between consecutive frames
    id_switches = 0
    prev_assignments: Dict[int, int] = {}
    for frame_assignments in per_frame_assignments:
        for tid, gt_idx in frame_assignments.items():
            if tid in prev_assignments and prev_assignments[tid] != gt_idx:
                id_switches += 1
        prev_assignments.update(frame_assignments)

    # Per-track RMSE against the assigned GT
    track_errors: Dict[int, List[float]] = {}
    for k, (frame_state, frame_assignments) in enumerate(
        zip(track_history, per_frame_assignments)
    ):
        for tid, pos in frame_state.items():
            if tid not in frame_assignments:
                continue
            gt_idx = frame_assignments[tid]
            err = float(np.linalg.norm(pos - trajectories[gt_idx][k]))
            track_errors.setdefault(tid, []).append(err)

    per_track_rmse: Dict[int, float] = {
        tid: float(np.sqrt(np.mean(np.array(errs) ** 2)))
        for tid, errs in track_errors.items()
        if errs
    }

    # mean_rmse is restricted to tracks that were GT-matched for ≥ half the scenario
    # (filters out clutter tracks that happen to drift near a GT path for a few frames)
    min_matched_frames = len(track_history) // 2
    long_matched_rmse = [
        rmse for tid, rmse in per_track_rmse.items()
        if len(track_errors.get(tid, [])) >= min_matched_frames
    ]
    mean_rmse = float(np.mean(long_matched_rmse)) if long_matched_rmse else float("nan")

    return {
        "id_switches": id_switches,
        "per_track_rmse": per_track_rmse,
        "mean_rmse": mean_rmse,
    }


# ---------------------------------------------------------------------------
# End-to-end demo
# ---------------------------------------------------------------------------

def run_mot_demo(
    *,
    rng: Optional[np.random.Generator] = None,
    n_frames: int = 100,
) -> Dict[str, Any]:
    """
    Run the 3-target crossing scenario and return structured results.

    Returns
    -------
    dict with keys: ``true_trajectories``, ``track_history``,
    ``all_measurements``, ``measurements_world``, ``ellipse_snapshots``,
    ``metrics``, ``n_frames``.
    """
    rng = rng or default_rng(42)
    trajectories = mot_sim.make_crossing_scenario(n_frames)
    per_frame_meas = mot_sim.generate_multi_target_measurements(trajectories, rng)

    manager = TrackerManager(dt=1.0)

    track_history: List[Dict[int, NDArray[np.float64]]] = []
    # Confirmed status per frame — needed for state-based animation markers
    track_confirmed_history: List[Dict[int, bool]] = []
    # Ellipse snapshots for confirmed tracks every ELLIPSE_FRAME_STEP frames (static plot)
    ellipse_snapshots: List[Tuple[int, Dict[int, ekf.UncertaintyEllipse2D]]] = []
    # Per-frame ellipses for all confirmed tracks (animation uses most-recent)
    ellipse_all_frames: List[Dict[int, ekf.UncertaintyEllipse2D]] = []
    # World-frame measurement positions for plotting (polar → Cartesian)
    measurements_world: List[List[NDArray[np.float64]]] = []

    for k in range(n_frames):
        manager.predict_all()
        zs = [
            np.array([ret.range_m, ret.azimuth_rad], dtype=np.float64)
            for ret in per_frame_meas[k]
        ]
        manager.update(zs)

        frame_state: Dict[int, NDArray[np.float64]] = {
            t.track_id: t.get_position().copy() for t in manager.get_all_tracks()
        }
        track_history.append(frame_state)

        # Confirmed-state history for animation markers
        confirmed_ids = {t.track_id for t in manager.get_confirmed_tracks()}
        track_confirmed_history.append(
            {t.track_id: t.track_id in confirmed_ids for t in manager.get_all_tracks()}
        )

        # Per-frame ellipses for confirmed tracks (used by animation)
        ellipse_all_frames.append(
            {t.track_id: t.get_uncertainty_ellipse() for t in manager.get_confirmed_tracks()}
        )

        # Snapshot ellipses every ELLIPSE_FRAME_STEP frames (used by static plot)
        if k % ELLIPSE_FRAME_STEP == 0:
            ellipse_snapshots.append((k, dict(ellipse_all_frames[-1])))

        # Convert measurements to world for the animation/plot
        frame_world = [
            np.array([ret.range_m * math.cos(ret.azimuth_rad),
                      ret.range_m * math.sin(ret.azimuth_rad)])
            for ret in per_frame_meas[k]
        ]
        measurements_world.append(frame_world)

    res: Dict[str, Any] = {
        "true_trajectories": trajectories,
        "track_history": track_history,
        "track_confirmed_history": track_confirmed_history,
        "all_measurements": per_frame_meas,
        "measurements_world": measurements_world,
        "ellipse_snapshots": ellipse_snapshots,
        "ellipse_all_frames": ellipse_all_frames,
        "n_frames": n_frames,
    }
    res["metrics"] = compute_mot_metrics(res)
    return res


# ---------------------------------------------------------------------------
# Static plot
# ---------------------------------------------------------------------------

def plot_mot_results(
    res: Dict[str, Any], *, show: bool = True
) -> plt.Figure:
    """
    Trajectory plot with GT (grey dashed), estimated track paths (coloured by ID),
    and 95% uncertainty ellipses for confirmed tracks every 10 frames.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")

    # Ground truth
    for i, traj in enumerate(res["true_trajectories"]):
        lbl = "Ground truth" if i == 0 else None
        ax.plot(traj[:, 0], traj[:, 1], color="0.55", linewidth=1.5, linestyle="--",
                label=lbl, zorder=1)
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", color="0.4", s=40, zorder=2)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="s", color="0.4", s=40, zorder=2)

    # Estimated track paths
    track_paths: Dict[int, List[NDArray[np.float64]]] = {}
    for frame_state in res["track_history"]:
        for tid, pos in frame_state.items():
            track_paths.setdefault(tid, []).append(pos)

    color_map: Dict[int, Any] = {}
    for idx, (tid, positions) in enumerate(sorted(track_paths.items())):
        xy = np.array(positions)
        color = cmap(idx % 10)
        color_map[tid] = color
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.8,
                label=f"Track {tid}", zorder=3)
        ax.scatter(xy[-1, 0], xy[-1, 1], color=color, marker="^", s=60, zorder=4)

    # Uncertainty ellipses for confirmed tracks
    for _k, frame_ellipses in res.get("ellipse_snapshots", []):
        for tid, ell in frame_ellipses.items():
            color = color_map.get(tid, "C0")
            e = Ellipse(
                ell.center, width=ell.width, height=ell.height, angle=ell.angle_deg,
                facecolor="none", edgecolor=color, linewidth=0.7, alpha=0.55, zorder=2,
            )
            ax.add_patch(e)

    metrics = res.get("metrics", {})
    title_suffix = ""
    if metrics:
        title_suffix = (
            f"\nID switches: {metrics['id_switches']} | "
            f"mean RMSE: {metrics['mean_rmse']:.1f} m"
        )
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_title(
        "MOT — 3 crossing targets, GNN + Mahalanobis gate" + title_suffix
    )
    ax.set_xlim(-20, mot_sim.WORLD_SIZE_M + 20)
    ax.set_ylim(-20, mot_sim.WORLD_SIZE_M + 20)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    if show and "agg" not in str(plt.get_backend()).lower():
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

_TRAIL_FULL: int = 20   # frames drawn at full opacity
_TRAIL_ALPHA_DIM: float = 0.18
_TRAIL_ALPHA_BRIGHT: float = 0.92


def make_mot_animation(res: Dict[str, Any], *, fps: int = 15) -> Any:
    """
    Build a ``FuncAnimation`` that plays back the MOT demo frame by frame.

    Rendering features:
    - Fading trails: last ``_TRAIL_FULL`` frames at full alpha, earlier at dim alpha.
    - Raw measurements as red × markers so the data-association problem is visible.
    - State-based track tips: confirmed = filled triangle, tentative = hollow circle.
    - Per-frame uncertainty ellipses for confirmed tracks.
    """
    from matplotlib.animation import FuncAnimation

    trajectories = res["true_trajectories"]
    track_history = res["track_history"]
    confirmed_history = res.get("track_confirmed_history", [])
    measurements_world = res["measurements_world"]
    ellipse_all_frames = res.get("ellipse_all_frames", [])
    n_frames = res["n_frames"]

    cmap = plt.get_cmap("tab10")
    track_paths: Dict[int, List[NDArray[np.float64]]] = {}
    for frame_state in track_history:
        for tid, pos in frame_state.items():
            track_paths.setdefault(tid, []).append(pos)

    # Assign stable colors by first-appearance order
    sorted_ids = sorted(track_paths.keys())
    color_map: Dict[int, Any] = {tid: cmap(i % 10) for i, tid in enumerate(sorted_ids)}

    # Pre-build per-track position arrays indexed by frame (None when absent)
    pos_by_tid_frame: Dict[int, List[Optional[NDArray[np.float64]]]] = {
        tid: [None] * n_frames for tid in sorted_ids
    }
    for k, frame_state in enumerate(track_history):
        for tid, pos in frame_state.items():
            if tid in pos_by_tid_frame:
                pos_by_tid_frame[tid][k] = pos

    fig_anim, ax_anim = plt.subplots(figsize=(8, 7))
    ax_anim.set_xlim(-20, mot_sim.WORLD_SIZE_M + 20)
    ax_anim.set_ylim(-20, mot_sim.WORLD_SIZE_M + 20)
    ax_anim.set_aspect("equal")
    ax_anim.grid(True, alpha=0.25)

    def _draw_frame(k: int) -> None:
        ax_anim.cla()
        ax_anim.set_xlim(-20, mot_sim.WORLD_SIZE_M + 20)
        ax_anim.set_ylim(-20, mot_sim.WORLD_SIZE_M + 20)
        ax_anim.set_aspect("equal")
        ax_anim.grid(True, alpha=0.25)
        ax_anim.set_title(f"MOT demo — frame {k:03d} / {n_frames - 1}", fontsize=10)

        # Ground truth up to frame k (faint dashed)
        for traj in trajectories:
            ax_anim.plot(traj[:k+1, 0], traj[:k+1, 1],
                         color="0.65", linewidth=1.1, linestyle="--", zorder=1)

        # Raw measurements — x markers so clutter is distinguishable from tracks
        for pt in measurements_world[k]:
            ax_anim.scatter(pt[0], pt[1], color="crimson", marker="x", s=25,
                            alpha=0.75, linewidths=1.1, zorder=5)

        # Per-frame ellipses for confirmed tracks
        frame_ellipses = ellipse_all_frames[k] if k < len(ellipse_all_frames) else {}
        for tid, ell in frame_ellipses.items():
            color = color_map.get(tid, "C0")
            e = Ellipse(
                ell.center, width=ell.width, height=ell.height, angle=ell.angle_deg,
                facecolor="none", edgecolor=color, linewidth=0.9, alpha=0.55, zorder=2,
            )
            ax_anim.add_patch(e)

        # Track trails with fading and state-dependent tip markers
        confirmed_map = confirmed_history[k] if k < len(confirmed_history) else {}
        for tid in sorted_ids:
            # Collect (frame_index, position) for all frames where this track exists up to k
            present: List[Tuple[int, NDArray[np.float64]]] = [
                (j, pos_by_tid_frame[tid][j])
                for j in range(k + 1)
                if pos_by_tid_frame[tid][j] is not None
            ]
            if not present:
                continue

            color = color_map[tid]
            frames_present, positions = zip(*present)
            # Only draw a segment if track is visible at this frame
            if frames_present[-1] != k:
                continue

            xy = np.array(positions)
            n = len(xy)

            if n > 1:
                split = max(1, n - _TRAIL_FULL)
                # Older portion — dim
                if split > 1:
                    ax_anim.plot(xy[:split, 0], xy[:split, 1],
                                 color=color, linewidth=1.0,
                                 alpha=_TRAIL_ALPHA_DIM, zorder=3)
                # Recent portion — bright
                ax_anim.plot(xy[split-1:, 0], xy[split-1:, 1],
                             color=color, linewidth=2.0,
                             alpha=_TRAIL_ALPHA_BRIGHT, zorder=3)

            # Tip marker: confirmed → filled triangle, tentative → hollow circle
            tip_x, tip_y = float(xy[-1, 0]), float(xy[-1, 1])
            if confirmed_map.get(tid, False):
                ax_anim.scatter(tip_x, tip_y, color=color, marker="^",
                                s=65, zorder=4, alpha=_TRAIL_ALPHA_BRIGHT)
            else:
                ax_anim.scatter(tip_x, tip_y, color=color, marker="o",
                                s=55, facecolors="none", edgecolors=color,
                                linewidths=1.6, zorder=4, alpha=0.85)

    anim = FuncAnimation(
        fig_anim, _draw_frame, frames=n_frames,
        init_func=lambda: _draw_frame(0), interval=1000 // fps, blit=False,
    )
    return anim


def save_mot_animation(
    res: Dict[str, Any],
    path: str = "mot_demo.gif",
    *,
    fps: int = 15,
) -> str:
    """
    Save the MOT animation as a GIF (requires Pillow).
    Returns the path written.
    """
    anim = make_mot_animation(res, fps=fps)
    anim.save(path, writer="pillow", fps=fps)
    plt.close("all")
    return path


if __name__ == "__main__":
    import os
    out = run_mot_demo()
    m = out["metrics"]
    print(f"ID switches: {m['id_switches']}")
    print(f"Mean RMSE:   {m['mean_rmse']:.2f} m")
    for tid, rmse in sorted(m["per_track_rmse"].items()):
        print(f"  Track {tid}: {rmse:.2f} m")

    plot_mot_results(out, show=True)

    gif_path = "mot_demo.gif"
    print(f"\nSaving animation → {gif_path} …")
    save_mot_animation(out, gif_path, fps=12)
    print(f"Saved {os.path.getsize(gif_path) // 1024} KB")
