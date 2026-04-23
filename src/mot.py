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

# A tentative track must collect this many consecutive hits to become confirmed.
HITS_TO_CONFIRM: Final[int] = 2
# A confirmed track is deleted after this many consecutive frames with no measurement.
MAX_MISSES_BEFORE_DELETE: Final[int] = 3
# A tentative track that never gets a second hit is pruned after this many frames.
MAX_TENTATIVE_AGE: Final[int] = 3

# Sentinel cost for gated (infeasible) assignments so Hungarian still runs cleanly.
_LARGE_COST: Final[float] = 1e9


class TrackState(Enum):
    TENTATIVE = auto()   # recently born; awaiting confirmation
    CONFIRMED = auto()   # sustained hit stream; report to downstream consumers
    DELETED = auto()     # to be pruned at end of this cycle


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

    # ------------------------------------------------------------------
    # Time update
    # ------------------------------------------------------------------
    def predict(self) -> None:
        self._tracker.predict()
        self.age += 1

    # ------------------------------------------------------------------
    # Measurement update and miss bookkeeping
    # ------------------------------------------------------------------
    def update(self, z_polar: NDArray[np.float64]) -> None:
        self._tracker.update_radar_polar(z_polar)
        self.hits += 1
        self.misses = 0

    def miss(self) -> None:
        self.misses += 1

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
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

    Parameters
    ----------
    dt
        Frame period in seconds; passed to each new Track.
    hits_to_confirm
        Consecutive hit count to promote TENTATIVE → CONFIRMED.
    max_misses
        Consecutive miss count to delete a CONFIRMED track.
    gate_chi2
        Mahalanobis gate threshold (chi-square, 2 DOF).
    r_radar_polar
        2×2 polar R matrix shared across all tracks.  Defaults to the physical
        noise parameters in :mod:`ekf`.
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
        """Time-update every live track."""
        for t in self.tracks:
            t.predict()

    def update(self, measurements: Sequence[NDArray[np.float64]]) -> None:
        """
        Associate ``measurements`` to existing tracks, then manage lifecycle.

        Parameters
        ----------
        measurements
            Sequence of ``[range_m, azimuth_rad]`` polar arrays for this frame.
            The order carries no identity information (shuffled by the sensor).
        """
        measurements = list(measurements)
        n_tracks = len(self.tracks)
        n_meas = len(measurements)

        # Edge cases: no tracks or no measurements
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

        # Only accept assignments within the gate (others get _LARGE_COST sentinel)
        assigned_tracks: set = set()
        assigned_meas: set = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < _LARGE_COST:
                self.tracks[r].update(measurements[c])
                assigned_tracks.add(r)
                assigned_meas.add(c)

        # Unmatched tracks → missed
        for i, t in enumerate(self.tracks):
            if i not in assigned_tracks:
                t.miss()

        # Unmatched measurements → birth new tentative track
        for j, z in enumerate(measurements):
            if j not in assigned_meas:
                self._birth(z)

        self._apply_lifecycle_rules()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_confirmed_tracks(self) -> List[Track]:
        """Return only confirmed (non-tentative, non-deleted) tracks."""
        return [t for t in self.tracks if t.state == TrackState.CONFIRMED]

    def get_all_tracks(self) -> List[Track]:
        """Return all live tracks (tentative + confirmed)."""
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
        Gating here (not just post-assignment) prevents the Hungarian from wasting
        capacity on clearly infeasible pairs.
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
        """
        d² = y^T S^{-1} y in polar measurement space (angle-normalised).

        Returns ``None`` if S is singular (degenerate state — skip this pair).
        """
        try:
            y, S = track.compute_innovation_polar(z_polar)
            y_flat = y.ravel()
            return float(y_flat @ np.linalg.inv(S) @ y_flat)
        except np.linalg.LinAlgError:
            return None

    def _birth(self, z_polar: NDArray[np.float64]) -> None:
        """
        Spawn a TENTATIVE track from an unmatched polar measurement.

        Initial velocity is zero with high uncertainty (INIT_VEL_VAR_M2S2 from ekf).
        Two confirmed measurements are enough to estimate velocity; one is not.
        """
        r, az = float(z_polar[0]), float(z_polar[1])
        x_w = r * math.cos(az)
        y_w = r * math.sin(az)
        x0 = np.array([x_w, y_w, 0.0, 0.0], dtype=np.float64)
        t = Track(x0, self._next_id, dt=self._dt, r_radar_polar=self._r_radar_polar)
        self._next_id += 1
        self.tracks.append(t)

    def _apply_lifecycle_rules(self) -> None:
        """Promote tentative → confirmed, mark excess misses → deleted, then prune."""
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
    dict with keys:
      ``true_trajectories`` — list of (n_frames, 2) arrays
      ``track_history``     — per-frame list of {track_id: (2,) position}
      ``all_measurements``  — per-frame list of PolarRadarReturn objects (shuffled)
      ``n_frames``          — frame count
    """
    rng = rng or default_rng(42)
    trajectories = mot_sim.make_crossing_scenario(n_frames)
    per_frame_meas = mot_sim.generate_multi_target_measurements(trajectories, rng)

    manager = TrackerManager(dt=1.0)

    track_history: List[Dict[int, NDArray[np.float64]]] = []
    for k in range(n_frames):
        manager.predict_all()
        zs = [
            np.array([ret.range_m, ret.azimuth_rad], dtype=np.float64)
            for ret in per_frame_meas[k]
        ]
        manager.update(zs)
        # Record positions of all live tracks (tentative + confirmed)
        frame_state: Dict[int, NDArray[np.float64]] = {}
        for t in manager.get_all_tracks():
            frame_state[t.track_id] = t.get_position().copy()
        track_history.append(frame_state)

    return {
        "true_trajectories": trajectories,
        "track_history": track_history,
        "all_measurements": per_frame_meas,
        "n_frames": n_frames,
    }


def plot_mot_results(
    res: Dict[str, Any], *, show: bool = True
) -> plt.Figure:
    """
    Plot the MOT demo: true trajectories (grey dashed) + estimated track paths
    (coloured by track ID, solid once confirmed, dotted while tentative).
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")

    # True trajectories — dashed grey for reference
    for i, traj in enumerate(res["true_trajectories"]):
        ax.plot(
            traj[:, 0], traj[:, 1],
            color="0.55", linewidth=1.5, linestyle="--",
            label=f"Truth T{i + 1}" if i == 0 else None,
            zorder=1,
        )
        ax.scatter(traj[0, 0], traj[0, 1], marker="o", color="0.4", s=40, zorder=2)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker="s", color="0.4", s=40, zorder=2)

    # Estimated track paths — group frames by track_id
    track_paths: Dict[int, List[NDArray[np.float64]]] = {}
    for frame_state in res["track_history"]:
        for tid, pos in frame_state.items():
            track_paths.setdefault(tid, []).append(pos)

    for idx, (tid, positions) in enumerate(sorted(track_paths.items())):
        xy = np.array(positions)
        color = cmap(idx % 10)
        ax.plot(
            xy[:, 0], xy[:, 1],
            color=color, linewidth=1.8,
            label=f"Track {tid}",
            zorder=3,
        )
        ax.scatter(xy[-1, 0], xy[-1, 1], color=color, marker="^", s=60, zorder=4)

    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_title(
        "Multi-Object Tracker — 3 crossing targets, GNN + Mahalanobis gate\n"
        "(grey dashed = ground truth, coloured = estimated track)"
    )
    ax.set_xlim(-20, mot_sim.WORLD_SIZE_M + 20)
    ax.set_ylim(-20, mot_sim.WORLD_SIZE_M + 20)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    if show and "agg" not in str(plt.get_backend()).lower():
        plt.show()
    return fig


if __name__ == "__main__":
    out = run_mot_demo()
    plot_mot_results(out, show=True)
