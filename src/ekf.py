"""
Linear position-state **Kalman** filter for 2D constant-velocity motion.

**Naming:** the class is :class:`KFTracker` (linear Kalman), not a true Extended Kalman
Filter. The file name ``ekf.py`` is kept as a short handle; a future
:class:`filterpy.kalman.ExtendedKalmanFilter` in native polar $h(x)$ can live
alongside or replace this when you want the project name "EKF" to match the math.

The transition uses :class:`filterpy.kalman.KalmanFilter` directly. Camera and radar
each supply Cartesian world $(x, y)$ with different $R$ — built in ``fusion`` from
pixel noise and from a tighter world variance for radar.

# INTERVIEW CRITICAL: True polar measurements are nonlinear in state; a full EKF
# would use h(x)=[range, az] and Jacobians, or a UKF. Linear KF in Cartesian
# with diagonal R is a bias-prone but interview-defensible first cut for short range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Kinematic and noise tuning — no magic numbers in logic below
# ---------------------------------------------------------------------------
# Time step (seconds) — must match simulation frame rate.
DEFAULT_DT_S: Final[float] = 1.0
# Discretized CV process noise: effective acceleration stdev (m/s^2) used to build Q.
# Physically, raising this says "the target can maneuver harder than a straight CV path".
# In filter behavior, a larger Q inflates the *predicted* P before a measurement, which
# generally increases the Kalman gain, i.e. more trust in the *new* data vs the prior
# (because you admit the state could have been perturbed in ways the straight CV model
# under-models). That is not a free lunch: a mis-sized Q that is too high also
# over-weights *bad* measurements (false alarms) because the state is no longer
# "smooth" a priori.
# INTERVIEW CRITICAL: Q trades off "how much can velocity change" vs measurement trust.
SIGMA_ACCEL_M_S2: Final[float] = 0.25
# Default initial position variance (m^2) and velocity variance (m^2/s^2) on the diagonal of P0.
INIT_POS_VAR_M2: Final[float] = 25.0
INIT_VEL_VAR_M2S2: Final[float] = 4.0
# Default camera: ~8 px 1-σ in ``utils`` → ~4 m 1-σ in world (METERS_PER_PIXEL=0.5) → 16 m².
R_CAMERA_DEFAULT_M2: Final[float] = 16.0
# Default radar: **tighter** than camera — ~2 m 1-σ per axis in world; keep in sync
# with :data:`fusion.RADAR_MEASUREMENT_VAR_M2`.
R_RADAR_DEFAULT_M2: Final[float] = 2.0**2
# 95% chi-square with 2 DOF (position plane); used for axis scaling of the ellipse
CHI2_95_2D: Final[float] = 5.991


@dataclass
class UncertaintyEllipse2D:
    """
    A convenience bundle for :class:`matplotlib.patches.Ellipse`.

    *center* is the estimated position; *width* and *height* are the full
    diameters of the 95% iso-probability contour in the position plane, and
    *angle_deg* is the rotation in degrees of the first eigenvector in screen/y-down plots.
    """

    center: Tuple[float, float]
    width: float
    height: float
    angle_deg: float


def _build_f_cv(dt: float) -> NDArray[np.float64]:
    """2D constant-velocity discrete-time F for state [x, y, vx, vy]."""
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _build_q_cv(dt: float, sigma_a: float) -> NDArray[np.float64]:
    """
    Q for random acceleration on x and y (same engine as classic CV): each axis
    gets white accelerations with power sigma_a^2, mapped through G that couples
    position and velocity. See Bar-Shalom / Rong Li style CV block structure.
    """
    g = np.array(
        [
            [0.5 * dt**2, 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=np.float64,
    )
    q = (sigma_a**2) * np.eye(2, dtype=np.float64)
    return g @ q @ g.T


def _h_position() -> NDArray[np.float64]:
    # z = H x, observe only the first two states (x, y)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


class KFTracker:
    """
    Track a single 2D target in world coordinates with a **linear** Kalman filter
    (constant-velocity process, position-only measurements for camera and radar).

    Why this exists: centralizes F, H, Q, and two sensor-specific $R$ matrices so
    fusion only passes measurements and optional overrides. Radar $R$ is set **smaller**
    than the camera in ``fusion`` to reflect the simulator’s meter-class range/azimuth
    noise when mapped to a diagonal world $R$ (a modeling choice, not a physical identity).

    Parameters
    ----------
    initial_state
        Initial 4-vector ``[x, y, vx, vy]`` in meters and m/s, typically at t=0
        or after a one-shot init from a detection.
    dt
        Frame period in seconds, must match simulation.
    r_camera, r_radar
        Diagonal (or full 2x2) measurement noise for camera and radar updates in
        world meters, after the camera pixel path has been linearized in ``utils``.

    The process noise (Q) is built from :data:`SIGMA_ACCEL_M_S2`; larger values
    allow faster maneuvers in the *internal* model and generally increase
    covariance between measurements (more trust in fresh measurements in steady state).
    """

    def __init__(
        self,
        initial_state: NDArray[np.float64],
        *,
        dt: float = DEFAULT_DT_S,
        r_camera: Optional[NDArray[np.float64]] = None,
        r_radar: Optional[NDArray[np.float64]] = None,
    ) -> None:
        self._dt: float = float(dt)
        self._kf: KalmanFilter = KalmanFilter(dim_x=4, dim_z=2)
        # Transition and process noise: CV model, explicit for clarity
        self._kf.F = _build_f_cv(self._dt)
        self._kf.Q = _build_q_cv(self._dt, SIGMA_ACCEL_M_S2)
        # H is shared for both sensor modalities: both return position
        self._kf.H = _h_position()
        self._kf.x = initial_state.reshape((4, 1)).astype(np.float64)
        self._kf.P = np.diag(
            [INIT_POS_VAR_M2, INIT_POS_VAR_M2, INIT_VEL_VAR_M2S2, INIT_VEL_VAR_M2S2]
        ).astype(np.float64)
        if r_camera is None:
            r_camera = R_CAMERA_DEFAULT_M2 * np.eye(2, dtype=np.float64)
        if r_radar is None:
            r_radar = R_RADAR_DEFAULT_M2 * np.eye(2, dtype=np.float64)
        self._R_camera: NDArray[np.float64] = np.asarray(r_camera, dtype=np.float64)
        self._R_radar: NDArray[np.float64] = np.asarray(r_radar, dtype=np.float64)
        # Pre-store default R for filterpy; update() can override
        self._kf.R = self._R_camera

    def predict(self) -> None:
        """
        Time-update one step: propagate mean and covariance with F and add Q.
        """
        # filterpy 1.4.x: predict with default F/Q already set
        self._kf.predict()

    def update_camera(self, z_xy: NDArray[np.float64], r_override: Optional[NDArray[np.float64]] = None) -> None:
        """
        Incorporate a noisy (x, y) measurement from a camera (already in world frame).

        Parameters
        ----------
        z_xy
            Shape (2,) position in meters, produced by :func:`utils.pixel_to_world_meters`.
        r_override
            Optional 2x2 full covariance; if None, use default camera R.
        """
        r = self._R_camera if r_override is None else r_override
        # Ensure measurement column vector for filterpy; R matches innovation covariance in world
        self._kf.update(z_xy.reshape((2, 1)), R=r, H=self._kf.H)

    def update_radar(self, z_xy: NDArray[np.float64], r_override: Optional[NDArray[np.float64]] = None) -> None:
        """
        Incorporate a noisy (x, y) measurement from radar, already converted from polar.

        A smaller R here models higher confidence in range/azimuth (after the
        polar-to-Cartesian step and any calibration).
        """
        r = self._R_radar if r_override is None else r_override
        self._kf.update(z_xy.reshape((2, 1)), R=r, H=self._kf.H)

    def get_state(self) -> NDArray[np.float64]:
        """
        Return the current posterior mean ``[x, y, vx, vy]`` as a 1D array.
        """
        return self._kf.x.reshape(4).copy()

    def get_covariance(self) -> NDArray[np.float64]:
        """Return the full 4x4 state covariance (posterior)."""
        return self._kf.P.copy()

    def get_position_covariance_2d(self) -> NDArray[np.float64]:
        """2x2 block of the covariance in the position subspace (x, y)."""
        return self._kf.P[0:2, 0:2].copy()

    def get_uncertainty_ellipse(self) -> UncertaintyEllipse2D:
        """
        Build a 95% two-dimensional Gaussian ellipse for the (x, y) marginals.
        # INTERVIEW CRITICAL: trace(P_xy) is not a geometric “area” but is often
        # used as a simple scalar spread metric in fusion comparisons.
        """
        p2 = self.get_position_covariance_2d()
        evals, evecs = np.linalg.eigh(p2)
        # 95% Mahalanobis ball in 2D: semi-axis a_i = sqrt(lambda_i * chi2_2,0.95)
        order = np.argsort(evals)[::-1]  # major eigenvalue first
        evals = np.maximum(evals[order], 1e-9)  # guard tiny numerical negatives
        semi = np.sqrt(evals * CHI2_95_2D)
        # Matplotlib ``Ellipse`` ``width``/``height`` are full axis lengths (diameters)
        width = float(2.0 * semi[0])
        height = float(2.0 * semi[1])
        v = evecs[:, order[0]]
        angle_rad = float(np.arctan2(v[1], v[0]))
        angle_deg = float(np.degrees(angle_rad))
        mean = self.get_state()[:2]
        return UncertaintyEllipse2D(
            center=(float(mean[0]), float(mean[1])),
            width=width,
            height=height,
            angle_deg=angle_deg,
        )
