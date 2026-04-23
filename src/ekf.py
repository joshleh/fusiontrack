"""
Kalman filter trackers for 2D constant-velocity single-target tracking.

Two classes live here side-by-side so the portfolio can compare them directly:

* :class:`KFTracker` — **linear** KF, both sensors use a Cartesian position H.
  Radar is pre-converted from polar before update, so R is a diagonal world
  covariance — a modeling approximation.

* :class:`EKFTracker` — **Extended** KF, radar measurements stay in native polar
  $(r, \\theta)$ and pass through a nonlinear ``h(x)`` with its 2×4 analytic
  Jacobian and an angle-normalizing residual.  Camera is still a linear H (it
  already lives in world Cartesian) so the two sensors genuinely differ.

# INTERVIEW CRITICAL: The KF/Cartesian-radar path is bias-prone when azimuth error
# converts to a large cross-range offset at long range.  The EKF uses the correct
# (r, θ) measurement model, so its innovation covariance S reflects actual polar
# geometry and the Kalman gain is better conditioned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Final, Optional, Tuple

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter, KalmanFilter
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
# Default radar (Cartesian KF): **tighter** than camera — ~2 m 1-σ per axis in world.
# Keep in sync with :data:`fusion.RADAR_MEASUREMENT_VAR_M2`.
R_RADAR_DEFAULT_M2: Final[float] = 2.0**2
# EKF polar radar R: physical noise in (range, azimuth) space.
# These MUST mirror radar_sim.RADAR_RANGE_NOISE_STD_M and RADAR_AZIMUTH_NOISE_STD_RAD
# (duplicated here to keep ekf.py import-free from sensor modules).
# INTERVIEW CRITICAL: R_polar = diag([σ_r², σ_θ²]) encodes *native* sensor error; the
# KF Cartesian approximation smears anisotropic polar noise into an isotropic world blob.
RADAR_RANGE_NOISE_STD_EKF_M: Final[float] = 3.0
RADAR_AZIMUTH_NOISE_STD_EKF_RAD: Final[float] = float(np.radians(0.5))
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


# ---------------------------------------------------------------------------
# EKF polar-radar measurement functions (module-level, stateless)
# ---------------------------------------------------------------------------

def _h_radar_polar(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Nonlinear measurement h(x) = [[range], [azimuth]] (2×1 column) from state [px, py, vx, vy].

    filterpy's EKF computes ``x += K @ (z - h(x))``; returning a (2,1) column vector
    keeps the shapes consistent with K (4,2) and z (2,1) so no spurious broadcasting occurs.

    Guard: clamp range at 1 µm to avoid undefined azimuth at the radar origin.
    """
    px, py = float(x.flat[0]), float(x.flat[1])
    r = max(math.hypot(px, py), 1e-6)
    az = math.atan2(py, px)
    return np.array([[r], [az]], dtype=np.float64)


def _h_radar_polar_jacobian(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Analytic 2×4 Jacobian of h(x) = [r, az] w.r.t. state [px, py, vx, vy].

    ∂r/∂px  = px/r        ∂r/∂py  = py/r       ∂r/∂vx = ∂r/∂vy = 0
    ∂az/∂px = -py/r²      ∂az/∂py = px/r²      ∂az/∂vx = ∂az/∂vy = 0

    # INTERVIEW CRITICAL: This is the linearization point — the Jacobian is
    # evaluated at the *predicted* state, so errors grow for large prediction
    # steps or highly curved paths.  A UKF avoids this by sigma-point integration.
    """
    px, py = float(x.flat[0]), float(x.flat[1])
    r2 = max(px ** 2 + py ** 2, 1e-12)
    r = math.sqrt(r2)
    return np.array(
        [
            [px / r,   py / r,  0.0, 0.0],
            [-py / r2, px / r2, 0.0, 0.0],
        ],
        dtype=np.float64,
    )


def _h_camera_linear(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear camera h(x) = [[px], [py]] (2×1 column) — consistent with KFTracker."""
    return np.array([[float(x.flat[0])], [float(x.flat[1])]], dtype=np.float64)


def _h_camera_jacobian(x: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: ARG001
    """Constant 2×4 Jacobian for the linear camera model."""
    return np.array(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float64,
    )


def _radar_polar_residual(
    z: NDArray[np.float64], hx: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Innovation y = z − h(x) with azimuth component wrapped to (−π, π].

    Both z and hx arrive as (2,1) column vectors; the result is also (2,1) so
    K @ y gives (4,1) and the state update preserves the (4,1) shape of self.x.

    Without wrapping, a target near azimuth ±π causes a ~2π innovation that
    drives the gain and state update wildly; wrapping makes the residual
    geometrically correct regardless of the ±π seam.

    # INTERVIEW CRITICAL: angle wrapping is mandatory for any bearing/heading
    # innovation — identical issue arises in GPS/compass fusion and SLAM.
    """
    y = np.subtract(z, hx).reshape(2, 1)
    y[1, 0] = (y[1, 0] + np.pi) % (2.0 * np.pi) - np.pi
    return y


class EKFTracker:
    """
    Single-target 2D tracker with a **true Extended Kalman Filter** for radar.

    Radar measurements arrive in native polar ``(range_m, azimuth_rad)`` and are
    processed through:

    * ``h(x) = [√(px²+py²),  atan2(py, px)]`` — nonlinear forward model
    * analytic 2×4 Jacobian evaluated at the predicted state
    * angle-normalizing residual to handle the ±π seam

    Camera measurements remain linear ``h(x) = [px, py]`` (they are already in
    world Cartesian after the pixel → meter mapping in ``utils``).

    The process model (constant-velocity F, Q) is identical to :class:`KFTracker`
    so the two classes are directly comparable in ``fusion.py``.

    Parameters
    ----------
    initial_state
        4-vector ``[x, y, vx, vy]`` in world metres / m s⁻¹.
    dt
        Frame period in seconds.
    r_camera
        2×2 camera measurement noise (world metres²).  Defaults to diagonal
        ``R_CAMERA_DEFAULT_M2 * I``.
    r_radar_polar
        2×2 polar measurement noise ``diag([σ_r², σ_θ²])``.  Defaults to the
        physical noise constants :data:`RADAR_RANGE_NOISE_STD_EKF_M` and
        :data:`RADAR_AZIMUTH_NOISE_STD_EKF_RAD`.
    """

    def __init__(
        self,
        initial_state: NDArray[np.float64],
        *,
        dt: float = DEFAULT_DT_S,
        r_camera: Optional[NDArray[np.float64]] = None,
        r_radar_polar: Optional[NDArray[np.float64]] = None,
    ) -> None:
        self._dt = float(dt)
        self._kf: ExtendedKalmanFilter = ExtendedKalmanFilter(dim_x=4, dim_z=2)
        self._kf.F = _build_f_cv(self._dt)
        self._kf.Q = _build_q_cv(self._dt, SIGMA_ACCEL_M_S2)
        self._kf.x = initial_state.reshape((4, 1)).astype(np.float64)
        self._kf.P = np.diag(
            [INIT_POS_VAR_M2, INIT_POS_VAR_M2, INIT_VEL_VAR_M2S2, INIT_VEL_VAR_M2S2]
        ).astype(np.float64)
        if r_camera is None:
            r_camera = R_CAMERA_DEFAULT_M2 * np.eye(2, dtype=np.float64)
        if r_radar_polar is None:
            r_radar_polar = np.diag(
                [RADAR_RANGE_NOISE_STD_EKF_M ** 2, RADAR_AZIMUTH_NOISE_STD_EKF_RAD ** 2]
            ).astype(np.float64)
        self._R_camera: NDArray[np.float64] = np.asarray(r_camera, dtype=np.float64)
        self._R_radar_polar: NDArray[np.float64] = np.asarray(r_radar_polar, dtype=np.float64)
        # filterpy EKF keeps self.R but we override it per update; set a placeholder
        self._kf.R = self._R_camera

    def predict(self) -> None:
        """Propagate mean and covariance one step via the linear CV model."""
        self._kf.predict()

    def update_camera(
        self, z_xy: NDArray[np.float64], r_override: Optional[NDArray[np.float64]] = None
    ) -> None:
        """
        Incorporate a camera position measurement ``[x_world, y_world]`` (metres).

        Uses the linear camera Jacobian — equivalent to the standard KF update.
        z is passed as a (2,1) column to keep K@(z-h(x)) shapes consistent with (4,1) self.x.
        """
        r = self._R_camera if r_override is None else r_override
        self._kf.update(
            z_xy.reshape(2, 1).astype(np.float64),
            HJacobian=_h_camera_jacobian,
            Hx=_h_camera_linear,
            R=r,
        )

    def update_radar_polar(
        self,
        z_polar: NDArray[np.float64],
        r_override: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """
        Incorporate a **polar** radar measurement ``[range_m, azimuth_rad]``.

        Do **not** pre-convert to Cartesian — the whole point of the EKF is to
        use the native sensor frame so R stays physically meaningful.
        z is passed as a (2,1) column; _radar_polar_residual also returns (2,1).
        """
        r = self._R_radar_polar if r_override is None else r_override
        self._kf.update(
            z_polar.reshape(2, 1).astype(np.float64),
            HJacobian=_h_radar_polar_jacobian,
            Hx=_h_radar_polar,
            R=r,
            residual=_radar_polar_residual,
        )

    def get_state(self) -> NDArray[np.float64]:
        """Posterior mean ``[x, y, vx, vy]`` as a 1D array."""
        return self._kf.x.reshape(4).copy()

    def get_covariance(self) -> NDArray[np.float64]:
        """Full 4×4 state covariance (posterior)."""
        return self._kf.P.copy()

    def get_position_covariance_2d(self) -> NDArray[np.float64]:
        """2×2 position-plane covariance block."""
        return self._kf.P[0:2, 0:2].copy()

    def get_uncertainty_ellipse(self) -> UncertaintyEllipse2D:
        """95% confidence ellipse in world Cartesian — same helper as KFTracker."""
        p2 = self.get_position_covariance_2d()
        evals, evecs = np.linalg.eigh(p2)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 1e-9)
        semi = np.sqrt(evals * CHI2_95_2D)
        width = float(2.0 * semi[0])
        height = float(2.0 * semi[1])
        v = evecs[:, order[0]]
        angle_deg = float(np.degrees(np.arctan2(v[1], v[0])))
        mean = self.get_state()[:2]
        return UncertaintyEllipse2D(
            center=(float(mean[0]), float(mean[1])),
            width=width,
            height=height,
            angle_deg=angle_deg,
        )
